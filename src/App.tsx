import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

type ScaleMode = 'major' | 'minor'
type Waveform = 'sine' | 'triangle' | 'sawtooth'

type HandPoint = {
  x: number
  y: number
}

const CHORDS_MAJOR = {
  C: ['C4', 'E4', 'G4'],
  Dm: ['D4', 'F4', 'A4'],
  Em: ['E4', 'G4', 'B4'],
  F: ['F4', 'A4', 'C5'],
  G: ['G4', 'B4', 'D5'],
  Am: ['A4', 'C5', 'E5'],
  Bdim: ['B4', 'D5', 'F5'],
} as const

const CHORDS_MINOR = {
  C: ['C4', 'Eb4', 'G4'],
  Dm: ['D4', 'F4', 'A4'],
  Em: ['E4', 'G4', 'Bb4'],
  F: ['F4', 'Ab4', 'C5'],
  G: ['G4', 'Bb4', 'D5'],
  Am: ['A4', 'C5', 'E5'],
  Bdim: ['B4', 'D5', 'F5'],
} as const

const CHORD_NAMES = ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim'] as const

const VIDEO_WIDTH = 960
const VIDEO_HEIGHT = 540

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value))
}

function quantizeToChord(y: number, notes: string[]): string {
  const safeY = clamp01(y)
  const index = Math.min(notes.length - 1, Math.floor((1 - safeY) * notes.length))
  return notes[index]
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const animationRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const synthRef = useRef<Tone.PolySynth | null>(null)
  const filterRef = useRef<Tone.Filter | null>(null)
  const reverbRef = useRef<Tone.Reverb | null>(null)
  const delayRef = useRef<Tone.FeedbackDelay | null>(null)

  const [audioReady, setAudioReady] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [scaleMode, setScaleMode] = useState<ScaleMode>('major')
  const [waveform, setWaveform] = useState<Waveform>('triangle')
  const [reverbWet, setReverbWet] = useState(0.35)
  const [delayWet, setDelayWet] = useState(0.25)
  const [selectedChord, setSelectedChord] = useState<(typeof CHORD_NAMES)[number]>('C')
  const [leftHand, setLeftHand] = useState<HandPoint | null>(null)
  const [rightHand, setRightHand] = useState<HandPoint | null>(null)

  const lastTriggeredNote = useRef<string | null>(null)
  const lastPlayTime = useRef(0)

  const leftZone = useMemo(() => ({ x: VIDEO_WIDTH * 0.25, y: VIDEO_HEIGHT * 0.5, r: VIDEO_HEIGHT * 0.32 }), [])
  const rightZone = useMemo(() => ({ x: VIDEO_WIDTH * 0.75, y: VIDEO_HEIGHT * 0.5, r: VIDEO_HEIGHT * 0.32 }), [])

  const currentChord = useMemo(() => {
    const bank = scaleMode === 'major' ? CHORDS_MAJOR : CHORDS_MINOR
    return bank[selectedChord]
  }, [selectedChord, scaleMode])

  const pointInZone = useCallback((point: HandPoint, zone: { x: number; y: number; r: number }) => {
    return Math.hypot(point.x - zone.x, point.y - zone.y) <= zone.r
  }, [])

  const initAudio = useCallback(async () => {
    if (audioReady) return
    await Tone.start()

    const filter = new Tone.Filter(1200, 'lowpass')
    const chorus = new Tone.Chorus(4, 2.5, 0.4).start()
    const delay = new Tone.FeedbackDelay('8n', 0.25)
    const reverb = new Tone.Reverb({ decay: 3, wet: reverbWet })
    const synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: waveform },
      envelope: { attack: 0.08, decay: 0.15, sustain: 0.5, release: 0.5 },
    })

    synth.chain(filter, chorus, delay, reverb, Tone.Destination)
    delay.wet.value = delayWet

    synthRef.current = synth
    filterRef.current = filter
    reverbRef.current = reverb
    delayRef.current = delay

    setAudioReady(true)
  }, [audioReady, delayWet, reverbWet, waveform])

  useEffect(() => {
    if (synthRef.current) {
      synthRef.current.set({ oscillator: { type: waveform } })
    }
  }, [waveform])

  useEffect(() => {
    if (reverbRef.current) reverbRef.current.wet.value = reverbWet
  }, [reverbWet])

  useEffect(() => {
    if (delayRef.current) delayRef.current.wet.value = delayWet
  }, [delayWet])

  const initHandTracking = useCallback(async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
    )

    const handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      },
      numHands: 2,
      runningMode: 'VIDEO',
    })

    handLandmarkerRef.current = handLandmarker
    return handLandmarker
  }, [])

  const drawOverlay = useCallback((landmarks: NormalizedLandmark[][]) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Instrument zones.
    ctx.strokeStyle = 'rgba(94, 234, 212, 0.45)'
    ctx.lineWidth = 4
    ctx.beginPath()
    ctx.arc(leftZone.x, leftZone.y, leftZone.r, 0, Math.PI * 2)
    ctx.stroke()
    ctx.beginPath()
    ctx.arc(rightZone.x, rightZone.y, rightZone.r, 0, Math.PI * 2)
    ctx.stroke()

    landmarks.forEach((hand) => {
      const wrist = hand[0]
      const x = wrist.x * canvas.width
      const y = wrist.y * canvas.height

      ctx.fillStyle = 'rgba(250, 204, 21, 0.9)'
      ctx.beginPath()
      ctx.arc(x, y, 10, 0, Math.PI * 2)
      ctx.fill()
    })
  }, [leftZone.r, leftZone.x, leftZone.y, rightZone.r, rightZone.x, rightZone.y])

  const runDetectionFrame = useCallback(async () => {
    const video = videoRef.current
    const tracker = handLandmarkerRef.current
    if (!video || !tracker) return

    const result = tracker.detectForVideo(video, performance.now())
    const landmarks = result.landmarks ?? []

    drawOverlay(landmarks)

    const points = landmarks
      .map((hand) => ({ x: hand[0].x * VIDEO_WIDTH, y: hand[0].y * VIDEO_HEIGHT }))
      .sort((a, b) => a.x - b.x)

    const left = points[0] ?? null
    const right = points[1] ?? null
    setLeftHand(left)
    setRightHand(right)

    if (left && pointInZone(left, leftZone)) {
      const normalized = clamp01((left.x - (leftZone.x - leftZone.r)) / (leftZone.r * 2))
      const chordIndex = Math.min(CHORD_NAMES.length - 1, Math.floor(normalized * CHORD_NAMES.length))
      setSelectedChord(CHORD_NAMES[chordIndex])
    }

    if (right && pointInZone(right, rightZone) && synthRef.current && filterRef.current) {
      const note = quantizeToChord(right.y / VIDEO_HEIGHT, currentChord as unknown as string[])
      const rightXNorm = clamp01((right.x - (rightZone.x - rightZone.r)) / (rightZone.r * 2))
      filterRef.current.frequency.value = 250 + rightXNorm * 4200

      const now = performance.now()
      // Avoid endless retriggering: note change + short throttle window.
      if (note !== lastTriggeredNote.current || now - lastPlayTime.current > 180) {
        synthRef.current.triggerAttackRelease(note, '16n')
        lastTriggeredNote.current = note
        lastPlayTime.current = now
      }
    }

    animationRef.current = requestAnimationFrame(runDetectionFrame)
  }, [currentChord, drawOverlay, leftZone, pointInZone, rightZone])

  const enableCamera = useCallback(async () => {
    if (cameraReady) return
    const video = videoRef.current
    if (!video) return

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
      audio: false,
    })
    video.srcObject = stream
    streamRef.current = stream

    await video.play()
    await initHandTracking()

    setCameraReady(true)
    animationRef.current = requestAnimationFrame(runDetectionFrame)
  }, [cameraReady, initHandTracking, runDetectionFrame])

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      handLandmarkerRef.current?.close()
      synthRef.current?.dispose()
      filterRef.current?.dispose()
      reverbRef.current?.dispose()
      delayRef.current?.dispose()
    }
  }, [])

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 p-6 text-slate-100">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Webcam Omnichord</h1>
        <p className="text-sm text-slate-300">Controla acordes con la mano izquierda y notas con la derecha.</p>
      </header>

      <section className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <div className="relative overflow-hidden rounded-2xl border border-slate-700 bg-black/30 shadow-2xl shadow-cyan-900/20">
          <video ref={videoRef} className="h-auto w-full scale-x-[-1]" width={VIDEO_WIDTH} height={VIDEO_HEIGHT} muted playsInline />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 h-full w-full scale-x-[-1]"
            width={VIDEO_WIDTH}
            height={VIDEO_HEIGHT}
          />
        </div>

        <aside className="space-y-4 rounded-2xl border border-slate-700 bg-slate-900/60 p-4">
          <button onClick={initAudio} className="w-full rounded-xl bg-emerald-500 px-4 py-2 font-semibold text-slate-950">
            Start Audio
          </button>
          <button onClick={enableCamera} className="w-full rounded-xl bg-cyan-400 px-4 py-2 font-semibold text-slate-950">
            Enable Camera
          </button>

          <label className="block text-sm">
            Escala
            <select
              value={scaleMode}
              onChange={(e) => setScaleMode(e.target.value as ScaleMode)}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 p-2"
            >
              <option value="major">Major</option>
              <option value="minor">Minor</option>
            </select>
          </label>

          <label className="block text-sm">
            Waveform
            <select
              value={waveform}
              onChange={(e) => setWaveform(e.target.value as Waveform)}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-800 p-2"
            >
              <option value="sine">sine</option>
              <option value="triangle">triangle</option>
              <option value="sawtooth">sawtooth</option>
            </select>
          </label>

          <label className="block text-sm">
            Reverb amount: {reverbWet.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={reverbWet}
              onChange={(e) => setReverbWet(Number(e.target.value))}
              className="mt-1 w-full"
            />
          </label>

          <label className="block text-sm">
            Delay amount: {delayWet.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={delayWet}
              onChange={(e) => setDelayWet(Number(e.target.value))}
              className="mt-1 w-full"
            />
          </label>

          <div className="rounded-xl border border-slate-700 bg-slate-800/70 p-3 text-sm">
            <p>Acorde actual: <strong>{selectedChord}</strong></p>
            <p>Mano izquierda: {leftHand ? `${leftHand.x.toFixed(0)}, ${leftHand.y.toFixed(0)}` : '—'}</p>
            <p>Mano derecha: {rightHand ? `${rightHand.x.toFixed(0)}, ${rightHand.y.toFixed(0)}` : '—'}</p>
          </div>
        </aside>
      </section>
    </main>
  )
}
