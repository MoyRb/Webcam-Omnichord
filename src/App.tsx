import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { FilesetResolver, HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'

type HandPoint = {
  x: number
  y: number
}

const ROOTS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] as const
const QUALITIES = ['maj', 'min', '7', 'maj7', 'm7', 'sus4', 'dim'] as const

type Root = (typeof ROOTS)[number]
type Quality = (typeof QUALITIES)[number]

const VIDEO_WIDTH = 960
const VIDEO_HEIGHT = 540
const REPLAY_COOLDOWN_MS = 450
const BASE_OCTAVE = 4

const QUALITY_INTERVALS: Record<Quality, number[]> = {
  maj: [0, 4, 7],
  min: [0, 3, 7],
  '7': [0, 4, 7, 10],
  maj7: [0, 4, 7, 11],
  m7: [0, 3, 7, 10],
  sus4: [0, 5, 7],
  dim: [0, 3, 6],
}

const OMNICHORD_POLY_OPTIONS: Tone.PolySynthOptions<Tone.SynthOptions> = {
  volume: -9,
  options: {
    oscillator: { type: 'triangle8' },
    envelope: { attack: 0.06, decay: 0.2, sustain: 0.55, release: 1.1 },
  },
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value))
}

function pointToSegmentIndex(point: HandPoint, center: { x: number; y: number }, segmentCount: number) {
  const angle = Math.atan2(point.y - center.y, point.x - center.x)
  const normalized = (angle + Math.PI * 2) % (Math.PI * 2)
  const index = Math.floor((normalized / (Math.PI * 2)) * segmentCount)
  return Math.min(segmentCount - 1, Math.max(0, index))
}

function buildChord(root: Root, quality: Quality, octave = BASE_OCTAVE): string[] {
  const rootMidi = Tone.Frequency(`${root}${octave}`).toMidi()
  return QUALITY_INTERVALS[quality].map((interval) => Tone.Frequency(rootMidi + interval, 'midi').toNote())
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const animationRef = useRef<number | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const synthRef = useRef<Tone.PolySynth | null>(null)
  const chorusRef = useRef<Tone.Chorus | null>(null)
  const filterRef = useRef<Tone.Filter | null>(null)
  const reverbRef = useRef<Tone.Reverb | null>(null)
  const delayRef = useRef<Tone.FeedbackDelay | null>(null)

  const [audioStarted, setAudioStarted] = useState(false)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [reverbWet, setReverbWet] = useState(0.35)
  const [delayWet, setDelayWet] = useState(0.25)
  const [selectedRoot, setSelectedRoot] = useState<Root | null>(null)
  const [selectedQuality, setSelectedQuality] = useState<Quality | null>(null)
  const [leftHand, setLeftHand] = useState<HandPoint | null>(null)
  const [rightHand, setRightHand] = useState<HandPoint | null>(null)
  const [handsDetected, setHandsDetected] = useState(0)
  const [debugMessage, setDebugMessage] = useState('Esperando interacción...')

  const lastTriggeredChord = useRef<string>('')
  const lastPlayTime = useRef(0)

  const leftZone = useMemo(() => ({ x: VIDEO_WIDTH * 0.25, y: VIDEO_HEIGHT * 0.5, r: VIDEO_HEIGHT * 0.32 }), [])
  const rightZone = useMemo(() => ({ x: VIDEO_WIDTH * 0.75, y: VIDEO_HEIGHT * 0.5, r: VIDEO_HEIGHT * 0.32 }), [])

  const currentChordLabel = useMemo(() => {
    if (!selectedRoot || !selectedQuality) return '—'
    return `${selectedRoot}${selectedQuality}`
  }, [selectedQuality, selectedRoot])

  const currentChordNotes = useMemo(() => {
    if (!selectedRoot || !selectedQuality) return []
    return buildChord(selectedRoot, selectedQuality)
  }, [selectedQuality, selectedRoot])

  const pointInZone = useCallback((point: HandPoint, zone: { x: number; y: number; r: number }) => {
    return Math.hypot(point.x - zone.x, point.y - zone.y) <= zone.r
  }, [])

  const initAudio = useCallback(async () => {
    if (audioStarted) return
    await Tone.start()

    const filter = new Tone.Filter(1200, 'lowpass')
    const chorus = new Tone.Chorus(1.8, 2.2, 0.35).start()
    const delay = new Tone.FeedbackDelay('8n', 0.25)
    const reverb = new Tone.Reverb({ decay: 3, wet: reverbWet })
    const synth = new Tone.PolySynth(Tone.Synth, OMNICHORD_POLY_OPTIONS)

    synth.chain(filter, chorus, delay, reverb, Tone.Destination)
    delay.wet.value = delayWet

    synthRef.current = synth
    chorusRef.current = chorus
    filterRef.current = filter
    reverbRef.current = reverb
    delayRef.current = delay

    setAudioStarted(true)
    setDebugMessage('Audio inicializado correctamente.')
  }, [audioStarted, delayWet, reverbWet])

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

  const drawSegmentedCircle = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      zone: { x: number; y: number; r: number },
      labels: readonly string[],
      activeLabel: string | null,
    ) => {
      const step = (Math.PI * 2) / labels.length
      labels.forEach((label, index) => {
        const start = index * step
        const end = start + step
        const isActive = activeLabel === label

        ctx.beginPath()
        ctx.moveTo(zone.x, zone.y)
        ctx.arc(zone.x, zone.y, zone.r, start, end)
        ctx.closePath()
        ctx.fillStyle = isActive ? 'rgba(16, 185, 129, 0.45)' : 'rgba(14, 116, 144, 0.2)'
        ctx.fill()
        ctx.strokeStyle = 'rgba(94, 234, 212, 0.65)'
        ctx.lineWidth = 2
        ctx.stroke()

        const labelAngle = start + step / 2
        const labelRadius = zone.r * 0.67
        const tx = zone.x + Math.cos(labelAngle) * labelRadius
        const ty = zone.y + Math.sin(labelAngle) * labelRadius

        ctx.save()
        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 17px sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.fillText(label, tx, ty)
        ctx.restore()
      })
    },
    [],
  )

  const drawOverlay = useCallback(
    (landmarks: NormalizedLandmark[][]) => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      drawSegmentedCircle(ctx, leftZone, ROOTS, selectedRoot)
      drawSegmentedCircle(ctx, rightZone, QUALITIES, selectedQuality)

      landmarks.forEach((hand) => {
        const wrist = hand[0]
        const x = wrist.x * canvas.width
        const y = wrist.y * canvas.height

        ctx.fillStyle = 'rgba(250, 204, 21, 0.9)'
        ctx.beginPath()
        ctx.arc(x, y, 10, 0, Math.PI * 2)
        ctx.fill()
      })
    },
    [drawSegmentedCircle, leftZone, rightZone, selectedQuality, selectedRoot],
  )

  const runDetectionFrame = useCallback(async () => {
    const video = videoRef.current
    const tracker = handLandmarkerRef.current
    if (!video || !tracker) return

    const result = tracker.detectForVideo(video, performance.now())
    const landmarks = result.landmarks ?? []
    setHandsDetected(landmarks.length)

    drawOverlay(landmarks)

    const points = landmarks
      .map((hand) => ({ x: hand[0].x * VIDEO_WIDTH, y: hand[0].y * VIDEO_HEIGHT }))
      .sort((a, b) => a.x - b.x)

    const left = points[0] ?? null
    const right = points[1] ?? null
    setLeftHand(left)
    setRightHand(right)

    if (left && pointInZone(left, leftZone)) {
      const rootIndex = pointToSegmentIndex(left, leftZone, ROOTS.length)
      setSelectedRoot(ROOTS[rootIndex])
    }

    if (right && pointInZone(right, rightZone)) {
      const qualityIndex = pointToSegmentIndex(right, rightZone, QUALITIES.length)
      setSelectedQuality(QUALITIES[qualityIndex])
    }

    if (right && pointInZone(right, rightZone) && filterRef.current) {
      const rightXNorm = clamp01((right.x - (rightZone.x - rightZone.r)) / (rightZone.r * 2))
      filterRef.current.frequency.value = 250 + rightXNorm * 4200
    }

    if (audioStarted && selectedRoot && selectedQuality && synthRef.current) {
      const notes = buildChord(selectedRoot, selectedQuality)
      const chordToken = `${selectedRoot}${selectedQuality}:${notes.join('-')}`

      const now = performance.now()
      if (chordToken !== lastTriggeredChord.current || now - lastPlayTime.current > REPLAY_COOLDOWN_MS) {
        synthRef.current.triggerAttackRelease(notes, '8n')
        lastTriggeredChord.current = chordToken
        lastPlayTime.current = now
        setDebugMessage(`Trigger: ${selectedRoot}${selectedQuality} -> ${notes.join(', ')}`)
      }
    }

    animationRef.current = requestAnimationFrame(runDetectionFrame)
  }, [audioStarted, drawOverlay, leftZone, pointInZone, rightZone, selectedQuality, selectedRoot])

  const enableCamera = useCallback(async () => {
    if (cameraReady) return
    const video = videoRef.current
    if (!video) return

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT, facingMode: 'user' },
        audio: false,
      })
      video.srcObject = stream
      streamRef.current = stream

      await video.play()
      await initHandTracking()

      setCameraError(null)
      setCameraReady(true)
      animationRef.current = requestAnimationFrame(runDetectionFrame)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'No se pudo acceder a la cámara.'
      setCameraError(`Error de cámara: ${message}`)
    }
  }, [cameraReady, initHandTracking, runDetectionFrame])

  useEffect(() => {
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      handLandmarkerRef.current?.close()
      synthRef.current?.dispose()
      chorusRef.current?.dispose()
      filterRef.current?.dispose()
      reverbRef.current?.dispose()
      delayRef.current?.dispose()
    }
  }, [])

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 bg-slate-950 p-6 text-slate-100">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Webcam Omnichord</h1>
        <p className="text-sm text-slate-300">Izquierda: raíz del acorde. Derecha: calidad del acorde.</p>
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

          {!audioStarted ? (
            <div className="rounded-xl border border-amber-400/50 bg-amber-500/10 p-3 text-sm text-amber-200">
              Primero presiona Start Audio
            </div>
          ) : null}

          {cameraError ? (
            <div className="rounded-xl border border-rose-400/50 bg-rose-500/10 p-3 text-sm text-rose-200">{cameraError}</div>
          ) : null}

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
            <p>
              Acorde actual: <strong>{currentChordLabel}</strong>
            </p>
            <p>Notas: {currentChordNotes.length ? currentChordNotes.join(', ') : '—'}</p>
            <p>Mano izquierda: {leftHand ? `${leftHand.x.toFixed(0)}, ${leftHand.y.toFixed(0)}` : '—'}</p>
            <p>Mano derecha: {rightHand ? `${rightHand.x.toFixed(0)}, ${rightHand.y.toFixed(0)}` : '—'}</p>
          </div>

          <div className="rounded-xl border border-cyan-400/40 bg-slate-950/80 p-3 font-mono text-xs text-cyan-100">
            <p>audioStarted: {String(audioStarted)}</p>
            <p>root seleccionada: {selectedRoot ?? '—'}</p>
            <p>quality seleccionada: {selectedQuality ?? '—'}</p>
            <p>chord final: {currentChordLabel}</p>
            <p>manos detectadas: {handsDetected}</p>
            <p className="mt-2 text-cyan-300">{debugMessage}</p>
          </div>
        </aside>
      </section>
    </main>
  )
}
