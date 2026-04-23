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
const REPLAY_COOLDOWN_MS = 300
const BASE_OCTAVE = 4
const INDEX_TIP_LANDMARK = 8

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
    oscillator: { type: 'triangle' },
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

function getIndexTipPoint(handLandmarks: NormalizedLandmark[], width: number, height: number): HandPoint | null {
  const landmark = handLandmarks[INDEX_TIP_LANDMARK]
  if (!landmark) return null

  return {
    x: (1 - landmark.x) * width,
    y: landmark.y * height,
  }
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

  const [audioStarted, setAudioStarted] = useState(false)
  const [toneContextState, setToneContextState] = useState<string>(Tone.getContext().state)
  const [outputVolume, setOutputVolume] = useState<number>(Tone.Destination.volume.value)
  const [audioError, setAudioError] = useState<string | null>(null)
  const [cameraReady, setCameraReady] = useState(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [reverbWet, setReverbWet] = useState(0.35)
  const [delayWet, setDelayWet] = useState(0.25)
  const [selectedRoot, setSelectedRoot] = useState<Root | null>(null)
  const [selectedQuality, setSelectedQuality] = useState<Quality | null>(null)
  const [leftHand, setLeftHand] = useState<HandPoint | null>(null)
  const [rightHand, setRightHand] = useState<HandPoint | null>(null)
  const [handsDetected, setHandsDetected] = useState(0)
  const [indexTipX, setIndexTipX] = useState<number | null>(null)
  const [indexTipY, setIndexTipY] = useState<number | null>(null)
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
    try {
      setAudioError(null)
      await Tone.start()
      await Tone.getContext().resume()

      const contextState = Tone.getContext().state
      setToneContextState(contextState)

      if (!synthRef.current) {
        const filter = new Tone.Filter(1200, 'lowpass')
        const delay = new Tone.FeedbackDelay('8n', 0.25)
        const reverb = new Tone.Reverb({ decay: 3, wet: reverbWet })
        const synth = new Tone.PolySynth(Tone.Synth, OMNICHORD_POLY_OPTIONS)

        delay.wet.value = delayWet
        Tone.Destination.volume.value = -8

        synth.chain(filter, delay, reverb, Tone.Destination)

        synthRef.current = synth
        filterRef.current = filter
        reverbRef.current = reverb
        delayRef.current = delay
      }

      setOutputVolume(Tone.Destination.volume.value)
      const isRunning = Tone.getContext().state === 'running'
      setAudioStarted(isRunning)

      if (!isRunning || !synthRef.current) {
        setAudioError('AudioContext no está corriendo. Revisa permisos/audio del navegador')
        setDebugMessage('No se pudo iniciar el contexto de audio.')
        return
      }

      const testNotes = ['C4', 'E4', 'G4']
      synthRef.current.triggerAttackRelease(testNotes, 0.5)
      setDebugMessage(`Test audio OK: ${testNotes.join(', ')}`)
    } catch (error) {
      setAudioStarted(false)
      setToneContextState(Tone.getContext().state)
      setAudioError('AudioContext no está corriendo. Revisa permisos/audio del navegador')
      const message = error instanceof Error ? error.message : 'Error al inicializar audio.'
      setDebugMessage(message)
    }
  }, [delayWet, reverbWet])

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
        const tipPoint = getIndexTipPoint(hand, canvas.width, canvas.height)
        if (!tipPoint) return

        ctx.fillStyle = 'rgba(250, 204, 21, 0.95)'
        ctx.beginPath()
        ctx.arc(tipPoint.x, tipPoint.y, 10, 0, Math.PI * 2)
        ctx.fill()

        ctx.strokeStyle = 'rgba(2, 6, 23, 0.95)'
        ctx.lineWidth = 2
        ctx.stroke()
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

    const indexTips = landmarks
      .map((hand) => getIndexTipPoint(hand, VIDEO_WIDTH, VIDEO_HEIGHT))
      .filter((point): point is HandPoint => point !== null)

    const sortedTips = [...indexTips].sort((a, b) => a.x - b.x)
    setLeftHand(sortedTips[0] ?? null)
    setRightHand(sortedTips[1] ?? null)

    if (indexTips[0]) {
      setIndexTipX(indexTips[0].x)
      setIndexTipY(indexTips[0].y)
    } else {
      setIndexTipX(null)
      setIndexTipY(null)
    }

    let nextRoot: Root | null = null
    let nextQuality: Quality | null = null

    indexTips.forEach((point) => {
      if (pointInZone(point, leftZone)) {
        const rootIndex = pointToSegmentIndex(point, leftZone, ROOTS.length)
        nextRoot = ROOTS[rootIndex]
      }

      if (pointInZone(point, rightZone)) {
        const qualityIndex = pointToSegmentIndex(point, rightZone, QUALITIES.length)
        nextQuality = QUALITIES[qualityIndex]

        if (filterRef.current) {
          const rightXNorm = clamp01((point.x - (rightZone.x - rightZone.r)) / (rightZone.r * 2))
          filterRef.current.frequency.value = 250 + rightXNorm * 4200
        }
      }
    })

    if (nextRoot) setSelectedRoot(nextRoot)
    if (nextQuality) setSelectedQuality(nextQuality)

    animationRef.current = requestAnimationFrame(runDetectionFrame)
  }, [drawOverlay, leftZone, pointInZone, rightZone])

  useEffect(() => {
    const chordId = selectedRoot && selectedQuality ? `${selectedRoot}-${selectedQuality}` : ''
    if (!chordId || !synthRef.current) return
    if (!audioStarted || Tone.getContext().state !== 'running') return

    const now = performance.now()
    if (chordId === lastTriggeredChord.current && now - lastPlayTime.current < REPLAY_COOLDOWN_MS) {
      return
    }

    const notes = buildChord(selectedRoot, selectedQuality)
    synthRef.current.triggerAttackRelease(notes, '8n')
    lastTriggeredChord.current = chordId
    lastPlayTime.current = now
    setDebugMessage(`Trigger: ${selectedRoot}${selectedQuality} -> ${notes.join(', ')}`)
  }, [audioStarted, selectedQuality, selectedRoot])

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

          {audioError ? (
            <div className="rounded-xl border border-rose-400/50 bg-rose-500/10 p-3 text-sm text-rose-200">{audioError}</div>
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
            <p>toneContextState: {toneContextState}</p>
            <p>outputVolume: {outputVolume.toFixed(1)} dB</p>
            <p>root seleccionada: {selectedRoot ?? '—'}</p>
            <p>quality seleccionada: {selectedQuality ?? '—'}</p>
            <p>chord final: {currentChordLabel}</p>
            <p>manos detectadas: {handsDetected}</p>
            <p>indexTipX: {indexTipX?.toFixed(1) ?? '—'}</p>
            <p>indexTipY: {indexTipY?.toFixed(1) ?? '—'}</p>
            <p>landmarkUsed: {INDEX_TIP_LANDMARK}</p>
            <p className="mt-2 text-cyan-300">{debugMessage}</p>
          </div>
        </aside>
      </section>
    </main>
  )
}
