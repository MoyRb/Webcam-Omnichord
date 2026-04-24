import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd

MASTER_GAIN = 3.0


ROOTS = ["C", "D", "E", "F", "G", "A", "B"]
QUALITIES = ["Mayor", "Menor", "7", "Maj7", "m7", "sus4", "dim"]
CHORD_INTERVALS: Dict[str, List[int]] = {
    "Mayor": [0, 4, 7],
    "Menor": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "Maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
}
NOTE_TO_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}
SEMITONE_TO_NAME = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}


@dataclass
class RingUI:
    center: Tuple[int, int]
    outer_radius: int
    inner_radius: int
    labels: List[str]
    color: Tuple[int, int, int]
    title: str


class AudioEngine:
    """Motor de audio continuo con timbre cálido retro y transición suave entre acordes."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.block_size = 256

        # Estado de selección sonora.
        self.current_freqs = np.array([], dtype=np.float64)
        self.target_freqs = np.array([], dtype=np.float64)
        self.prev_freqs = np.array([], dtype=np.float64)
        self.crossfade_progress = 1.0
        self.crossfade_time = 0.35

        # Envolvente de amplitud global.
        self.current_amp = 0.0
        self.target_amp = 0.0
        self.attack_time = 0.10
        self.release_time = 0.85
        self.note_smooth_time = 0.11

        # Estados de osciladores / modulación.
        self.vibrato_phase = 0.0
        self.delay_buffer = np.zeros(int(self.sample_rate * 0.18), dtype=np.float64)
        self.delay_pos = 0
        self.reverb_buffer = np.zeros(int(self.sample_rate * 0.6), dtype=np.float64)
        self.reverb_pos = 0
        self.lp_state = 0.0

        self.max_voices = 64
        self.phase_main = np.zeros(self.max_voices, dtype=np.float64)
        self.phase_chorus = np.zeros(self.max_voices, dtype=np.float64)

        self.muted = False
        self.master_gain = MASTER_GAIN

        self.stream = sd.OutputStream(
            channels=2,
            callback=self._callback,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
        )

    @staticmethod
    def midi_to_freq(midi_note: float) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

    def set_chord(self, freqs: Optional[List[float]]) -> None:
        """Define el acorde objetivo. None => silencio con fade-out suave."""
        if not freqs:
            self.target_freqs = np.array([], dtype=np.float64)
            self.target_amp = 0.0
            return

        new_target = np.array(freqs, dtype=np.float64)
        if self.current_freqs.size > 0:
            self.prev_freqs = self.current_freqs.copy()
            self.crossfade_progress = 0.0

        self.target_freqs = new_target
        if self.current_freqs.size == 0:
            self.current_freqs = new_target.copy()
            self.prev_freqs = np.array([], dtype=np.float64)
            self.crossfade_progress = 1.0

        # Volumen moderado para evitar dureza y clipping.
        self.target_amp = 0.20

    def toggle_mute(self) -> None:
        self.muted = not self.muted

    def change_master_gain(self, delta: float) -> None:
        self.master_gain = float(np.clip(self.master_gain + delta, 0.5, 6.0))

    def status_text(self) -> str:
        if self.muted:
            return "MUTED"
        if self.target_amp <= 1e-5 and self.current_amp <= 1e-4:
            return "SILENT"
        return "PLAYING"

    def start(self) -> None:
        self.stream.start()

    def stop(self) -> None:
        self.stream.stop()
        self.stream.close()

    @staticmethod
    def _triangle(phase: np.ndarray) -> np.ndarray:
        return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0

    @staticmethod
    def _saw(phase: np.ndarray) -> np.ndarray:
        return 2.0 * phase - 1.0

    def _render_bank(self, freq_bank: np.ndarray, phase_offset: int, frames: int, t: np.ndarray) -> np.ndarray:
        """Renderiza una banca de voces con capas: cuerpo + color retro + pluck sutil."""
        if freq_bank.size == 0:
            return np.zeros(frames, dtype=np.float64)

        vib_rate = 4.7
        vib_depth = 0.0018
        vib = np.sin(2 * math.pi * (self.vibrato_phase + t * vib_rate / self.sample_rate))

        out = np.zeros(frames, dtype=np.float64)
        voice_counter = 0

        for note_idx, base_freq in enumerate(freq_bank):
            octave_support = [1.0, 0.5] if note_idx == 0 else [1.0]
            for octave_mul in octave_support:
                f0 = base_freq * octave_mul
                for detune in (-0.0032, 0.0030):
                    voice_idx = phase_offset + voice_counter
                    voice_counter += 1
                    if voice_idx >= self.max_voices:
                        break

                    freq = f0 * (1.0 + detune) * (1.0 + vib_depth * vib)
                    phase_inc = freq / self.sample_rate
                    phase = (self.phase_main[voice_idx] + np.cumsum(phase_inc)) % 1.0

                    tri = self._triangle(phase)
                    sine = np.sin(2 * math.pi * phase)
                    saw = self._saw(phase)

                    # Capa cálida (tri+sine), capa retro (saw leve), sustain órgano.
                    body = 0.68 * tri + 0.42 * sine
                    retro = 0.21 * saw
                    organ = 0.34 * np.sin(2 * math.pi * (phase * 0.5 + 0.15))

                    # Pluck/harp muy sutil por bloque (ataque rápido decreciente).
                    attack_env = np.exp(-t / (0.030 * self.sample_rate))
                    pluck = 0.16 * np.sin(2 * math.pi * phase * 2.0) * attack_env

                    # Chorus suave por modulación de fase muy lenta.
                    chorus_rate = 0.22 + (voice_idx % 3) * 0.06
                    chorus_lfo = np.sin(
                        2 * math.pi * (self.phase_chorus[voice_idx] + t * chorus_rate / self.sample_rate)
                    )
                    chorused = body + retro + organ + pluck
                    chorused *= 1.0 + 0.08 * chorus_lfo

                    out += chorused
                    self.phase_main[voice_idx] = phase[-1]
                    self.phase_chorus[voice_idx] = (
                        self.phase_chorus[voice_idx] + frames * chorus_rate / self.sample_rate
                    ) % 1.0

        return out / max(1, voice_counter)

    def _apply_fx(self, x: np.ndarray) -> np.ndarray:
        # Low-pass suave para quitar filo metálico.
        alpha = 0.07
        filtered = np.zeros_like(x)
        lp = self.lp_state
        for i, s in enumerate(x):
            lp = lp + alpha * (s - lp)
            filtered[i] = lp
        self.lp_state = lp

        # Delay corto y sutil.
        delayed = np.zeros_like(filtered)
        for i, s in enumerate(filtered):
            d = self.delay_buffer[self.delay_pos]
            delayed[i] = s + 0.12 * d
            self.delay_buffer[self.delay_pos] = s + 0.28 * d
            self.delay_pos = (self.delay_pos + 1) % self.delay_buffer.size

        # Reverb ligera (un solo feedback diffused).
        wet = np.zeros_like(delayed)
        for i, s in enumerate(delayed):
            r = self.reverb_buffer[self.reverb_pos]
            wet[i] = s + 0.10 * r
            self.reverb_buffer[self.reverb_pos] = s + 0.45 * r
            self.reverb_pos = (self.reverb_pos + 1) % self.reverb_buffer.size

        return wet

    def _callback(self, outdata, frames, _time_info, _status):
        if self.current_freqs.size != self.target_freqs.size:
            if self.target_freqs.size == 0:
                self.current_freqs = np.array([], dtype=np.float64)
            elif self.current_freqs.size == 0:
                self.current_freqs = self.target_freqs.copy()
            else:
                m = min(self.current_freqs.size, self.target_freqs.size)
                resized = self.target_freqs.copy()
                resized[:m] = self.current_freqs[:m]
                self.current_freqs = resized

        smoothing = 1.0 - math.exp(-frames / (self.note_smooth_time * self.sample_rate))
        if self.current_freqs.size > 0 and self.target_freqs.size > 0:
            self.current_freqs = self.current_freqs * (1.0 - smoothing) + self.target_freqs * smoothing

        amp_tc = self.attack_time if self.target_amp > self.current_amp else self.release_time
        amp_smoothing = 1.0 - math.exp(-frames / max(amp_tc * self.sample_rate, 1.0))
        self.current_amp = self.current_amp * (1.0 - amp_smoothing) + self.target_amp * amp_smoothing

        if self.muted:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return

        if self.current_freqs.size == 0 and self.current_amp < 1e-4:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return

        t = np.arange(frames, dtype=np.float64)
        new_sig = self._render_bank(self.current_freqs, 0, frames, t)
        old_sig = self._render_bank(self.prev_freqs, self.max_voices // 2, frames, t)

        self.vibrato_phase = (self.vibrato_phase + frames * 4.7 / self.sample_rate) % 1.0

        if self.crossfade_progress < 1.0:
            step = frames / max(self.crossfade_time * self.sample_rate, 1.0)
            self.crossfade_progress = min(1.0, self.crossfade_progress + step)

        xfade = self.crossfade_progress
        g_new = math.sin(0.5 * math.pi * xfade)
        g_old = math.cos(0.5 * math.pi * xfade)
        mixed = g_new * new_sig + g_old * old_sig
        if self.crossfade_progress >= 1.0:
            self.prev_freqs = np.array([], dtype=np.float64)

        effected = self._apply_fx(mixed)

        # Seguridad post-mezcla: normalización suave por bloque.
        peak = float(np.max(np.abs(effected))) if effected.size > 0 else 0.0
        if peak > 0.80:
            effected = effected * (0.80 / peak)

        # Ganancia maestra configurable al final de la cadena.
        out = effected * self.current_amp * self.master_gain

        # Limitador suave anti-clipping.
        out = np.tanh(out)
        out = np.clip(out, -0.95, 0.95)

        outdata[:, 0] = out.astype(np.float32)
        outdata[:, 1] = out.astype(np.float32)


def quality_suffix(quality: str) -> str:
    mapping = {
        "Mayor": "",
        "Menor": "m",
        "7": "7",
        "Maj7": "maj7",
        "m7": "m7",
        "sus4": "sus4",
        "dim": "dim",
    }
    return mapping[quality]


def format_chord_name(root: str, quality: str) -> str:
    return f"{root}{quality_suffix(quality)}"


def midi_to_note_name(midi_note: int) -> str:
    pitch = midi_note % 12
    octave = (midi_note // 12) - 1
    return f"{SEMITONE_TO_NAME[pitch]}{octave}"


def build_chord(root: str, quality: str, base_octave_midi: int = 48) -> Tuple[str, List[int], List[float]]:
    root_midi = base_octave_midi + NOTE_TO_SEMITONE[root]
    intervals = CHORD_INTERVALS[quality]
    midi_notes = [root_midi + interval for interval in intervals]
    freqs = [AudioEngine.midi_to_freq(m) for m in midi_notes]
    return format_chord_name(root, quality), midi_notes, freqs


def ring_zone(point: Tuple[float, float], ring: RingUI) -> str:
    dx = point[0] - ring.center[0]
    dy = point[1] - ring.center[1]
    d2 = dx * dx + dy * dy
    if d2 <= ring.inner_radius * ring.inner_radius:
        return "off"
    if d2 <= ring.outer_radius * ring.outer_radius:
        return "active"
    return "outside"


def segment_at(point: Tuple[float, float], ring: RingUI) -> int:
    dx = point[0] - ring.center[0]
    dy = point[1] - ring.center[1]
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    section = 360.0 / len(ring.labels)
    return int(angle // section) % len(ring.labels)


def draw_donut(frame: np.ndarray, ring: RingUI, selected_idx: Optional[int], is_off: bool) -> None:
    overlay = frame.copy()
    num = len(ring.labels)
    section = 360.0 / num

    for i, label in enumerate(ring.labels):
        start = i * section
        end = (i + 1) * section

        base_mult = 0.72 if i % 2 == 0 else 0.58
        seg_color = tuple(int(c * base_mult) for c in ring.color)
        if selected_idx == i and not is_off:
            seg_color = tuple(min(255, int(c * 1.35)) for c in ring.color)

        cv2.ellipse(
            overlay,
            ring.center,
            (ring.outer_radius, ring.outer_radius),
            0,
            start,
            end,
            seg_color,
            -1,
            cv2.LINE_AA,
        )

        mid = math.radians((start + end) * 0.5)
        tx = int(ring.center[0] + math.cos(mid) * (ring.inner_radius + ring.outer_radius) * 0.5)
        ty = int(ring.center[1] + math.sin(mid) * (ring.inner_radius + ring.outer_radius) * 0.5)
        cv2.putText(
            overlay,
            label,
            (tx - 28, ty + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 252),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(overlay, ring.center, ring.inner_radius, (0, 0, 0), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    # Bordes y divisiones.
    cv2.circle(frame, ring.center, ring.outer_radius, (240, 230, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, ring.center, ring.inner_radius, (240, 230, 255), 2, cv2.LINE_AA)
    for i in range(num):
        a = math.radians(i * section)
        x1 = int(ring.center[0] + math.cos(a) * ring.inner_radius)
        y1 = int(ring.center[1] + math.sin(a) * ring.inner_radius)
        x2 = int(ring.center[0] + math.cos(a) * ring.outer_radius)
        y2 = int(ring.center[1] + math.sin(a) * ring.outer_radius)
        cv2.line(frame, (x1, y1), (x2, y2), (220, 205, 255), 1, cv2.LINE_AA)

    # Centro OFF negro.
    off_color = (190, 190, 190) if is_off else (120, 120, 120)
    cv2.putText(
        frame,
        "OFF",
        (ring.center[0] - 24, ring.center[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        off_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        ring.title,
        (ring.center[0] - ring.outer_radius // 2, ring.center[1] - ring.outer_radius - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (230, 220, 255),
        2,
        cv2.LINE_AA,
    )


def evaluate_rings(
    fingertip_points: List[Tuple[float, float]],
    left_ring: RingUI,
    right_ring: RingUI,
    selected_root: Optional[str],
    selected_quality: Optional[str],
) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """Aplica la lógica de selección por posición visual (sin handedness)."""
    left_off = selected_root is None
    right_off = selected_quality is None

    for point in fingertip_points:
        left_zone = ring_zone(point, left_ring)
        right_zone = ring_zone(point, right_ring)

        if left_zone == "off":
            selected_root = None
            left_off = True
        elif left_zone == "active":
            selected_root = left_ring.labels[segment_at(point, left_ring)]
            left_off = False

        if right_zone == "off":
            selected_quality = None
            right_off = True
        elif right_zone == "active":
            selected_quality = right_ring.labels[segment_at(point, right_ring)]
            right_off = False

    return selected_root, selected_quality, left_off, right_off


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam")

    engine = AudioEngine()
    engine.start()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.6,
    )

    smoothed_points: Dict[int, Tuple[float, float]] = {}
    selected_root: Optional[str] = None
    selected_quality: Optional[str] = None
    left_off = True
    right_off = True

    chord_name = "-"
    chord_notes_text = "-"

    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mantener vista espejada como en la versión original.
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            outer = min(h, w) // 5
            inner = int(outer * 0.43)
            purple = (180, 90, 235)
            left_ring = RingUI(
                center=(w // 4, h // 2),
                outer_radius=outer,
                inner_radius=inner,
                labels=ROOTS,
                color=purple,
                title="ROOT",
            )
            right_ring = RingUI(
                center=(3 * w // 4, h // 2),
                outer_radius=outer,
                inner_radius=inner,
                labels=QUALITIES,
                color=purple,
                title="QUALITY",
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            fingertip_points: List[Tuple[float, float]] = []
            hands_count = 0

            if result.multi_hand_landmarks:
                hands_count = len(result.multi_hand_landmarks)
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    # Se mantiene landmark 8 (punta del índice).
                    tip = hand_landmarks.landmark[8]
                    x = tip.x * w
                    y = tip.y * h

                    # Suavizado para reducir jitter visual y cambios abruptos de segmento.
                    if hand_idx in smoothed_points:
                        px, py = smoothed_points[hand_idx]
                        sx = px * 0.72 + x * 0.28
                        sy = py * 0.72 + y * 0.28
                    else:
                        sx, sy = x, y
                    smoothed_points[hand_idx] = (sx, sy)
                    fingertip_points.append((sx, sy))

            selected_root, selected_quality, left_off, right_off = evaluate_rings(
                fingertip_points,
                left_ring,
                right_ring,
                selected_root,
                selected_quality,
            )

            left_idx = left_ring.labels.index(selected_root) if selected_root in left_ring.labels else None
            right_idx = right_ring.labels.index(selected_quality) if selected_quality in right_ring.labels else None

            draw_donut(frame, left_ring, left_idx, left_off)
            draw_donut(frame, right_ring, right_idx, right_off)

            for point in fingertip_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 14, (190, 110, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)

            # Solo hay acorde si root y quality están activos (no OFF).
            if selected_root is not None and selected_quality is not None:
                chord_name, midi_notes, freqs = build_chord(selected_root, selected_quality)
                note_names = [midi_to_note_name(n) for n in midi_notes]
                chord_notes_text = ", ".join(f"{n}({m})" for n, m in zip(note_names, midi_notes))
                engine.set_chord(freqs)
            else:
                chord_name = "-"
                chord_notes_text = "-"
                engine.set_chord(None)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            info_lines = [
                f"root: {selected_root or 'OFF'}",
                f"quality: {selected_quality or 'OFF'}",
                f"chord: {chord_name}",
                f"notes: {chord_notes_text}",
                f"hands detected: {hands_count}",
                f"left ring OFF: {left_off}",
                f"right ring OFF: {right_off}",
                f"audio: {engine.status_text()} (M mute)",
                f"master gain: {engine.master_gain:.2f} (+ / -)",
                f"fps: {fps:.1f}",
                "Q para salir",
            ]

            y0 = 28
            for line in info_lines:
                cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (245, 245, 255), 1, cv2.LINE_AA)
                y0 += 27

            cv2.imshow("Webcam Omnichord", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")):
                engine.toggle_mute()
            if key in (ord("+"), ord("=")):
                engine.change_master_gain(0.2)
            if key in (ord("-"), ord("_")):
                engine.change_master_gain(-0.2)

    finally:
        engine.stop()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
