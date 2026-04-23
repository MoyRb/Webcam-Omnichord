import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd


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


@dataclass
class CircleUI:
    center: Tuple[int, int]
    outer_radius: int
    inner_radius: int
    labels: List[str]
    color: Tuple[int, int, int]


class AudioEngine:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.block_size = 256
        self.max_voices = 24
        self.phase = np.zeros(self.max_voices, dtype=np.float64)
        self.chorus_phase = np.zeros(self.max_voices, dtype=np.float64)
        self.vibrato_phase = 0.0
        self.current_freqs = np.array([], dtype=np.float64)
        self.target_freqs = np.array([], dtype=np.float64)
        self.prev_freqs = np.array([], dtype=np.float64)
        self.crossfade_progress = 1.0
        self.current_amp = 0.0
        self.target_amp = 0.0
        self.muted = False
        self.lp_state = 0.0
        self.attack_time = 0.12
        self.release_time = 0.85
        self.note_smooth_time = 0.09
        self.crossfade_time = 0.22
        self.stream = sd.OutputStream(
            channels=1,
            callback=self._callback,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype="float32",
        )

    @staticmethod
    def midi_to_freq(midi_note: float) -> float:
        return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))

    def set_chord(self, freqs: Optional[List[float]]):
        if freqs is None or len(freqs) == 0:
            self.target_freqs = np.array([], dtype=np.float64)
            self.target_amp = 0.0
            return
        new_target = np.array(freqs, dtype=np.float64)
        # Transición de acordes: guardamos el acorde anterior para crossfade suave.
        if self.current_freqs.size > 0:
            self.prev_freqs = self.current_freqs.copy()
            self.crossfade_progress = 0.0
        self.target_freqs = new_target
        if self.current_freqs.size == 0:
            self.current_freqs = new_target.copy()
            self.prev_freqs = np.array([], dtype=np.float64)
            self.crossfade_progress = 1.0
        self.target_amp = 0.18

    def toggle_mute(self):
        self.muted = not self.muted

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def _callback(self, outdata, frames, _time_info, _status):
        # Ajuste de tamaño para evitar cortes cuando cambia la cantidad de notas.
        if self.current_freqs.size != self.target_freqs.size:
            if self.target_freqs.size == 0:
                self.current_freqs = np.array([], dtype=np.float64)
            elif self.current_freqs.size == 0:
                self.current_freqs = self.target_freqs.copy()
            else:
                m = min(self.current_freqs.size, self.target_freqs.size)
                new_freqs = self.target_freqs.copy()
                new_freqs[:m] = self.current_freqs[:m]
                self.current_freqs = new_freqs

        smoothing = 1.0 - math.exp(-frames / (self.note_smooth_time * self.sample_rate))
        if self.current_freqs.size > 0 and self.target_freqs.size > 0:
            self.current_freqs = self.current_freqs * (1 - smoothing) + self.target_freqs * smoothing
        amp_tc = self.attack_time if self.target_amp > self.current_amp else self.release_time
        amp_smoothing = 1.0 - math.exp(-frames / max(amp_tc * self.sample_rate, 1.0))
        self.current_amp = self.current_amp * (1 - amp_smoothing) + self.target_amp * amp_smoothing

        if self.muted:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return

        if self.current_freqs.size == 0 and self.current_amp < 1e-4:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return

        t = np.arange(frames, dtype=np.float64)
        signal_new = np.zeros(frames, dtype=np.float64)
        signal_old = np.zeros(frames, dtype=np.float64)
        vib_rate = 5.0
        vib_depth = 0.0022

        def render_bank(freq_bank: np.ndarray, start_idx: int) -> np.ndarray:
            acc = np.zeros(frames, dtype=np.float64)
            if freq_bank.size == 0:
                return acc
            for i, base_freq in enumerate(freq_bank):
                voice_idx = start_idx + i * 2
                if voice_idx + 1 >= self.max_voices:
                    break
                vib = np.sin(2 * math.pi * (self.vibrato_phase + t * vib_rate / self.sample_rate))
                base = base_freq * (1 + vib_depth * vib)
                # Sonido cálido: dos voces por nota con leve detune.
                detunes = (-0.0045, 0.0045)
                stereo_offsets = (0.13, 0.37)
                voice_sum = np.zeros(frames, dtype=np.float64)
                for local, det in enumerate(detunes):
                    f = base * (1.0 + det)
                    phase_inc = f / self.sample_rate
                    phases = (self.phase[voice_idx + local] + np.cumsum(phase_inc)) % 1.0
                    tri = 2 * np.abs(2 * phases - 1) - 1
                    saw = 2 * phases - 1
                    mix = 0.7 * tri + 0.3 * saw
                    # Chorus/ensemble suave por modulación de amplitud lenta.
                    self.chorus_phase[voice_idx + local] = (
                        self.chorus_phase[voice_idx + local] + frames * (0.24 + 0.07 * local) / self.sample_rate
                    ) % 1.0
                    chorus_lfo = np.sin(
                        2 * math.pi * (self.chorus_phase[voice_idx + local] + t / self.sample_rate * (0.24 + 0.07 * local))
                    )
                    ensemble_gain = 1.0 + 0.08 * chorus_lfo * math.cos(2 * math.pi * stereo_offsets[local])
                    voice_sum += mix * ensemble_gain
                    self.phase[voice_idx + local] = phases[-1]
                acc += 0.5 * voice_sum
            return acc / max(1, freq_bank.size)

        signal_new = render_bank(self.current_freqs, 0)
        signal_old = render_bank(self.prev_freqs, self.max_voices // 2)

        self.vibrato_phase = (self.vibrato_phase + frames * vib_rate / self.sample_rate) % 1.0

        if self.crossfade_progress < 1.0:
            step = frames / max(self.crossfade_time * self.sample_rate, 1.0)
            self.crossfade_progress = min(1.0, self.crossfade_progress + step)
        x = self.crossfade_progress
        # Crossfade con curva suave (equal power) para evitar saltos bruscos.
        gain_new = math.sin(0.5 * math.pi * x)
        gain_old = math.cos(0.5 * math.pi * x)
        signal = signal_new * gain_new + signal_old * gain_old
        if self.crossfade_progress >= 1.0:
            self.prev_freqs = np.array([], dtype=np.float64)

        # one-pole low pass filter
        alpha = 0.08
        filtered = np.zeros_like(signal)
        state = self.lp_state
        for n, s in enumerate(signal):
            state = state + alpha * (s - state)
            filtered[n] = state
        self.lp_state = state

        output = np.tanh(filtered * 1.45) * self.current_amp
        outdata[:, 0] = output.astype(np.float32)


def build_chord(root: str, quality: str, base_octave_midi: int = 48) -> Tuple[str, List[int], List[float]]:
    root_midi = base_octave_midi + NOTE_TO_SEMITONE[root]
    intervals = CHORD_INTERVALS[quality]
    midi_notes = [root_midi + i for i in intervals]
    freqs = [AudioEngine.midi_to_freq(m) for m in midi_notes]
    return f"{root}{quality}", midi_notes, freqs


def point_in_circle(point: Tuple[float, float], circle: CircleUI) -> bool:
    dx = point[0] - circle.center[0]
    dy = point[1] - circle.center[1]
    return dx * dx + dy * dy <= circle.outer_radius * circle.outer_radius


def ring_zone(point: Tuple[float, float], circle: CircleUI) -> str:
    """
    Lógica de anillo:
    - inner: zona neutra -> mantiene selección actual
    - active: anillo activo -> cambia por ángulo
    - outside: fuera del selector -> no hace nada
    """
    dx = point[0] - circle.center[0]
    dy = point[1] - circle.center[1]
    d2 = dx * dx + dy * dy
    if d2 <= circle.inner_radius * circle.inner_radius:
        return "inner"
    if d2 <= circle.outer_radius * circle.outer_radius:
        return "active"
    return "outside"


def segment_at(point: Tuple[float, float], circle: CircleUI) -> int:
    dx = point[0] - circle.center[0]
    dy = point[1] - circle.center[1]
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    section = 360.0 / len(circle.labels)
    return int(angle // section) % len(circle.labels)


def draw_radial_circle(frame, circle: CircleUI):
    overlay = frame.copy()
    num = len(circle.labels)
    section = 360.0 / num

    for i, label in enumerate(circle.labels):
        start = i * section
        end = (i + 1) * section
        shade = 0.65 + 0.25 * ((i % 2) == 0)
        color = tuple(int(c * shade) for c in circle.color)
        cv2.ellipse(
            overlay,
            circle.center,
            (circle.outer_radius, circle.outer_radius),
            0,
            start,
            end,
            color,
            -1,
            cv2.LINE_AA,
        )

        mid = math.radians((start + end) * 0.5)
        tx = int(circle.center[0] + math.cos(mid) * (circle.inner_radius + circle.outer_radius) * 0.5)
        ty = int(circle.center[1] + math.sin(mid) * (circle.inner_radius + circle.outer_radius) * 0.5)
        cv2.putText(
            overlay,
            label,
            (tx - 18, ty + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (250, 250, 250),
            2,
            cv2.LINE_AA,
        )

    # Convertimos a rueda tipo anillo: enmascaramos centro para mostrar dead zone.
    cv2.circle(overlay, circle.center, circle.inner_radius, (28, 28, 28), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)
    cv2.circle(frame, circle.center, circle.outer_radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, circle.center, circle.inner_radius, (255, 255, 255), 2, cv2.LINE_AA)

    for i in range(num):
        a = math.radians(i * section)
        x1 = int(circle.center[0] + math.cos(a) * circle.inner_radius)
        y1 = int(circle.center[1] + math.sin(a) * circle.inner_radius)
        x2 = int(circle.center[0] + math.cos(a) * circle.outer_radius)
        y2 = int(circle.center[1] + math.sin(a) * circle.outer_radius)
        cv2.line(frame, (x1, y1), (x2, y2), (230, 230, 230), 1, cv2.LINE_AA)


def highlight_selected_segment(frame, circle: CircleUI, selected_index: Optional[int]):
    if selected_index is None:
        return
    num = len(circle.labels)
    section = 360.0 / num
    start = selected_index * section
    end = (selected_index + 1) * section
    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        circle.center,
        (circle.outer_radius, circle.outer_radius),
        0,
        start,
        end,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(overlay, circle.center, circle.inner_radius, (0, 0, 0), -1, cv2.LINE_AA)
    frame[:] = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)


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
    chord_name = "-"
    chord_notes_text = "-"

    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            outer = min(h, w) // 5
            inner = int(outer * 0.42)
            left_circle = CircleUI(center=(w // 4, h // 2), outer_radius=outer, inner_radius=inner, labels=ROOTS, color=(230, 120, 100))
            right_circle = CircleUI(center=(3 * w // 4, h // 2), outer_radius=outer, inner_radius=inner, labels=QUALITIES, color=(90, 170, 240))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            fingertip_points: List[Tuple[float, float]] = []
            hands_count = 0

            if result.multi_hand_landmarks:
                hands_count = len(result.multi_hand_landmarks)
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    tip = hand_landmarks.landmark[8]
                    x = tip.x * w
                    y = tip.y * h

                    if hand_idx in smoothed_points:
                        px, py = smoothed_points[hand_idx]
                        sx = px * 0.7 + x * 0.3
                        sy = py * 0.7 + y * 0.3
                    else:
                        sx, sy = x, y
                    smoothed_points[hand_idx] = (sx, sy)
                    fingertip_points.append((sx, sy))

            left_idx = left_circle.labels.index(selected_root) if selected_root in left_circle.labels else None
            right_idx = right_circle.labels.index(selected_quality) if selected_quality in right_circle.labels else None
            draw_radial_circle(frame, left_circle)
            draw_radial_circle(frame, right_circle)
            highlight_selected_segment(frame, left_circle, left_idx)
            highlight_selected_segment(frame, right_circle, right_idx)

            for point in fingertip_points:
                # Selección por anillo activo: sólo cambia cuando está entre radio interno y externo.
                if ring_zone(point, left_circle) == "active":
                    idx = segment_at(point, left_circle)
                    selected_root = left_circle.labels[idx]

                if ring_zone(point, right_circle) == "active":
                    idx = segment_at(point, right_circle)
                    selected_quality = right_circle.labels[idx]

                cv2.circle(frame, (int(point[0]), int(point[1])), 14, (90, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)

            if selected_root and selected_quality:
                chord_name, midi_notes, freqs = build_chord(selected_root, selected_quality)
                note_names = [str(n) for n in midi_notes]
                chord_notes_text = ", ".join(note_names)
                engine.set_chord(freqs)
            else:
                engine.set_chord(None)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            info_lines = [
                f"root: {selected_root or '-'}",
                f"quality: {selected_quality or '-'}",
                f"chord: {chord_name}",
                f"notes (MIDI): {chord_notes_text}",
                f"fps: {fps:.1f}",
                f"hands: {hands_count}",
                f"audio: {'MUTED' if engine.muted else 'ON'} (M mute)",
                "Q para salir",
            ]
            y0 = 28
            for line in info_lines:
                cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(frame, line, (16, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 245, 250), 1, cv2.LINE_AA)
                y0 += 28

            cv2.imshow("Webcam Omnichord", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('m'), ord('M')):
                engine.toggle_mute()

    finally:
        engine.stop()
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
