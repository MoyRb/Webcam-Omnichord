# Webcam Omnichord (Python Prototype)

Prototipo de instrumento musical controlado por webcam, inspirado en Omnichord/synth analógico suave.

## Requisitos

- Windows 10/11
- Python 3.10+
- Webcam funcional

## Instalación (PowerShell)

```powershell
cd .\python-prototype
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecutar

```powershell
cd .\python-prototype
.\.venv\Scripts\Activate.ps1
python .\main.py
```

## Controles

- **M**: mute / unmute
- **Q**: salir

## Uso rápido

1. Coloca la punta del índice sobre el círculo izquierdo para elegir raíz (C, D, E, F, G, A, B).
2. Coloca la punta del índice sobre el círculo derecho para elegir calidad (maj, min, 7, maj7, m7, sus4, dim).
3. El acorde se sostiene mientras exista selección de raíz + calidad.
4. Si cambias raíz o calidad, el audio hace transición suave.

## Notas

- La imagen se voltea horizontalmente (`cv2.flip(frame, 1)`) antes de correr MediaPipe para mantener coordenadas y visual en espejo.
- El tracking usa **landmark 8** (punta del índice) con suavizado:
  - `smoothed = previous * 0.7 + current * 0.3`
- Si detecta dos manos, evalúa ambas puntas del índice.
- El motor de audio usa `sounddevice.OutputStream` continuo con mezcla triangle+saw, vibrato lento, low-pass, delay corto y reverb ligera.
