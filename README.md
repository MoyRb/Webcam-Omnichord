# webcam-omnichord

App web experimental construida con **Vite + React + TypeScript**.
Usa webcam para controlar un instrumento estilo theremin/omnichord:

- Mano izquierda → selección de acorde (C, Dm, Em, F, G, Am, Bdim).
- Mano derecha → ejecución de notas cuantizadas al acorde.
- Altura (Y) → nota del acorde.
- Posición horizontal (X) → cutoff de filtro low-pass.

## Stack

- React + Vite + TypeScript
- `@mediapipe/tasks-vision` para hand tracking
- Tone.js para síntesis
- Tailwind CSS para UI

## Instalación

```bash
npm install
npm run dev
```

Luego abre el URL local que imprime Vite (normalmente `http://localhost:5173`).

## Uso

1. Click en **Start Audio**.
2. Click en **Enable Camera** y acepta permisos de webcam.
3. Mueve mano izquierda dentro del círculo izquierdo para cambiar acorde.
4. Mueve mano derecha dentro del círculo derecho para tocar notas del acorde.
5. Ajusta escala, waveform, reverb y delay desde el panel.

## Notas técnicas

- La app cuantiza las notas al acorde activo para evitar desafinación.
- Se evita retrigger infinito de audio con:
  - detección de cambio de nota, y
  - throttling temporal (~180ms).
- El tracking usa el landmark de muñeca (`hand[0]`) como punto principal de cada mano.
