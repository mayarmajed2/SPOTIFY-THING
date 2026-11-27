# Installation & Local Development Guide

## Prerequisites

- **Python 3.12+** (with virtual environment at `backend/.venv`)
- **Node.js 22+** and `npm`
- **Google Chrome** (or Chromium-based browser supporting Manifest V3)
- Generated local catalog & model checkpoints in `data/processed/` and `data/artifacts/` (see `README.md` Quick Start if setting up on a fresh machine).

---

## 1. Start the FastAPI & PyTorch Companion Backend

From the repository root (`SPOTIFY-THING/`):

### Windows (PowerShell)
```powershell
$env:PYTHONPATH = "."
backend\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### macOS / Linux
```bash
PYTHONPATH=. backend/.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify the backend and neural models are online:
```bash
curl http://127.0.0.1:8000/api/v1/health
```

---

## 2. Build the Chrome Companion Extension

Open a second terminal:

```bash
cd extension
npm install
npm test
npm run build
```

This compiles the TypeScript + React 19 UI, bundles the Manifest V3 `service-worker.js`, and writes the complete Chrome-loadable package to `extension/dist/`.

---

## 3. Load Unpacked Extension in Chrome (Developer Mode)

1. Open Google Chrome and navigate to `chrome://extensions`.
2. Enable the **Developer mode** toggle in the top-right corner.
3. Click **Load unpacked**.
4. Select the `SPOTIFY-THING/extension/dist` directory.
5. Click the **Extensions** puzzle icon in the Chrome toolbar and pin **SPOTIFY-THING — AI Music Discovery & Adaptive Shuffle**.
6. Click the **SPOTIFY-THING** icon to launch the companion popup, complete the 3-step onboarding (Independent Mode works immediately with zero Spotify credentials), and explore **Discovery**, **Adaptive Shuffle Studio**, **Music Insights**, and **Settings**!

> **Optional Dev Preview Server:** While iterating on UI styles, you can also run `cd extension && npm run dev` to preview `popup.html?view=expanded` at `http://localhost:5173/popup.html?view=expanded` connected live to `:8000`.
