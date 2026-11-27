#!/usr/bin/env bash
# Cross-platform (macOS / Linux) development launcher for Resonance Backend & Studio UI
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/backend/.venv/bin/python"
UVICORN_BIN="$ROOT_DIR/backend/.venv/bin/uvicorn"

if [[ ! -x "$UVICORN_BIN" ]]; then
  echo "Error: backend virtualenv not found at $UVICORN_BIN"
  echo "Create it first with: uv venv backend/.venv --python 3.12"
  exit 1
fi

export PYTHONPATH="$ROOT_DIR"

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo ""
    echo "Stopping FastAPI backend (PID $BACKEND_PID)..."
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting Resonance FastAPI Engine on http://127.0.0.1:8000 ..."
"$UVICORN_BIN" backend.app.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

echo "Starting Next.js Studio on http://localhost:3000 ..."
cd "$ROOT_DIR/frontend"
npm run dev
