# Launch both FastAPI Backend (port 8000) and Next.js Studio (port 3000)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "Starting Resonance FastAPI Engine on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process -FilePath "$root\backend\.venv\Scripts\uvicorn.exe" -ArgumentList "backend.app.main:app --host 127.0.0.1 --port 8000" -WorkingDirectory $root

Write-Host "Starting Next.js Studio on http://localhost:3000 ..." -ForegroundColor Cyan
Set-Location "$root\frontend"
& npm.cmd run dev
