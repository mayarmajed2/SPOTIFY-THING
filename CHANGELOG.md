# Changelog

All notable changes to **SPOTIFY-THING (Resonance)** are documented in this file.

## [1.1.0] — Branch `feat/repo-polish-and-shuffle-experiment`

### Added
- **GitHub Actions CI Workflow (`.github/workflows/ci.yml`):** Automated backend unit & persistence test execution (`pytest backend/tests -m "not requires_dataset" -v`) on Python 3.12 and strict frontend production build (`npm ci && npm run build`) on Node.js 22.
- **Code-Scoped MIT License (`LICENSE`):** Applies strictly to original source code and documentation in this repository, with an explicit scope notice excluding external third-party datasets and Spotify embeds/trademarks.
- **Cross-Platform Development Launcher (`start_dev.sh`):** POSIX bash startup script for macOS/Linux with `trap` cleanup mirroring `start_dev.ps1`.
- **Experimental Adaptive Shuffle Remaining-Quota Urgency (`backend/app/engine/adaptive_shuffle.py`):** Added opt-in `ShuffleConfig.quota_urgency_weight` (default `0.0`, preserving the original Adaptive Shuffle behavior verbatim) plus extended sequence diagnostics (`p25/median/max_artist_gap`, `window_genre_diversity`, `sequence_validity`).
- **Multi-Seed Shuffle Experiment Runner & Report (`backend/training/run_shuffle_experiment.py`, `docs/shuffle_experiment_report.md`, `data/reports/shuffle_experiment_report.json`):** Reproducible 5-seed × 200-playlist evaluation comparing 7 variants with stratified artist-heavy vs. balanced playlist analysis.
- **SQLite Feedback Persistence (`backend/app/engine/feedback_store.py`):** Thread-safe WAL-mode SQLite persistence for per-user LinUCB bandit matrices (`A_diag`, `b`, `delta_u`), liked/disliked/skipped sets, fatigue/boost maps, and feedback event logs across backend restarts.
- **Persistence & CORS Tests (`backend/tests/test_feedback_persistence.py`):** Verifies restart recovery, strict user isolation, concurrent write safety, and `RESONANCE_CORS_ORIGINS` parsing.
- **Modular Frontend Components:** Refactored the 1,299-line `frontend/src/app/page.tsx` into `DiscoveryTab.tsx`, `ShuffleArenaTab.tsx`, `BenchmarksTab.tsx`, and `PlayerDock.tsx` with zero UI or behavioral changes.

### Changed
- **CORS Hardening (`backend/app/main.py`):** Replaced wildcard `allow_origins=["*"]` with environment-configured `RESONANCE_CORS_ORIGINS` (defaulting safely to `http://localhost:3000,http://127.0.0.1:3000`) and restricted methods/headers (`GET, POST, OPTIONS`; `Content-Type, Authorization`).
- **Dataset-Dependent E2E Test Marker (`backend/tests/test_api_e2e.py`, `backend/pyproject.toml`):** Marked `test_full_api_lifecycle` with `@pytest.mark.requires_dataset` so CI explicitly deselects it without silently skipping failures.
- **Updated `.gitignore` & `README.md`:** Ignored runtime SQLite files (`*.db`, `*.db-wal`, `*.db-shm`) and documented CI, CORS, persistence, scaling path, license scope, and experiment findings.
