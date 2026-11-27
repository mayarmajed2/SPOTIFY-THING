# Changelog

All notable changes to **SPOTIFY-THING (Resonance)** are documented in this file.

## [2.0.0] — Branch `feature/chrome-companion-extension`

### Added
- **Manifest V3 Chrome Companion Extension (`extension/`):** Production-grade TypeScript + React 19 + Vite Chrome Extension featuring a dark music-intelligence popup & side-panel/expanded studio UI, event-driven MV3 background service worker (`service-worker.js`), strict CSP (`script-src 'self'; object-src 'self'`), minimal permissions (`storage`, `identity`, `sidePanel`), and zero content-script DOM scraping.
- **First-Time Onboarding UX (`extension/src/components/OnboardingFlow.tsx`):** 3-step onboarding explaining SPOTIFY-THING's companion role alongside Spotify playback, offering immediate Independent Mode or optional Spotify OAuth 2.0 PKCE, and calibrating initial genre & discovery intensity preferences.
- **Four Integrated Companion Pages (`extension/src/pages/`):**
  - `Discovery.tsx`: Familiarity vs. Discovery controller, neural explanations, 128D Similar Tracks & Similar Artists drawers, Open-in-Spotify deep links, and rich feedback (`Like`, `Dislike`, `More like this`, `Less like this`, `Request alternatives`).
  - `AdaptiveShuffle.tsx`: 5-axis preset configuration (`Artist Repetition`, `Genre Diversity`, `Musical Coherence`, `Discovery`, `Randomness`), Original vs. Generated sequence comparison, baseline diagnostics, artist/genre/discovery distributions, and URI/M3U/CSV + optional private Spotify playlist export.
  - `Insights.tsx`: Provenance-tagged analytics (`Observed`, `Inferred Preference`, `Model Prediction`) with temporary artist/genre fatigue triggers and zero fabricated listening-time history.
  - `Settings.tsx`: PKCE Client ID & Redirect URI helper, backend health checker, bandit memory reset, and full **Delete Stored Preferences & Application Data** control.
- **Official OAuth 2.0 PKCE & Capability-Aware `SpotifyProvider` (`extension/src/services/spotify-auth.ts`, `spotify-api.ts`):** RFC 7636 `S256` PKCE flow (no client secret), automatic token expiry/refresh, current 2026 `/v1/me/playlists` & `/v1/playlists/{id}/items` routes, owned/collaborative playlist filtering, graceful `401/403` Development Mode handling, and strict compliance with Spotify's non-ML-training policy.
- **FastAPI `/api/v1/*` Companion Router & Persistence (`backend/app/api/v1.py`, `backend/app/engine/extension_store.py`, `backend/app/schemas/v1_models.py`):** SHA-256-hashed session tokens, per-user isolated `feedback_slot_idx`, persistent user/shuffle preferences in WAL SQLite, sliding-window rate limiting, `chrome-extension://` CORS support, token-redacting error sanitization, and out-of-catalog shuffle metadata fallback.
- **Automated Test Suites & CI (`extension/src/test/extension.test.tsx`, `backend/tests/test_v1_companion_api.py`, `.github/workflows/ci.yml`):** 10 extension tests (Manifest V3, PKCE RFC 7636 vector, `SpotifyProvider` 403/filtering, service worker, full UI journey) + synthetic PyTorch `/api/v1/*` integration, isolation, persistence, and privacy deletion tests + CI `extension-build-and-tests` job.
- **Complete Companion Documentation & Config (`.env.example`, `docs/{EXTENSION_ARCHITECTURE,SPOTIFY_INTEGRATION,OAUTH_SETUP,API_INTEGRATION,PRIVACY,INSTALLATION}.md`, `PRODUCTION_PROGRESS.md`).**

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
