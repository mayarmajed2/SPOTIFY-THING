# Repository Inventory

## Audit Scope

Repository: `mayarmajed2/SPOTIFY-THING`
Branch: `mayarmajed2-spotify-technical-audit`
Audit date: 2026-11-26

This inventory maps the checked-in repository to source evidence. Raw datasets, processed Parquet artifacts, model checkpoints, and Python virtual environments are not present in the checkout. The checked-in evidence consists of source code, tests, reports, and a README.

## Repository Layout

| Path | Role | Evidence |
|---|---|---|
| `README.md` | Product overview, setup, and claimed benchmark results | README claims are not all corroborated by checked-in data |
| `backend/data_pipeline/download_and_build.py` | Downloads public datasets and builds tracks/interactions/playlists Parquet tables | Primary data construction path |
| `backend/training/train_and_evaluate.py` | Trains CF/BPR/NeuMF/Transformer models and produces benchmark reports | Full offline evaluation pipeline |
| `backend/app/main.py` | FastAPI routes and application lifespan | Serving and API integration |
| `backend/app/engine/recommender.py` | Hybrid retrieval, MMR, feedback, and shuffle orchestration | Runtime recommendation engine |
| `backend/app/engine/adaptive_shuffle.py` | Adaptive shuffle and baseline shuffle implementations | Runtime and benchmark sequencing |
| `backend/app/engine/online_feedback.py` | LinUCB-style feedback and fatigue/boost state | Live feedback implementation |
| `backend/app/models/embeddings.py` | 128D shared multimodal track encoder | NeuMF and transformer track representation |
| `backend/app/models/collaborative.py` | Item-KNN, BPR-MF, and NeuMF | Model implementations |
| `backend/app/models/sequential_transformer.py` | Causal dual-horizon transformer and horizon fusion | Sequence model implementation |
| `backend/app/schemas/api_models.py` | Pydantic API input contracts | Request validation |
| `backend/tests/test_engine_and_shuffle.py` | Unit/property tests for shuffle, feedback, and model shapes | Two unit-style tests and one model test |
| `backend/tests/test_api_e2e.py` | API lifecycle test against trained artifacts | Requires generated processed data/artifacts |
| `frontend/src/app/page.tsx` | Next.js Resonance Studio UI | Discovery, shuffle, feedback, and benchmarks |
| `frontend/src/lib/api.ts` | Typed browser-to-FastAPI client | API contract and fetch wrappers |
| `frontend/src/components/VisualCharts.tsx` | UI chart components | Radar and flow visualizations |
| `frontend/package.json` | Next.js/React frontend dependencies | Next 15, React 19 |
| `backend/pyproject.toml` | Python dependencies and version constraint | Python >=3.12,<3.13 |
| `data/reports/evaluation_results.json` | Checked-in output report | 40k tracks, 327,754 interactions, 943 users |
| `data/reports/*.png` | Checked-in benchmark plots | Visual outputs only; source run is not checked in |
| `docs/studio_preview.png` | README UI preview | Static screenshot |
| `start_dev.ps1` | Starts backend and frontend | Local Windows launcher |

## Data Pipeline

- Downloads a Spotify audio catalog, configurable MPD slices, and Last.fm 1K train/valid/test Parquet files.
- Resolves entities by normalized artist and track names.
- Adds frequent MPD tracks that match known Spotify artists, using artist-prior genre/audio values plus seeded synthetic perturbations.
- Selects an active catalog using Last.fm, MPD, popularity, exact-audio presence, and genre coverage.
- Filters each user's recent 400 interactions after requiring at least 25 matched listens.
- Infers sessions by a 30-minute gap and derives skip/partial/complete/repeat/favorite action labels from next-event timing and replay counts.
- Writes `tracks.parquet`, `interactions.parquet`, `playlists.parquet`, and `dataset_stats.json`.

## Evaluation Pipeline

- Chronological leave-last-two-positive-items split.
- Item-KNN, BPR-MF, NeuMF, and dual-horizon transformer.
- Recommendation ranking uses 1 target plus 499 sampled unseen negatives.
- Shuffle benchmark uses the first 200 curated playlists with at least 20 tracks.
- Feedback benchmark simulates 100 users, skipping three dominant-genre tracks and liking/repeating target-genre tracks.
- The training script sets NumPy and torch seeds to 42 and saves the resulting JSON and charts.

## Existing Tests

- `test_adaptive_shuffle_reduces_artist_clumping_and_smooths_flow`
- `test_online_feedback_bandit_shifts_affinity_directionally`
- `test_multimodal_neumf_and_transformer_shapes_and_cold_start`
- `test_full_api_lifecycle`

The unit test suite does not test dataset construction, split leakage, replay negative sampling, report reproducibility, or frontend behavior.

## Checked-In Reproducibility Evidence

- Report: `data/reports/evaluation_results.json`
- Reported device: NVIDIA GeForce RTX 3050 Laptop GPU
- Reported runtime: 1023.54 seconds
- Reported dataset: 40,000 tracks, 26,113 exact-audio tracks, 8,146 artists, 114 genres, 943 users, 327,754 interactions, 6,932 playlists
- Data/model output directories are ignored, so source data and trained artifacts cannot be reconstructed or inspected in this checkout.

## Verified Claim Discrepancies

1. The README should distinguish 2,067,551 raw matched Last.fm events from 327,754 retained timestamp-derived interactions.
2. The frontend loading text refers to a FAISS index, but runtime code implements an exact NumPy/BLAS inner-product search rather than loading or using FAISS.
