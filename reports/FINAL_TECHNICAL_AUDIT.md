# Final Technical Audit

## Outcome

The repository is a compact, source-complete demo of a music recommendation and shuffle system. The architecture and benchmark report are internally connected, but the benchmark evidence is not yet independently reproducible in this checkout because raw data and model/processed artifacts are gitignored and absent.

## Highest-Risk Findings

1. The README needed to distinguish matched Last.fm events from retained interactions. A fresh build confirmed 2,067,551 matched events and 327,754 retained interactions; the README was corrected.
2. Benchmark claims are unverified against a fresh rerun. Raw datasets, processed Parquet files, model checkpoints, dataset hashes, and snapshot revisions are absent.
3. The data pipeline creates inferred weak labels and prior-derived synthetic audio for MPD tracks without exact Spotify audio. These values should not be described as measured listening or measured acoustics.
4. The serving engine previously could initialize with random/untrained model state when processed data existed but artifacts were missing. It now fails explicitly unless all required generated artifacts exist.
5. Feedback is process-memory-only and the feedback adaptation benchmark is simulated rather than evidence from live users.

## Applied Changes

- Added `reports/REPOSITORY_INVENTORY.md`.
- Added `reports/DATA_EVALUATION_AUDIT.md`.
- Added `reports/MODELS_SHUFFLE_FEEDBACK_AUDIT.md`.
- Added `reports/ARCHITECTURE_FRONTEND_BACKEND_AUDIT.md`.
- Corrected the README's matched-interaction claim.
- Added `torch` to `backend/pyproject.toml`.
- Restricted API CORS origins and added `ALLOWED_ORIGINS`.
- Added explicit serving-artifact validation.
- Corrected exact inner-product retrieval terminology.
- Corrected and tested the manual horizon override.
- Removed exact 40k/500-user assumptions from the API E2E test.

## Verification

- Backend engine/shuffle tests: **3 passed**.
- Next.js production build: **passed**.
- Python source compilation: not used as a separate gate; focused pytest and source execution exercised the relevant backend modules.
- Data construction completed with the default public-data settings. Full CPU-only training was stopped after about 70 minutes without complete NeuMF/Transformer checkpoints, so the benchmark suite was not rerun and no replacement methodology was used.
- Full API E2E was not run because processed data and model artifacts are absent and now correctly block startup.
- No restricted dataset or secret was downloaded or distributed.
- No commit, push, or pull request was created.

## Remaining Priority

Add a pinned dataset snapshot/hash, document exact training versions and seeds, add chronological-split leakage tests and multi-seed benchmark repetition, and only then present the current benchmark values as reproducible evidence rather than a single reported run.
