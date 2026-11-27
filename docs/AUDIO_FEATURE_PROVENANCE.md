# Audio Feature Provenance (`docs/AUDIO_FEATURE_PROVENANCE.md`)

**Implementation:** [backend/data_pipeline/build_verified_dataset.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/data_pipeline/build_verified_dataset.py) & `fit_transform_audio_features` in [backend/training/splits_and_sampling.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/splits_and_sampling.py)

---

## 1. Provenance Columns in `tracks_v2.parquet`

Every track in `tracks_v2.parquet` carries four explicit provenance columns:

| Column | Possible Values | Meaning |
| :--- | :--- | :--- |
| `audio_feature_source` | `maharshipandya_spotify_csv` \| `artist_mean_prior_from_spotify_csv` | Origin of the 12 acoustic values |
| `audio_feature_type` | `third_party_measured_spotify_audio` \| `artist_level_estimate` | Direct track-level measurement vs. artist-mean estimate |
| `audio_feature_confidence` | `1.0` (measured) \| `0.25–0.50` (artist estimate) | Scales monotonically with artist's measured track count, capped at `0.50` |
| `is_synthetic` | `False` \| `True` | Explicit boolean separating measured audio from estimated priors |

## 2. Empirical Distribution in `tracks_v2.parquet` (40,000 Tracks)
- **Measured Track-Level Audio (`is_synthetic = False`):** **30,450 tracks (76.13%)** — directly sourced from `maharshipandya/spotify-tracks-dataset`.
- **Artist-Level Estimates (`is_synthetic = True`):** **9,550 tracks (23.88%)** — deterministic artist mean from `spotify-tracks-dataset` for frequent train-MPD tracks missing direct audio rows. **Random Gaussian noise injection from the legacy pipeline has been completely removed.**

## 3. Leakage Prevention & Evaluation Safeguards
1. **No Full-Catalog Scaling in ETL:** `tracks_v2.parquet` stores raw unscaled audio values only.
2. **Training-Only Measured Scaler:** `fit_transform_audio_features` fits z-score `(mean, std)` strictly on tracks satisfying `(item_idx in train_items) AND (is_synthetic == False)` (verified by `test_audio_scaler_ignores_test_only_and_synthetic_tracks`).
3. **Measured-Only Shuffle Acoustic Evaluation:** Protocol 1 in `validate_shuffle_and_feedback.py` reports `measured_audio_smoothness` strictly on consecutive track pairs where **both** tracks have `is_synthetic == False`.
