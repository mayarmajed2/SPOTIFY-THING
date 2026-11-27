# Data Integrity & Source Pipeline Audit (`reports/DATA_AUDIT.md`)

**Audit Date:** 2026-11-27  
**Branch:** `research/data-integrity-and-ml-validation` (audited from `feat/repo-polish-and-shuffle-experiment` commit `9d78b58`)  
**Primary Audited File:** [backend/data_pipeline/download_and_build.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/data_pipeline/download_and_build.py)  
**Hardware Audited:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM, 3,907 MiB free), 15.4 GB RAM, 329 GB free disk.

---

## 1. Raw Source Dataset Inventory

| Source Dataset | Local Path | Raw Records | Unique Entities | Nulls / Anomalies | Time Span | Observed Signals |
| :--- | :--- | ---: | :--- | :--- | :--- | :--- |
| **Spotify Tracks Audio Features** (`maharshipandya/spotify-tracks-dataset`) | `data/raw/spotify_tracks/dataset.csv` (19.19 MB) | 114,000 | 89,741 unique `track_id`s, 114–125 genres | 1 null row (`artists`, `album_name`, `track_name`); ~24,259 multi-genre duplicate `track_id`s | Snapshot (undated) | `track_id`, metadata, `popularity`, 12 audio features, `track_genre` |
| **Spotify Million Playlist Dataset** (`jaxliu/Spotify_Million_Playlist_Dataset_Challenge`, 10 slices) | `data/raw/spotify_mpd/data/mpd.slice.*.json` (323.1 MB) | 10,000 playlists (9,808 with ≥8 tracks) | 169,433 unique `spotify:track:` URIs | No user IDs, no timestamps, no audio features, no skips/likes | Snapshot (`modified_at` Unix epoch) | Ordered playlist track lists (`pid`, `pos`, `track_uri`, `duration_ms`) |
| **Last.fm 1K** (`matthewfranglen/lastfm-1k`) | `data/raw/lastfm_1k/{train,valid,test}.gz.parquet` (447.5 MB) | 16,936,134 plays | ~992 users (`valid` split alone: 1,521,569 rows, 0 nulls in core fields) | Contains identical `0s` timestamp duplicates (772 in active slice); **no** play duration, **no** skip flag, **no** like/dislike flag | `2005-02-14` to `2009-06-18` (4.3 years) | `user_id`, `timestamp`, `artist_name`, `track_name`, `country` |

---

## 2. Confirmed Data Integrity Problems

### Problem D1 (CRITICAL) — Invalid Implicit Feedback & Manufactured Ground-Truth Labels
- **Location:** `backend/data_pipeline/download_and_build.py:392–425`
- **Problematic Code:**
  ```python
  listen_sec = (matched_events["next_ts"] - matched_events["ts"]).dt.total_seconds().fillna(240.0)
  played_ratio = np.where(listen_sec > 1800.0, 1.0, np.clip(listen_sec / dur_sec, 0.0, 1.5))
  ut_counts = matched_events.groupby(["user_idx", "item_idx"])["ts"].transform("count")
  action_id[(played_ratio < 0.30) & (listen_sec < 45.0)] = 0   # labeled early_skip
  action_id[(ut_counts.to_numpy() >= 5) & (played_ratio >= 0.70)] = 4  # labeled favorite_listen
  ```
- **Empirical Reproduction on `data/processed/interactions.parquet` (`327,754` rows):**
  - **`135,788` events (`41.4%`)** had `dt > 1800s` (session endings or pauses) and were artificially stamped `played_ratio = 1.0` (`complete_play`).
  - **`772` events** had `dt == 0s` (batch scrobbler timestamp collisions) and were misclassified as `early_skip` (`action_id = 0`, out of `1,148` total `early_skip` labels — meaning **67.2% of all "early skips" were actually 0-second timestamp collisions!**).
  - **`193,084` events (`58.9%`)** were labeled `favorite_listen` purely from `ut_counts >= 5`, manufacturing an explicit-sounding preference label from repeat count — and doing so using **full-timeline future counts**!
- **Scientific Verdict:** Last.fm 1K only observes `track_listened_event` timestamps. Timestamp differences cannot prove playback duration, completion, or skips. All manufactured `played_ratio`, `early_skip`, `complete_play`, and `favorite_listen` labels must be removed from ground-truth training data and replaced with an honest provenance-aware schema (`track_listened_event`, `repeated_track_event` strictly causal).

### Problem D2 (HIGH) — Unseparated Synthetic Audio Features (34.72% of Active Catalog)
- **Location:** `backend/data_pipeline/download_and_build.py:237–266`
- **Problematic Code:** Imputes missing MPD track audio features using `artist_priors` plus Gaussian noise (`rng.normal(0, 6.0)` for tempo, `rng.normal(0, 0.04)` for continuous features).
- **Empirical Reproduction:** **`13,887` of `40,000` active tracks (`34.72%`)** have synthetic artist-mean + noise audio values (`has_exact_audio = False`), while `26,113` (`65.28%`) come from `maharshipandya/spotify-tracks-dataset`.
- **Impact:** Acoustic Smoothness and Mean BPM Jump metrics in Adaptive Shuffle were evaluating **Gaussian-perturbed synthetic estimates** alongside measured audio features without distinguishing provenance (`audio_feature_source`, `audio_feature_confidence`, `is_synthetic`).

### Problem D3 (HIGH) — Aggressive Entity Resolution Collisions & Missing Confidence Tiers
- **Location:** `backend/data_pipeline/download_and_build.py:59–67, 286–326`
- **Problematic Code:** Regex strips `live`, `remaster`, `radio edit`, `version`, `deluxe`, `bonus` unconditionally before joining Last.fm `(artist_norm, track_norm)` onto Spotify tracks.
- **Empirical Reproduction:**
  - `1,720` active tracks contain version qualifiers (`live`, `remix`, `acoustic`, `remaster`, etc.).
  - `3,261` `(artist_norm, track_norm)` collisions exist inside the 40,000 active tracks.
  - `lookup_df.drop_duplicates(subset=["artist_norm", "track_norm"], keep="first")` silently collapses live/remix/acoustic variants to a single track with no match confidence tier (`exact_normalized`, `version_stripped_high`, `ambiguous`, `unmatched`).

### Problem D4 (MEDIUM) — Truncation & Sessionization Sensitivity
- **Location:** `backend/data_pipeline/download_and_build.py:394–398, 429–431`
- **Observation:** `tail(400)` truncates long-term user histories before temporal splitting without recording how many historical plays were dropped; `0s` duplicate timestamps are not deduplicated; the 30-minute (`1800s`) session gap threshold was hardcoded without sensitivity analysis.

### Problem D5 (ENVIRONMENT / PORTABILITY) — DuckDB Binary Blocked by Windows Smart App Control
- **Observation:** `_duckdb.pyd` is blocked on Windows 11 under Application Control (`ImportError: DLL load failed while importing _duckdb`). Replacing the join with vectorized **PyArrow + Pandas** makes the ETL pipeline 100% portable without sacrificing speed.

---

## 3. Correctly Implemented Components (To Preserve)
- Deterministic `spotify_track_id` deduplication preserving multi-genre lists (`all_genres`).
- Zero-auth reproducible Hugging Face Hub downloads (no credential leaks, no restricted data committed to Git).
- Clean Parquet columnar output structure (`tracks.parquet`, `interactions.parquet`, `playlists.parquet`).
- Strict separation of raw (`data/raw/`) vs. processed (`data/processed/`) directories (we will write redesigned outputs to `data/processed_v2/` so existing `data/processed/` is never overwritten!).
