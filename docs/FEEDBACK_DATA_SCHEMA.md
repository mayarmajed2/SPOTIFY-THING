# Feedback & Interaction Data Schema (`docs/FEEDBACK_DATA_SCHEMA.md`)

---

## 1. Strict Separation of Feedback Categories

| Category | Source | Used for Offline Model Training? | Representation |
| :--- | :--- | :---: | :--- |
| **1. Directly Observed Listening Events** | Last.fm 1K scrobbles (`interactions_v2.parquet`) | **Yes** | `track_listened_event` (82,251), strictly causal `repeated_track_event` (286,938) |
| **2. Explicit User Feedback** | Live Resonance Studio UI (`POST /api/feedback`, SQLite `feedback_state.db`) | Online adaptation only (`LinUCB` + Fatigue Memory) | `like`, `dislike` |
| **3. Implicit Session Signals** | Live Resonance Studio UI playback simulator | Online adaptation only | `early_skip`, `partial_play`, `complete_play`, `repeat` |
| **4. Synthetic / Simulated Feedback** | Phase E stress benchmark (`validate_shuffle_and_feedback.py`) | **Never** (evaluation stress testing only) | Explicitly labeled `simulated_feedback` |

## 2. Canonical Offline Interaction Schema (`interactions_v2.parquet`)

| Column | Type | Provenance / Guarantee |
| :--- | :--- | :--- |
| `user_idx`, `user_id`, `country` | `int32`, `str`, `str` | Observed Last.fm 1K user attributes |
| `item_idx`, `spotify_track_id` | `int32`, `str` | Resolved catalog item ID |
| `event_timestamp` | `int64` | Original UTC Unix seconds |
| `session_id` | `int32` | Derived via configurable inactivity threshold (`1800s`) |
| `event_type` | `str` | Strictly `track_listened_event` or causal `repeated_track_event` (`cumcount() > 0`) |
| `event_source` | `str` | `"lastfm_1k_scrobble"` |
| `event_confidence` | `float32` | Entity-resolution confidence (`1.0` or `0.85`) |
| `is_observed` / `is_inferred` | `bool` / `bool` | `True` / `False` for all rows in `interactions_v2.parquet` |
| `match_tier` / `match_confidence` | `str` / `float64` | `exact_normalized` (`1.0`) or `packaging_stripped_high` (`0.85`) |

## 3. Deprecated / Removed Legacy Labels
- `played_ratio`, `early_skip`, `partial_play`, `complete_play`, and `favorite_listen` derived from `next_ts - ts` and full-timeline `ut_counts` have been **removed from `interactions_v2.parquet`** because Last.fm 1K does not observe playback duration, skips, or explicit Likes.
