# Session Construction & Threshold Sensitivity (`docs/SESSION_CONSTRUCTION.md`)

**Implementation:** `sessionize_timestamps` & `run_session_sensitivity` in [backend/data_pipeline/build_verified_dataset.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/data_pipeline/build_verified_dataset.py)

---

## 1. Sessionization Rules (v2)
- **Observed Timestamps Only:** Preserves original Last.fm UTC `event_timestamp` (`unix_ts`). Never manufactures `listen_sec` or `played_ratio`.
- **Duplicate Timestamp Handling:** Exact duplicates `(user_id, item_idx, event_timestamp)` are removed prior to sessionization.
- **Configurable Boundary:** A new `session_id` begins per user whenever `event_timestamp[t] - event_timestamp[t-1] > session_gap_seconds` (default `1800s` = 30 minutes).
- **Repeated Tracks:** Preserved in chronological order and tagged causally (`repeated_track_event` iff `cumcount() > 0` prior to timestamp `t`).

## 2. Empirical Sensitivity Analysis (`369,189` Interactions, `929` Users)

| Inactivity Threshold | Total Sessions | Mean Events / Session | Median Events / Session | Sessions with ≥ 5 Events |
| :--- | ---: | ---: | ---: | ---: |
| `900s` (15 min) | 230,156 | 1.60 | 1.0 | 3.51% |
| **`1800s` (30 min — default)** | **186,320** | **1.98** | **1.0** | **7.02%** |
| `3600s` (60 min) | 149,070 | 2.48 | 1.0 | 11.88% |
| `7200s` (120 min) | 123,029 | 3.00 | 2.0 | 16.93% |

## 3. Interpretation
Because only `8.39%` of raw Last.fm 1K scrobbles match the active 40k Spotify catalog, inter-event gaps in the matched subset are sparser than in raw unfiltered scrobbles (median 1–2 matched events per 30-min window). Consequently, our Causal Session Transformer uses a rolling window of up to `L = 50` consecutive matched events accompanied by discrete `delta_t` time-gap buckets (`<1m`, `1–5m`, `5–30m`, `30m–6h`, `6–48h`, `>48h`) rather than truncating context at hard 30-minute session boundaries.
