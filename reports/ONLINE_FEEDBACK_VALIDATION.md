# Online Feedback Validation Report (`reports/ONLINE_FEEDBACK_VALIDATION.md`)

**Script:** [backend/training/validate_shuffle_and_feedback.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/validate_shuffle_and_feedback.py)  
**Raw Artifact:** [data/reports/shuffle_and_feedback_validation_v2.json](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/data/reports/shuffle_and_feedback_validation_v2.json)  
**Scope:** Controlled **simulated feedback** stress testing across 60 real user profiles (explicitly synthetic; does not prove real-user online A/B performance).

---

## 1. Ten Behavioral Stress Scenarios

| Scenario | Dominant Genre Share@20 | Target Genre Share@20 | Mean Bandit Drift `\|\|Δu\|\|` | Behavioral Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **1. No feedback (baseline)** | 40.33% | 0.67% | 0.0000 | Stable baseline |
| **2. One early skip** | **2.17%** | 0.75% | 0.2100 | **OVERREACTION DETECTED** (single skip collapses dominant genre from 40.3% → 2.2%) |
| **3. Three consecutive skips** | 0.00% | 0.92% | 0.3879 | Full suppression achieved |
| **4. Six consecutive skips** | 0.00% | 1.00% | 0.5371 | Bounded by `0.65` residual norm cap |
| **5. Three repeated likes (target)** | 9.58% | 67.83% | 0.5783 | Strong positive steering |
| **6. Three repeated listens (target)** | 14.00% | 45.75% | 0.5076 | Moderate positive steering |
| **7. 2 skips → 6 neutral plays (decay)** | 4.83% | 0.50% | 0.2128 | **Slow recovery** (genre fatigue decays too slowly over 6 neutral plays) |
| **8. Contradictory (like → dislike same track)** | 39.58% | 0.00% | 0.0504 | Cleanly cancels drift (`0.05`) and suppresses disliked item |
| **9. Preference reversal (3 skips + 3 likes)** | 0.17% | 46.25% | 0.4991 | Fast pivot to new mood |
| **10. Return to previously preferred genre** | 33.92% | 23.25% | 0.5706 | Recovers prior taste when user explicitly re-engages |

---

## 2. Component Ablation (Scenario 9: Preference Reversal)

| Ablation Variant | Fatigued Dominant Genre Share@20 ↓ | Newly Liked Target Genre Share@20 ↑ |
| :--- | :---: | :---: |
| **Static Profile** | 40.00% | 0.67% |
| **LinUCB Only** | 11.42% | 8.08% |
| **Fatigue Memory Only** | 0.00% | 7.50% |
| **Combined System (LinUCB + Fatigue)** | **0.17%** | **46.25%** |

---

## 3. Critical Finding & Calibration Recommendation
- **Single-Skip Overreaction:** In `UserFeedbackState.record_feedback`, a single `early_skip` currently adds `+0.35` to `genre_fatigue` (`-0.105` flat score penalty across every track of that genre) plus `-0.21` EMA embedding repulsion. As Scenario 2 proves, one accidental skip wipes out `94.6%` of the user's primary genre in Top-20.
- **Recommended Calibration (Next Iteration):** Require `>= 2` skips within a rolling window before activating genre-wide fatigue (keeping a single skip strictly track- and artist-local), and increase neutral-play fatigue decay from `0.90` to `0.78` so temporary fatigue recovers smoothly.
- **Persistence Verified:** `SQLiteFeedbackStore` (`WAL` mode + `BEGIN IMMEDIATE`) passes all restart-recovery, user-isolation, and multi-threaded concurrency tests in `test_feedback_persistence.py`.
