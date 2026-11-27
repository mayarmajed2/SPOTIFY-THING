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

## 3. Critical Finding & Post-Calibration Verification
- **Pre-Fix Overreaction:** Before calibration, a single `early_skip` dropped the user's dominant genre share in Top-20 from `40.33%` to `2.17%`, and 6 neutral plays only recovered it to `4.83%`.
- **Calibration Applied ([online_feedback.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/app/engine/online_feedback.py)):**
  - Require `>= 2` skips of the same genre (`genre_skip_counts[g] >= 2`) before activating genre-wide fatigue (1 skip remains strictly track- and artist-local with gentler `step_lr = 0.10`).
  - Accelerated neutral-play fatigue decay (`0.90 → 0.78` for genres, `0.88 → 0.80` for artists).
- **Verified Post-Fix Results (`phase_e_online_feedback_calibrated`):**
  - **Scenario 2 (1 early skip):** Dominant genre share@20 = **`32.42%`** (gentle `-7.9 pp` nudge from `40.33%` baseline, `||Δu|| = 0.075`) — overreaction eliminated!
  - **Scenario 3 (3 consecutive skips):** Dominant genre share@20 = **`0.00%`** — sustained skip suppression preserved!
  - **Scenario 7 (2 skips → 6 neutral plays):** Dominant genre share@20 recovers nearly 4× faster (**`4.83% → 18.25%`**)!
  - **Scenario 9 (Preference reversal: 3 skips + 3 likes):** Dominant `1.92%` → Target **`46.42%`**!
- **Persistence Verified:** `SQLiteFeedbackStore` (`WAL` mode + `BEGIN IMMEDIATE`) persists `genre_skip_counts` alongside all LinUCB and fatigue state, passing all restart-recovery, user-isolation, and multi-threaded concurrency tests.
