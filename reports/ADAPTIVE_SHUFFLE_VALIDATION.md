# Adaptive Shuffle Validation Report (`reports/ADAPTIVE_SHUFFLE_VALIDATION.md`)

**Script:** [backend/training/validate_shuffle_and_feedback.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/validate_shuffle_and_feedback.py)  
**Raw Artifact:** [data/reports/shuffle_and_feedback_validation_v2.json](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/data/reports/shuffle_and_feedback_validation_v2.json)  
**Seeds:** 5 independent seeds (`[1000, 2000, 3000, 4000, 5000]`)

---

## 1. Metric Semantics Correction
- Renamed `session_ndcg` → `predicted_relevance_ndcg` (while retaining the legacy key alias for API backward compatibility) to state explicitly that it measures **discounted alignment with predicted model relevance scores**, NOT ground-truth human enjoyment or user satisfaction.
- Strictly separated **Protocol 1 (Held-Out Playlist-Only)** from **Protocol 2 (User-Grounded Real Last.fm Users)** so non-personalized playlists are never conflated with user-grounded personalization.
- Added **`measured_audio_smoothness`** (computed strictly on consecutive pairs where **both** tracks have `is_synthetic == False`) and **`order_variation_across_seeds`** (`1 - |Kendall tau|` between seed 1000 and seed 2000).

---

## 2. Protocol 1 — Held-Out Playlist-Only Sequencing (`200` Strictly Unseen `test_playlists`, Neutral Relevance)

| Algorithm | Artist Clump Rate ↓ | Min Artist Gap ↑ | Genre Switch % ↑ | All-Track Smoothness ↑ | **Measured-Only Smoothness** ↑ | Mean BPM Jump ↓ | Order Variation (Seeds) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher-Yates Random** | 11.79 ± 0.20% | 2.04 | 73.8% | 71.44% | 67.31% | 20.75 | 0.901 |
| **Artist-Spacing Heuristic** | **4.51 ± 0.11%** | **5.93** | 76.7% | 70.36% | 66.92% | 21.44 | 0.894 |
| **Original Adaptive (`w=0.00`)** | 7.07 ± 0.17% | 3.69 | 80.6% | **72.60%** | **69.22%** | **19.94** | 0.771 |
| **Adaptive + Urgency (`w=0.40`)** | 5.00 ± 0.05% | 5.07 | 81.6% | 72.13% | 69.00% | 20.40 | 0.802 |
| **Adaptive + Urgency (`w=0.75`)** | 4.55 ± 0.05% | 5.46 | 81.9% | 72.07% | 68.58% | 20.26 | 0.803 |
| **Adaptive + Urgency (`w=1.20`)** | **4.42 ± 0.02%** | 5.56 | **82.0%** | 71.94% | 68.86% | 20.35 | 0.790 |

**Findings (Protocol 1):**
- Even on **100% unseen `test_playlists`** and restricting acoustic scoring strictly to **measured audio pairs (`is_synthetic == False`)**, Adaptive Shuffle outperforms Artist-Spacing by **`+1.66` to `+2.30` percentage points** in acoustic smoothness (`68.58%–69.22%` vs `66.92%`), lowers BPM jumps (`19.94–20.26` vs `21.44`), and maintains strong seed-to-seed unpredictability (`0.77–0.80` Kendall distance).
- `Adaptive + Urgency (w=0.75)` matches Artist-Spacing's clump rate (`4.55%` vs `4.51%`) while preserving acoustic coherence gains.

---

## 3. Protocol 2 — User-Grounded Historical Session Shuffle (`120` Real Last.fm 1K Users, True Familiar/Discovery Pools)

| Algorithm | Artist Clump Rate ↓ | Min Artist Gap ↑ | Predicted Relevance NDCG ↑ | Acoustic Smoothness ↑ | Mean BPM Jump ↓ | Discovery Pacing Std ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher-Yates Random** | 11.34 ± 0.22% | 1.69 | 0.9182 | 66.09% | 28.97 | 0.1947 |
| **Artist-Spacing Heuristic** | **2.65 ± 0.14%** | **4.77** | 0.9224 | 65.38% | 29.32 | 0.1786 |
| **Greedy Relevance** | 20.97 ± 0.00% | 1.15 | **1.0000** | 67.10% | 28.91 | 0.3831 |
| **Original Adaptive (`w=0.00`)** | 6.53 ± 0.08% | 3.22 | 0.9452 | **67.41%** | 28.12 | 0.1583 |
| **Adaptive + Urgency (`w=0.40`)** | 4.12 ± 0.09% | 4.13 | **0.9453** | 67.17% | 28.14 | 0.1544 |
| **Adaptive + Urgency (`w=0.75`)** | 3.16 ± 0.04% | 4.71 | 0.9447 | 67.15% | **28.10** | **0.1542** |

**Findings (Protocol 2):**
- On real user-grounded pools (20 historical familiar tracks + 12 unseen discovery tracks), `Adaptive + Urgency (w=0.75)` cuts artist clumping by **51.6%** vs. Original Adaptive (`6.53% → 3.16%`), matches Artist-Spacing's Min Artist Gap (`4.71` vs `4.77`), achieves the **lowest BPM jump (`28.10`)** and **smoothest discovery pacing (`0.1542`)**, with virtually zero drop in `predicted_relevance_ndcg` (`0.9452 → 0.9447`).
- **Limitation:** True human listening satisfaction requires an online A/B study with real listeners; offline proxies cannot prove subjective enjoyment.
