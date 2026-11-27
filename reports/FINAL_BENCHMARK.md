# Final Consolidated Benchmark Summary (`reports/FINAL_BENCHMARK.md`)

**Date:** 2026-11-27 | **Branch:** `research/data-integrity-and-ml-validation`

---

## 1. Recommendation Engine (Zero-Leakage v2, 500 Test Users, 40,000 Tracks)

| Model | Sampled `NDCG@10` (1+499) | Full-Catalog `HR@10` (40k) | Full-Catalog `NDCG@10` (40k) | `ILD@20` | `Novelty@20` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Popularity (Train-Only) | 0.2556 | 1.20% | 0.0064 | 0.7453 | 0.0731 |
| Item-KNN CF (Train-Only) | 0.5249 | 21.00% | 0.1319 | 0.6389 | 0.3520 |
| BPR-MF | 0.4159 | 5.20% | 0.0284 | 0.7089 | 0.1750 |
| NeuMF — Full Multimodal | 0.6209 | 17.20% | 0.1089 | **0.7627** | 0.2826 |
| Dual-Horizon Transformer | 0.6474 | 43.00% | **0.3568** | 0.6523 | **0.4432** |
| Z-Score Hybrid *(Deprecated)* | 0.6552 | 28.40% | 0.1952 *(regression)* | 0.6296 | 0.3443 |
| **RRF Hybrid (Dual + CF)** | **0.6792** | **45.00%** | 0.3376 *(+72.9% vs Z-Score)* | 0.6523 | 0.4310 |

## 2. Adaptive Shuffle (5 Seeds, Strictly Separated Protocols)

| Protocol & Algorithm | Artist Clump Rate ↓ | Min Artist Gap ↑ | Measured-Only / Smoothness ↑ | Mean BPM Jump ↓ | Discovery Pacing Std ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **P1 (Held-Out Playlists):** Fisher-Yates | 11.79% | 2.04 | 67.31% *(measured)* | 20.75 | — |
| **P1 (Held-Out Playlists):** Artist-Spaced | **4.51%** | **5.93** | 66.92% *(measured)* | 21.44 | — |
| **P1 (Held-Out Playlists):** Adaptive (`w=0.00`) | 7.07% | 3.69 | **69.22%** *(measured)* | **19.94** | — |
| **P1 (Held-Out Playlists):** Adaptive + Urgency (`w=0.75`) | 4.55% | 5.46 | 68.58% *(measured)* | 20.26 | — |
| **P2 (120 Real Users):** Fisher-Yates | 11.34% | 1.69 | 66.09% | 28.97 | 0.1947 |
| **P2 (120 Real Users):** Artist-Spaced | **2.65%** | **4.77** | 65.38% | 29.32 | 0.1786 |
| **P2 (120 Real Users):** Adaptive (`w=0.00`) | 6.53% | 3.22 | **67.41%** | 28.12 | 0.1583 |
| **P2 (120 Real Users):** Adaptive + Urgency (`w=0.75`) | 3.16% | 4.71 | 67.15% | **28.10** | **0.1542** |

## 3. Online Feedback Calibration (60 Real User Profiles, Simulated Stress Test)

| Scenario | Pre-Calibration Dom. Share@20 | **Post-Calibration Dom. Share@20** | Target Share@20 |
| :--- | :---: | :---: | :---: |
| No Feedback (Baseline) | 40.33% | **40.33%** | 0.67% |
| 1 Early Skip | 2.17% *(overreaction)* | **32.42%** *(gentle nudge)* | 0.58% |
| 3 Consecutive Skips | 0.00% | **0.00%** *(full suppression)* | 0.92% |
| 2 Skips → 6 Neutral Plays (Recovery) | 4.83% | **18.25%** *(~4× faster recovery)* | 0.33% |
| Preference Reversal (3 Skips + 3 Likes) | 0.17% | **1.92%** | **46.42%** |
