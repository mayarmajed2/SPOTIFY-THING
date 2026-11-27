# Adaptive Shuffle Optimization Experiment — Remaining-Quota Urgency

## 1. Hypothesis & Experimental Setup

In our baseline benchmark (Suite B), forward causal sequencing (`Original Adaptive Shuffle`) outperformed both Fisher-Yates Random and Spotify-Style Artist-Spacing on Genre Switch Rate, Acoustic Flow Smoothness, BPM Continuity, Discovery Pacing, and Session NDCG (+71%), but exhibited higher artist clumping (`7.41%`) than global proportional artist spacing (`4.76%`) due to end-of-queue pileups on artist-heavy playlists.

**Hypothesis:** Adding a **Remaining-Quota Urgency** bonus behind an opt-in configuration flag (`ShuffleConfig.quota_urgency_weight`, defaulting to `0.0` so the original algorithm is preserved identically) could pull heavily represented artists forward as soon as their cooldown expires, reducing end-of-queue clumping.

**Experimental Controls:**
- **Playlists:** Exact same 200 real Spotify MPD playlists (`num_tracks >= 20`, capped at 50 tracks per pool): `85` artist-heavy playlists (`artist_concentration >= 0.40`) and `115` balanced multi-artist playlists (`artist_concentration < 0.40`).
- **Inputs:** Identical candidate tracks, `NeuMF` long-term relevance scores, and discovery masks (`rng(99)`).
- **Random Seeds:** 5 independent seeds (`[1000, 2000, 3000, 4000, 5000]`), totaling **7,000 shuffle evaluations** across 7 variants.
- **Reproducibility:** Run `backend\.venv\Scripts\python.exe -m backend.training.run_shuffle_experiment` (raw JSON output saved at `data/reports/shuffle_experiment_report.json`).

---

## 2. Overall Results Across All 200 Playlists (Mean ± Std over 5 Seeds)

| Algorithm | Artist Clump Rate ↓ | Min Artist Gap ↑ | Median Artist Gap ↑ | Session NDCG ↑ | Genre Switch % ↑ | Window Genre Div ↑ | Local ILD (W=5) ↑ | Acoustic Smoothness ↑ | Mean BPM Jump ↓ | Discovery Pacing Std ↓ | Valid % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fisher-Yates Random** | 11.59 ± 0.12% | 2.03 ± 0.08 | 7.38 ± 0.11 | 0.3060 ± 0.0059 | 75.42 ± 0.18% | 0.7165 ± 0.0009 | 0.3832 ± 0.0011 | 70.56 ± 0.08% | 21.49 ± 0.16 | 0.1922 ± 0.0022 | 100% |
| **Artist-Spacing Heuristic** | **4.76 ± 0.06%** | **6.24 ± 0.07** | **11.51 ± 0.06** | 0.3045 ± 0.0068 | 78.63 ± 0.21% | 0.7428 ± 0.0010 | **0.3882 ± 0.0013** | 69.93 ± 0.11% | 21.83 ± 0.10 | 0.1885 ± 0.0047 | 100% |
| **Greedy Relevance** | 15.87 ± 0.00% | 1.61 ± 0.00 | 5.15 ± 0.00 | **0.7400 ± 0.0000** | 73.08 ± 0.00% | 0.6966 ± 0.0000 | 0.2966 ± 0.0000 | 71.33 ± 0.00% | 21.02 ± 0.00 | 0.1920 ± 0.0000 | 100% |
| **Original Adaptive Shuffle (`w=0.00`)** | 7.41 ± 0.16% | 3.55 ± 0.09 | 8.19 ± 0.09 | **0.5186 ± 0.0121** | 82.31 ± 0.10% | 0.7781 ± 0.0008 | 0.3588 ± 0.0010 | **71.86 ± 0.03%** | **20.63 ± 0.08** | 0.1588 ± 0.0013 | 100% |
| **Adaptive + Urgency (`w=0.40`)** | 5.47 ± 0.07% | 4.73 ± 0.07 | 8.95 ± 0.12 | 0.5100 ± 0.0093 | 83.32 ± 0.25% | 0.7835 ± 0.0011 | 0.3597 ± 0.0009 | 71.67 ± 0.04% | 20.66 ± 0.08 | **0.1586 ± 0.0011** | 100% |
| **Adaptive + Urgency (`w=0.75`)** | 5.06 ± 0.02% | 5.13 ± 0.08 | 9.33 ± 0.12 | 0.5032 ± 0.0085 | 83.60 ± 0.14% | 0.7843 ± 0.0010 | 0.3614 ± 0.0006 | 71.52 ± 0.08% | 20.76 ± 0.12 | 0.1605 ± 0.0021 | 100% |
| **Adaptive + Urgency (`w=1.20`)** | 4.91 ± 0.03% | 5.29 ± 0.06 | 9.69 ± 0.11 | 0.4970 ± 0.0096 | **83.71 ± 0.10%** | **0.7853 ± 0.0008** | 0.3626 ± 0.0006 | 71.47 ± 0.08% | 20.74 ± 0.17 | 0.1610 ± 0.0008 | 100% |

*(Note: Global Intra-List Diversity `ILD` is invariant across permutations of a fixed track set at `0.3833`; `Local ILD (W=5)` measures diversity within consecutive 5-track listening windows.)*

---

## 3. Stratified Analysis: Balanced vs. Artist-Heavy Playlists

### A. Balanced Multi-Artist Playlists (`115` playlists, artist concentration `< 40%`)
| Variant | Artist Clump Rate ↓ | Min Artist Gap ↑ | Session NDCG ↑ | Acoustic Smoothness ↑ | Mean BPM Jump ↓ | Genre Switch % ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original Adaptive (`w=0.00`) | 1.24 ± 0.13% | 4.98 ± 0.19 | 0.5815 ± 0.0112 | 70.47 ± 0.05% | 21.63 ± 0.12 | 94.55 ± 0.25% |
| **Adaptive + Urgency (`w=0.40`)** | **0.21 ± 0.04%** | **6.40 ± 0.08** | **0.5810 ± 0.0091** | **70.51 ± 0.03%** | **21.45 ± 0.14** | **95.26 ± 0.19%** |
| Adaptive + Urgency (`w=0.75`) | 0.10 ± 0.02% | 6.83 ± 0.13 | 0.5768 ± 0.0080 | 70.40 ± 0.07% | 21.63 ± 0.20 | 95.51 ± 0.14% |

### B. Artist-Heavy Playlists (`85` playlists, artist concentration `>= 40%`)
| Variant | Artist Clump Rate ↓ | Min Artist Gap ↑ | Session NDCG ↑ | Acoustic Smoothness ↑ | Mean BPM Jump ↓ | Genre Switch % ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Original Adaptive (`w=0.00`) | 15.76 ± 0.24% | 1.62 ± 0.06 | **0.4333 ± 0.0174** | **73.75 ± 0.06%** | **19.26 ± 0.19** | 65.75 ± 0.28% |
| Adaptive + Urgency (`w=0.40`) | 12.58 ± 0.15% | 2.47 ± 0.06 | 0.4140 ± 0.0124 | 73.24 ± 0.08% | 19.58 ± 0.14 | 67.17 ± 0.58% |
| Adaptive + Urgency (`w=0.75`) | 11.77 ± 0.04% | 2.84 ± 0.03 | 0.4036 ± 0.0148 | 73.03 ± 0.11% | 19.59 ± 0.16 | 67.48 ± 0.19% |
| Adaptive + Urgency (`w=1.20`) | **11.44 ± 0.08%** | **2.96 ± 0.06** | 0.3943 ± 0.0135 | 73.03 ± 0.11% | 19.55 ± 0.17 | **67.50 ± 0.18%** |

---

## 4. Honest Assessment: Tradeoffs, Regressions & Recommendation

1. **Significant Artist Spacing Gains:** Across all 200 playlists, `w=0.40` reduces artist clump rate by **26.2%** (`7.41% → 5.47%`) and raises minimum artist gap by **33.2%** (`3.55 → 4.73`), while `w=0.75` cuts clumping by **31.7%** (`7.41% → 5.06%`, within `0.30 pp` of global Artist-Spacing's `4.76%`) while maintaining **+65.3% higher Session NDCG** (`0.5032` vs `0.3045`) and **+1.59 pp higher Acoustic Smoothness** (`71.52%` vs `69.93%`).
2. **Real Tradeoffs / Regressions (Not a Free Lunch):** On artist-heavy playlists (`>= 40%`), quota urgency introduces a measurable regression in **Session NDCG** (`0.4333 → 0.4036` at `w=0.75`, `-6.9%`) and **Acoustic Smoothness** (`73.75% → 73.03%`, `-0.72 pp`) because urgent artists are promoted earlier even when another artist has slightly higher immediate relevance or BPM match. Beyond `w=0.75`, clump reduction flattens (`5.06% → 4.91%`) while NDCG continues to decline.
3. **Engineering Decision:** Keep **Original Adaptive Shuffle (`quota_urgency_weight=0.0`)** as the preserved default, and expose `quota_urgency_weight` (`0.40–0.75` recommended when minimizing artist clumps is prioritized) as an opt-in experimental configuration parameter.
