# Recommendation Engine Validation & Ablation Study (`reports/RECOMMENDATION_VALIDATION.md`)

**Script:** [backend/training/validate_recommendations.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/validate_recommendations.py)  
**Raw Artifact:** [data/reports/recommendation_validation_v2.json](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/data/reports/recommendation_validation_v2.json)  
**Dataset:** `data/processed_v2/` (`40,000` tracks, `929` users, `137,302` train pairs, `4,663` train playlists, `984` held-out test playlists)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (`fp16` AMP, `seed=42`, `eval_seed=123`, `500` test users)

---

## 1. Protocol & Integrity Guarantees
- **Zero Leakage:** Chronological per-user splits, disjoint 70/15/15 playlist splits, training-only measured-audio `StandardScaler`, training-only popularity/novelty, positive-excluded duplicate-free negative sampling, and isolated cloned encoder for Transformer training (no post-Transformer contamination of NeuMF).
- **Consistent Scoring:** Each model uses **one canonical scoring function** identically across Sampled (`1 + 499` shared candidates) and Full-Catalog (`40,000` items) evaluation.
- **Uncertainty & Significance:** 1,000-sample user-level bootstrap 95% CIs and 4,000-permutation paired sign-flip p-values.

---

## 2. Sampled (`1 + 499`) vs. Full-Catalog (`40,000`) Benchmark Results

| Model / Ablation | Sampled `HR@10` | Sampled `NDCG@10` (95% CI) | Full-Catalog `HR@10` (`Recall@10`) | Full-Catalog `NDCG@10` | Full-Catalog `MRR@10` | `ILD@20` | `Novelty@20` | `Coverage@20` | Unseen Artist % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity (Train-Only)** | 0.4520 | 0.2556 `[0.2242, 0.2845]` | 0.0120 | 0.0064 | 0.0047 | 0.7453 | 0.0731 | 0.12% | 81.58% |
| **Item-KNN CF (Train-Only)** | 0.7040 | 0.5249 `[0.4876, 0.5608]` | 0.2100 | 0.1319 | 0.1085 | 0.6389 | 0.3520 | 7.75% | 46.26% |
| **BPR-MF** | 0.6480 | 0.4159 `[0.3811, 0.4497]` | 0.0520 | 0.0284 | 0.0213 | 0.7089 | 0.1750 | 1.69% | 59.83% |
| **NeuMF — GMF Only** | 0.7420 | 0.4584 `[0.4264, 0.4916]` | 0.0700 | 0.0329 | 0.0219 | 0.7362 | 0.2248 | 3.28% | 59.46% |
| **NeuMF — MLP Only** | 0.7840 | 0.4858 `[0.4551, 0.5174]` | 0.0760 | 0.0422 | 0.0319 | 0.6908 | 0.2953 | 6.54% | 49.25% |
| **NeuMF — Interaction Only** | 0.8220 | 0.6065 `[0.5735, 0.6398]` | 0.1500 | 0.0925 | 0.0750 | 0.7528 | 0.2777 | 5.76% | 54.39% |
| **NeuMF — Full Multimodal** | **0.8400** | 0.6209 `[0.5903, 0.6532]` | 0.1720 | 0.1089 | 0.0897 | **0.7627** | 0.2826 | 6.25% | 53.33% |
| **Transformer — Short-Term Only** | 0.7640 | 0.6259 `[0.5882, 0.6599]` | 0.4220 | 0.3492 | 0.3263 | 0.6414 | **0.4476** | **10.98%** | **61.57%** |
| **Dual-Horizon Transformer** | 0.7840 | 0.6474 `[0.6122, 0.6820]` | **0.4300** | **0.3568** | **0.3339** | 0.6523 | 0.4432 | 10.94% | 57.87% |
| **Hybrid (Dual-Horizon + CF)** | 0.8300 | 0.6552 `[0.6217, 0.6849]` | 0.2840 | 0.1952 | 0.1681 | 0.6296 | 0.3443 | 8.12% | 42.05% |
| **Hybrid + Novelty** | 0.8200 | **0.6583** `[0.6234, 0.6894]` | 0.2980 | 0.2024 | 0.1735 | 0.6295 | 0.3558 | 8.42% | 42.13% |
| **Hybrid + Novelty + MMR** | 0.8200 | **0.6583** `[0.6234, 0.6894]` | 0.3040 | 0.2079 | 0.1785 | 0.6920 | 0.3631 | 8.78% | 46.54% |

---

## 3. Paired Statistical Significance Tests (4,000 Sign-Flip Permutations)

| Comparison | Sampled $\Delta$NDCG@10 | Sampled $p$-value | Full-Catalog $\Delta$NDCG@10 | Full-Catalog $p$-value | Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dual-Horizon Transformer vs. NeuMF Multimodal** | `+0.0265` | `0.1522` | **`+0.2479`** | **`0.00025`** | Massive, highly significant full-catalog superiority for sequential Transformer. |
| **Dual-Horizon vs. Short-Term Transformer Only** | **`+0.0215`** | **`0.0010`** | `+0.0076` | `0.2744` | Significant uplift in sampled ranking; modest positive trend in full-catalog. |
| **Hybrid (Dual-Horizon + CF) vs. Dual-Horizon** | `+0.0078` | `0.5456` | **`-0.1616`** | **`0.00025`** | **SIGNIFICANT FULL-CATALOG REGRESSION!** |
| **Hybrid (Dual-Horizon + CF) vs. Item-KNN CF** | **`+0.1303`** | **`0.00025`** | **`+0.0633`** | **`0.00025`** | Highly significant uplift over Item-KNN CF alone. |
| **NeuMF Multimodal vs. NeuMF Interaction Only** | `+0.0144` | `0.3119` | `+0.0165` | `0.2069` | Consistent positive trend (+17.7% full-catalog) from audio+genre modalities, not yet significant at $N=500$. |

---

## 4. Key Scientific Findings & Caught Regressions
1. **Causal Left-Padding `NaN` Bug Fixed:** Unit testing (`test_transformer_masks.py`) uncovered a real PyTorch `nn.TransformerEncoder` NaN propagation bug on left-padded causal sequences; fixing the combined per-head attention mask enabled `Dual-Horizon Transformer` to reach **`43.0%` Full-Catalog HR@10** and **`0.3568` Full-Catalog NDCG@10** across all 40,000 tracks.
2. **NeuMF Ablation Validated:** Combining GMF + MLP (`0.1089` full NDCG@10) dramatically outperforms GMF-only (`0.0329`) and MLP-only (`0.0422`).
3. **Hybrid Z-Score Blending Regression Caught by Full-Catalog Evaluation:** Adding z-scored sparse `Item-KNN CF` to `Dual-Horizon Transformer` **degraded Full-Catalog NDCG@10 by `-45.3%` (`0.3568 → 0.1952`, $p = 0.00025$)** because sparse CF produces extreme positive z-score spikes on a handful of co-occurring items, hijacking the top-10 of a 40,000-item catalog — an error completely invisible under `1 + 499` sampled evaluation! **Recommendation:** Use `Dual-Horizon Transformer` as the primary full-catalog ranker, or replace z-score blending with **Reciprocal Rank Fusion (RRF)** and validate on full-catalog ranking before enabling CF blending.
