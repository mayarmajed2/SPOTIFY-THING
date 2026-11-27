# Data Splitting & Reproducibility (`docs/DATA_SPLITTING_AND_REPRODUCIBILITY.md`)

**Implementation:** [backend/training/splits_and_sampling.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/splits_and_sampling.py)  
**Automated Enforcement:** [backend/tests/test_data_leakage.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/tests/test_data_leakage.py) & [backend/tests/test_negative_sampling.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/tests/test_negative_sampling.py)

---

## 1. Splitting Protocols

| Artifact | Split Strategy | Proportions / Rule | Leakage Guarantees |
| :--- | :--- | :--- | :--- |
| **User Interactions** (`interactions_v2.parquet`) | Chronological Leave-Last-Two-Out per user | `train`: `[0 .. N-3]`; `val`: `N-2`; `test`: `N-1` (users with `N >= 10`) | Asserts `max(train_ts) <= val_ts <= test_ts`; held-out items never enter `train_items` or `seq_train_samples`. |
| **Playlists** (`playlists_v2.parquet`) | Deterministic playlist-level split (`seed=42`) | **70% `train`** (`4,663`) / **15% `val`** (`1,011`) / **15% `test`** (`984`) | `train_pids`, `val_pids`, `test_pids` are strictly disjoint; only `train_playlists` feed `ItemCooccurrenceCF` and encoder co-occurrence pairs. |
| **Audio Feature Normalization** | Training-only measured `StandardScaler` | Fitted strictly on `(item_idx in train_items) & (~is_synthetic)` | Perturbing test-only or synthetic tracks leaves scaler `(mean, std)` strictly invariant. |
| **Popularity & Novelty** | Training-period counts only | Computed from `train_items` + `train_playlists` only | Zero future/test interaction counts enter popularity baseline or `Novelty@20` exposure probabilities. |

## 2. Negative Sampling Protocol
- **Training (`PositiveExcludedPairDataset`):** For each `(u, pos)`, draws `k = 4` **unique** negative items without replacement strictly outside `user_train_pos_sets[u]`.
- **Sampled Evaluation (`build_shared_eval_candidates`):** Pre-generates a single deterministic `[N_eval, 500]` candidate matrix (`seed=123`) shared identically across every evaluated model: index `0` is `target`; indices `1..499` are unique items outside `seen_pos | {target}`.
- **Full-Catalog Evaluation:** Ranks all `40,000` catalog tracks (masking `seen_pos - {target}`), reported separately alongside sampled metrics so protocols are never conflated.

## 3. Seeds & Hardware Reproducibility
- Fixed seeds: `seed = 42` (splits & model init), `seed = 123` (shared eval candidates), `seeds = [1000, 2000, 3000, 4000, 5000]` (5-seed shuffle benchmarks).
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, CUDA 12.4, `torch==2.6.0+cu124`, `fp16` AMP). Note: minor non-determinism (`~1e-4`) can occur across GPU driver versions with cuBLAS/AMP reductions; CPU evaluation and NumPy sampling are bit-for-bit deterministic.
