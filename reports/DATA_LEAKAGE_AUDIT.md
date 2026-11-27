# Data Leakage & Evaluation Validity Audit (`reports/DATA_LEAKAGE_AUDIT.md`)

**Audit Date:** 2026-11-27  
**Branch:** `research/data-integrity-and-ml-validation`  
**Primary Audited Files:**
- [backend/training/train_and_evaluate.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/training/train_and_evaluate.py)
- [backend/data_pipeline/download_and_build.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/data_pipeline/download_and_build.py)
- [backend/app/models/collaborative.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/app/models/collaborative.py)
- [backend/app/engine/adaptive_shuffle.py](file:///c:/Users/Mayoura/Desktop/SPOTIFY-THING/backend/app/engine/adaptive_shuffle.py)

---

## 1. Confirmed Leakage Paths

| ID | Severity | File & Lines | Leakage Mechanism | Empirical Reproduction | Required Fix |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **L1** | **CRITICAL** | `train_and_evaluate.py:820, 851, 891` | **100% Playlist Train/Test Overlap:** All `6,932` MPD playlists were passed into `ItemCooccurrenceCF.fit()` and the first `120,000` adjacent pairs into `MultimodalTrackEncoder` training, while `playlists_df[num_tracks>=20].head(200)` were evaluated in Suite B. | **`200 / 200` (100%)** test playlists overlapped with CF training; **`200 / 200` (100%)** overlapped with encoder training. | Deterministic **70% / 15% / 15% playlist-level split** (`train_playlists`, `val_playlists`, `test_playlists`) created BEFORE any CF, encoder, or feature computation. |
| **L2** | **CRITICAL** | `download_and_build.py:405, 421` | **Future Repeat Count Leakage into Action Labels:** `ut_counts = matched_events.groupby(["user_idx", "item_idx"])["ts"].transform("count")` counted plays across the user's *entire* timeline (including future val/test plays) to assign `action_id = 4` (`favorite_listen`) to early training events. | `193,084` events (`58.9%`) received future-leaked `action_id = 4`. | Remove `favorite_listen` entirely; any `repeated_track_event` flag must use strictly causal `cumcount() > 0` within the training prefix. |
| **L3** | **HIGH** | `download_and_build.py:330–369`, `train_and_evaluate.py:437–442` | **Full-Timeline Popularity, Exposure & Feature Normalization Leakage:** `lfm_count` (used for catalog selection and `Novelty@20` exposure) summed all train+val+test Last.fm events. Audio z-score normalization (`norm_*`) was fitted across the full catalog (including synthetic audio rows and test-only tracks). | Test-period interactions contributed to `lfm_count`, `activity_score`, `Novelty@20`, and `norm_*` scalers. | Fit audio `StandardScaler` strictly on **training-eligible measured-audio tracks** (`is_synthetic == False`); compute popularity & exposure novelty strictly from **training-period interactions**. |
| **L4** | **HIGH** | `train_and_evaluate.py:359, 871–875` | **Post-Transformer Encoder Contamination of NeuMF:** `train_dual_horizon_transformer` fine-tunes `track_encoder` in-place after `NeuMF` finishes training; Suite A then evaluates `NeuMF` using the Transformer-fine-tuned `item_embeddings`. | `NeuMF` evaluation did not reflect standalone `NeuMF` weights. | Snapshot `neumf_embeddings` immediately after `NeuMF` training (or train Transformer on a cloned encoder) so every model is evaluated strictly on its own learned parameters. |

---

## 2. Confirmed Evaluation Validity Bugs

### Bug E1 (CRITICAL) — Non-User-Grounded Shuffle Evaluation Misrepresented as Personalized
- **Location:** `train_and_evaluate.py:619–628`
- **Code:** `u_vec = all_u_long[idx % num_users]` and `is_disc = rng.random(len(pool)) < 0.35`.
- **Root Cause:** Spotify MPD playlists are anonymous (`pid` only) and have **no valid user-ID mapping** to Last.fm 1K users. Pairing playlist `idx` with Last.fm user `idx % num_users` and assigning discovery via 35% coin flip is a **synthetic simulation**, NOT user-grounded personalization.
- **Additionally (`adaptive_shuffle.py:166–169`):** `session_ndcg` ranks the model's own predicted cosine scores (`relevance_scores`), NOT observed user satisfaction or ground-truth listening enjoyment.
- **Required Fix:**
  1. Rename `session_ndcg` → `predicted_relevance_ndcg` (explicitly documented as ordering alignment with predicted relevance scores, never user satisfaction).
  2. Strictly separate three shuffle benchmarks:
     - **Protocol 1 — Playlist-Only Sequencing Evaluation** (on held-out `test_playlists`, zero invented user embeddings).
     - **Protocol 2 — User-Grounded Historical Session Shuffle** (on real Last.fm 1K user listening pools where true user history defines familiarity vs. discovery!).
     - **Protocol 3 — Synthetic User Shuffle Simulation** (explicitly labeled synthetic).

### Bug E2 (HIGH) — False-Negative Collisions & Duplicate Negatives in Sampling
- **Location:** `train_and_evaluate.py:79, 377, 465–474`
- **Empirical Reproduction:**
  - **Training (`InteractionPairDataset`):** `np.random.randint(0, num_items, size=4)` draws with replacement and without excluding known user positives (`avg 152.2` positives/user → **1.52% false-negative rate** per draw).
  - **Sampled Evaluation (`run_recommendation_benchmarks`):** `rng.integers(0, num_items, size=550)` draws with replacement without deduplication → **`583 / 600` (97.2%)** of evaluated users had duplicate negatives (avg **3.15 duplicates/user**).
- **Required Fix:**
  - Training negative sampler must exclude `user_train_pos_sets[u]` and sample without replacement.
  - Evaluation must pre-generate a **shared, deterministic, duplicate-free negative candidate set** per user (excluding all prior positives and target) identical across all models, AND report **both Sampled (`1 + 499`) and Full-Catalog (`40,000` items) ranking metrics** separately!

### Bug E3 (HIGH) — NeuMF & Hybrid Scoring Function Inconsistency
- **Location:** `train_and_evaluate.py:495–524`
- **Confirmed:**
  - `NeuMF` used `0.5 * sigmoid(neumf_logits) + 0.5 * cosine(u_long, e_i)` for sampled ranking (`line 512`), but `cosine(u_long, e_i)` alone for full-catalog Top-20 (`line 521`).
  - `Proposed Hybrid` used `s_hyb_acc` (no novelty, no MMR) for sampled accuracy (`line 514`), but `s_hyb_disc` (with novelty + MMR) for full-catalog diversity/novelty (`line 523`).
- **Required Fix:** Define a single canonical `score_all(u)` / `score_candidates(u, items)` interface per model used identically across validation, sampled evaluation, full-catalog evaluation, and serving.

---

## 3. Prioritized Correction Plan

1. **Stage 3 (Phase A — Data Pipeline v2):** Build `backend/data_pipeline/build_verified_dataset.py` (writing to `data/processed_v2/` without touching `data/processed/`): PyArrow/Pandas portable join, version-aware entity resolution with confidence tiers, explicit audio provenance (`audio_feature_source`, `audio_feature_confidence`, `is_synthetic`), configurable sessionization with sensitivity analysis, and honest observed-only interaction schema (`track_listened_event`, causal `repeated_track_event`).
2. **Stage 4 (Phase B — Zero-Leakage Splits, Negative Sampling & Automated Leakage Tests):** Implement deterministic chronological user splits + 70/15/15 playlist splits + training-only scalers/popularity + duplicate-free positive-excluded negative sampling + `backend/tests/test_data_leakage.py` & `test_negative_sampling.py`.
3. **Stage 5 (Phase C — Model Validation & Ablations):** Fix NeuMF scoring consistency, snapshot encoders cleanly, verify Transformer causal masking (`test_transformer_masks.py`), run 5-seed + user-level bootstrap NeuMF & Hybrid ablations (both sampled and full-catalog) → `reports/RECOMMENDATION_VALIDATION.md` & `reports/RECOMMENDATION_ABLATION_STUDY.md`.
4. **Stage 6 (Phase D — Adaptive Shuffle Validation):** Rename `session_ndcg` → `predicted_relevance_ndcg`, separate Playlist-Only vs. User-Grounded vs. Synthetic protocols, audit `quota_urgency_weight` across playlist regimes (artist-heavy, balanced, genre-heavy, short, long) → `reports/ADAPTIVE_SHUFFLE_VALIDATION.md`.
5. **Stage 7 (Phase E & F — Online Feedback Stress Tests & Compliance):** Run all 10 feedback scenarios + 4 ablations (`reports/ONLINE_FEEDBACK_VALIDATION.md`) and official licensing/Spotify compliance audit (`docs/DATASET_LICENSE_AUDIT.md`, `docs/SPOTIFY_COMPLIANCE.md`).
6. **Stage 8–9 (Phase G & Final Audit):** Run complete test suite + frontend build, write all technical docs & `reports/FINAL_TECHNICAL_AUDIT.md`.
