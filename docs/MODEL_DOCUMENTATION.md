# Model Architecture Documentation (`docs/MODEL_DOCUMENTATION.md`)

---

## 1. Shared Multimodal Track Encoder (`backend/app/models/embeddings.py`)
- **Input:** `item_id` (0..39,999; `40,000` = pad), `artist_idx`, `genre_idx`, and 12D standardized audio features (`danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness`, `liveness`, `loudness`, `key`, `mode`, `explicit_num`).
- **Architecture:**
  - `item_emb`: `Embedding(40001, 64)` with `id_dropout_prob = 0.15` during training (forces cold-start alignment between content modalities and collaborative ID space).
  - `artist_emb`: `Embedding(num_artists + 1, 32)`
  - `genre_emb`: `Embedding(num_genres + 1, 16)`
  - `audio_mlp`: `Linear(12 → 48) → GELU → Linear(48 → 32)`
  - `fusion_proj`: `Linear(144 → 128) → GELU → Linear(128 → 128) → LayerNorm(128) → L2-Normalize`
- **Output:** Unit-norm track embedding $\mathbf{e}_i \in \mathbb{R}^{128}$.

## 2. Collaborative Filtering & NeuMF (`backend/app/models/collaborative.py`)
- **`ItemCooccurrenceCF`:** Sparse item-item cosine similarity over log-TF user-item + training-split playlist co-occurrence matrix, pruned to `top_k_neighbors = 80` per item.
- **`BPRMatrixFactorization`:** `64`-dim user/item embeddings + item bias trained with Bayesian Personalized Ranking loss (`-log σ(s_pos - s_neg)`) using positive-excluded negative sampling.
- **`NeuralCollaborativeFiltering` (`NeuMF`):**
  - **GMF Branch:** `Embedding(num_users, 64) ⊙ Embedding(num_items, 64)`
  - **MLP Branch:** `Linear(384 → 192) → GELU → Dropout(0.15) → Linear(192 → 64) → GELU` over `[u_long || e_i || u_long ⊙ e_i]`
  - **Head:** `Linear(128 → 1)` interaction logit + `u_long ∈ R^128` aligned in the shared track embedding space.
  - **Canonical Scoring:** Raw `neu_head` logits used identically across validation, sampled eval, and full-catalog eval.

## 3. Dual-Horizon Causal Session Transformer (`backend/app/models/sequential_transformer.py`)
- **Inputs:** Rolling causal history of length `L = 50` (`item_seq`, `action_seq`, `time_bucket_seq`).
- **Backbone:** 2-layer, 4-head Pre-LN Transformer Encoder (`d_model=128`, `dim_feedforward=256`, `GELU`, `dropout=0.15`).
- **Causal Left-Padding Mask:** Combined per-head boolean mask `attn_mask = causal_mask | pad_mask` with diagonal unmasked (`attn_mask[:, diag, diag] = False`) so padded queries attend only to themselves (zero `NaN`s) while valid queries never attend to future tokens or padding keys.
- **Gated Fusion:** `g = σ(W_g [u_long || u_short || u_long ⊙ u_short])`, `u_fused = Normalize(g ⊙ u_long + (1 - g) ⊙ u_short)`.
- **Loss:** `0.70 * InfoNCE(u_fused, pos/neg_pool, τ=0.08) + 0.30 * InfoNCE(u_short, pos/neg_pool, τ=0.08)` with false-negative batch masking.

## 4. Hybrid Ranking & Online Adaptation (`recommender.py`, `online_feedback.py`)
- **Reciprocal Rank Fusion (RRF):** Bounded rank scores `(k+1)/(k+rank)` (`k=60`) fusing `0.78 * RRF(Dual-Horizon) + 0.22 * RRF(Item-KNN CF)` — eliminating sparse CF score-spike regressions on full-catalog ranking.
- **Online Feedback (`UserFeedbackState`):** Diagonal Ridge LinUCB (`A_diag`, `b`) + signed EMA residual (`delta_u`, capped at `||Δu|| <= 0.65`) + artist/genre fatigue memory requiring `>= 2` skips before activating genre-wide fatigue, persisted atomically via WAL-mode `SQLiteFeedbackStore`.

## 5. Adaptive Shuffle (`backend/app/engine/adaptive_shuffle.py`)
- Sequential Boltzmann-sampled multi-objective sequencer optimizing relevance, Circle-of-Fifths + harmonic BPM + mood transition flow, sliding-window max-min embedding diversity, exponential artist/genre cooldown penalties, discovery pacing, and opt-in `quota_urgency_weight` (default `0.0`).
