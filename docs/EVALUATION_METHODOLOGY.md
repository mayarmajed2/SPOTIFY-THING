# Evaluation Methodology & Metric Definitions (`docs/EVALUATION_METHODOLOGY.md`)

---

## 1. Recommendation Ranking Metrics (Phase C)
Evaluated separately under **Sampled Ranking (`1` ground-truth target + `499` unique unseen negatives, shared identically across all models)** and **Full-Catalog Ranking (`40,000` tracks masking `seen_pos - {target}`)**:

- **`HR@K` / `Recall@K`:** Fraction of test users whose held-out next track appears in the top-$K$ (`K=10`). With 1 held-out target per user, `HR@10 == Recall@10`.
- **`Precision@K`:** `HR@K / K` (with 1 held-out target per user, upper bound is `0.10` at `K=10`).
- **`NDCG@K`:** $\frac{1}{\log_2(\text{rank} + 1)}$ if $\text{rank} \le K$ else $0$, averaged across users.
- **`MRR@K`:** $\frac{1}{\text{rank}}$ if $\text{rank} \le K$ else $0$, averaged across users.
- **`ILD@20` (Intra-List Diversity):** Mean pairwise cosine distance $1 - \langle \mathbf{e}_i, \mathbf{e}_j \rangle$ over all $\binom{20}{2} = 190$ pairs in Top-20.
- **`Novelty@20`:** Mean normalized self-information $-\log_2 P_{\text{train}}(i)$ computed strictly from **training-period** exposure counts.
- **`CatalogCoverage@20`:** Fraction of unique catalog tracks appearing at least once across all users' Top-20 lists.
- **`UnseenArtistRatio@20`:** Fraction of Top-20 recommended tracks whose primary artist never appeared in the user's training history.

### Statistical Uncertainty & Significance
- **95% Confidence Intervals:** 1,000-sample user-level nonparametric bootstrap (`seed=42`).
- **Significance Tests:** Two-sided 4,000-permutation paired sign-flip test on per-user `NDCG@10` differences (exact nonparametric p-value).

---

## 2. Adaptive Shuffle Metrics (Phase D)
Evaluated strictly under separated protocols (**Protocol 1: Held-Out Playlist-Only**, **Protocol 2: User-Grounded Real Last.fm Users**) across 5 random seeds (`[1000, 2000, 3000, 4000, 5000]`):

- **`artist_clump_rate`:** Fraction of consecutive transitions where the same artist reappears within `<= 2` tracks.
- **`min / p25 / median / mean / max_artist_gap`:** Distribution of track-index intervals between consecutive plays of the same artist.
- **`genre_switch_rate` & `window_genre_diversity`:** Rate of genre changes between adjacent tracks and mean unique-genre ratio in 5-track windows.
- **`acoustic_smoothness` & `measured_audio_smoothness`:** Mean transition flow score (`0.50 * mood + 0.35 * harmonic_BPM + 0.15 * Circle_of_Fifths_key`), reported separately strictly on pairs where **both** tracks have measured audio (`is_synthetic == False`).
- **`predicted_relevance_ndcg`:** Discounted ordering alignment with predicted model relevance scores (explicitly NOT ground-truth human satisfaction).
- **`discovery_pacing_std`:** Standard deviation of rolling 5-track discovery fraction (lower = smoother interleaving).
- **`order_variation_across_seeds`:** Mean $1 - |\tau_{\text{Kendall}}|$ between seed 1000 and seed 2000 orderings.
- **`sequence_validity`:** `1.0` iff output is an exact permutation of `0..M-1` with zero dropped or duplicated tracks.
