# Models, Shuffle, and Feedback Audit

## Model Mapping

- `ItemCooccurrenceCF` builds sparse item-item similarity from training positive interactions and playlist adjacency.
- `BPRMatrixFactorization` trains 64D user/item factors with pairwise BPR loss.
- `NeuralCollaborativeFiltering` combines GMF and MLP interactions with a 128D long-term taste embedding.
- `MultimodalTrackEncoder` combines item, artist, genre, and standardized 12D audio representations.
- `DualHorizonSessionTransformer` combines a causal two-layer sequence context with action and time embeddings, then gates it against the long-term NeuMF embedding.
- `UserFeedbackState` maintains a diagonal ridge LinUCB approximation, signed embedding residual, and artist/genre fatigue/boost counters.

## Findings

### High

1. **Long-term and short-term model checkpoints are not independently inspectable in this checkout.**
   - `data/artifacts/*.pt` and `track_embeddings.npy` are gitignored.
   - A source-only audit can verify architecture and loading code, but cannot verify that benchmark metrics came from the same trained state.

2. **Benchmark recommendation rank is sampled, not full-catalog.**
   - A target is ranked among 499 negatives, and only the first 600 test cases are scored.
   - This is useful for model smoke comparison but should not be presented as production full-catalog ranking accuracy.

3. **The feedback benchmark is simulation.**
   - It is based on random dominant/target genres and synthetic feedback events.
   - No live or held-out interaction data validates the 38.30% to 0.05% and 0.95% to 44.45% outcomes.

### Medium

4. **The diagonal LinUCB uses a 128D ridge update but the "posterior" claim is approximated, not a full covariance model.**
   - The implementation is directionally useful and fast, but it is not the standard full or diagonal context covariance fit with cross-dimension off-diagonal effects.

5. **Feedback residual and LinUCB use overlapping adaptation signals.**
   - Recommendation quality may be influenced twice: through `delta_u` and through bandit mean/uncertainly bonus plus fatigue/boost counters.
   - There is no held-out feedback validation showing each mechanism's marginal contribution.

6. **Shuffle metrics mix independent objectives.**
   - Artist-gap, acoustic flow, genre switching, diversity, discovery pacing, and NDCG are internally computed.
   - The adaptive result is not uniformly dominant; for example, the checked-in report has lower artist spacing than Spotify-style spacing and lower session NDCG than greedy relevance.

## Implemented Corrections

- Simplified the manual short-term horizon override so the returned long-term gate exactly equals `1 - short_term_weight`.
- Added a focused test for manual horizon override, ensuring the fused vector is the normalized convex combination and the gate reports the correct long-term weight.
- Relaxed the API lifecycle test from the default 40k/500-user dataset assumptions to positive catalog and user counts, allowing valid alternate builds.

## Remaining Validation

- Re-run model and shuffle unit tests once a Python 3.12 environment with project dependencies is available.
- Run the GPU training pipeline only if source datasets and adequate hardware/storage are available.
