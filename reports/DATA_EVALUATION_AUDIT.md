# Data and Evaluation Audit

## Verified Dataset Construction

The executable pipeline combines three public sources:

- A Spotify track catalog with 12 audio features.
- Spotify Million Playlist Dataset JSON slices.
- Last.fm 1K timestamped listening histories.

The checked-in report records 40,000 tracks, 26,113 exact-audio tracks, 8,146 artists, 114 primary genres, 943 users, 327,754 retained interactions, and 6,932 curated playlists. These counts are consistent with the report and source summary fields. They are not proof that a rerun will reproduce the report exactly.

The README previously presented 2.06M as retained interactions. A fresh build confirms 2,067,551 raw matched Last.fm events and 327,754 retained interactions. The README now reports both counts.

A fresh CPU-only build completed data construction in 53.26 seconds with the default 40,000-track limit and 10 MPD slices.

## Methodology Findings

### High

1. **Implicit-action labels are timestamp proxies, not observed listening behavior.**
   - Skip, partial, complete, repeat, and favorite labels are inferred from the interval to the next event, a 30-minute session boundary, playlist replay counts, and a synthetic 240-second fallback for the last event.
   - Repeated-listening and favorite labels combine replay counts with timing heuristics rather than explicit Spotify/Last.fm feedback fields.
   - These labels should be reported as inferred weak supervision.

2. **Catalog expansion uses stochastic prior-derived synthetic audio.**
   - Frequent MPD tracks missing exact audio receive genre, popularity, and audio priors from artist matches.
   - Tempo, loudness, and generic continuous features receive seeded Gaussian perturbation.
   - Such tracks are model inputs that can be mistaken for measured Spotify acoustics if not identified by `has_exact_audio=False`.

### Medium

3. **Chronological split implementation deserves stronger leakage tests.**
   - The split holds out the last two positive actions for each eligible user.
   - Transformer training includes action and time context only before the validation target.
   - However, the public definition and implementation should test exclusion of future rows, positive targets, target replay, and `seen_pos` behavior.

4. **Recommendation ranking uses sampled negatives.**
   - Each test case uses 499 random catalog negatives plus one target.
   - Sampled rankings can bias HR/NDCG estimates; they are not full-catalog ranking metrics.
   - Negative sampling is reproducible only through the fixed RNG and does not require full-catalog item exposure.

5. **Top-20 diversity/coverage are dependent on a fixed, truncated candidate history.**
   - Ranking metrics vary with the first 600 test cases and the fixed negative RNG.
   - Full-catalog coverage is more meaningful across repeat reruns than no-variance single-run values.

6. **Shuffle feedback benchmark is simulated, not observed.**
   - It chooses a dominant genre and target genre randomly for up to 100 users.
   - It reports after-intervention shares on the synthetic feedback state.
   - This should not be described as live user evidence.

## Reproducibility Risks

- Processed datasets and artifacts are gitignored, so only source and final reports remain in this checkout.
- The pipeline can download and build fresh datasets, but no lockfile pins the downloaded dataset revisions.
- `set_seed(42)` sets NumPy and torch seeds, but it does not explicitly configure CUDA deterministic algorithms or cudnn settings.
- The report does not include per-run standard deviations or repeated seeds for recommendation/shuffle metrics.
- The report has no dataset hashes or dataset snapshot identifiers.

## Recommended Corrections

1. Add dataset snapshot/revision identifiers and generated hashes to `dataset_stats.json`.
2. Report exact retained interaction count and clearly distinguish it from raw Last.fm event count.
3. Separate exact-audio and prior-synthetic tracks in evaluation outputs.
4. Add split and negative-sampling unit tests.
5. Report seed, mean, and variance across multiple reruns for benchmark metrics.
