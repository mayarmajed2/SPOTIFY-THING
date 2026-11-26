"""
Adaptive Shuffle Sequencer & Baseline Shuffle Algorithms.

Solves: "In what order should a selected set of songs be played?"

Implements 4 side-by-side algorithms for live comparison & quantitative benchmarking:
1. `fisher_yates`     : Standard uniform random permutation (suffers from artist/genre clumping)
2. `artist_spaced`    : Heuristic proportional artist-spacing (Spotify-inspired baseline, ignores genre & audio flow)
3. `greedy_relevance` : Deterministic rank order by user affinity (zero shuffle unpredictability)
4. `adaptive`         : Proposed Constrained Probabilistic Sequencer optimizing:
   - Personalized Relevance <u_t, e_i>
   - Smooth Acoustic Transition Flow (BPM harmonic ratio, energy, valence, Circle-of-Fifths key)
   - Max-Min Embedding Diversity over sliding window W
   - Hard + Exponentially Decaying Artist & Genre Cooldown Penalties
   - Familiarity/Discovery Pacing + Boltzmann Temperature Sampling (genuine unpredictability)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Circle of Fifths mapping for pitch classes 0..11 (C=0, C#=1, ..., B=11)
CIRCLE_OF_FIFTHS_POS = np.array([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5], dtype=np.int32)


@dataclass
class ShuffleConfig:
    mode: str = "adaptive"  # adaptive | fisher_yates | artist_spaced | greedy_relevance
    diversity_weight: float = 0.65
    flow_weight: float = 0.45
    relevance_weight: float = 0.55
    artist_cooldown: int = 4
    genre_cooldown: int = 2
    entropy_temperature: float = 0.18  # Controls unpredictability (0.05 = focused, 0.60 = wild)
    discovery_target: float = 0.35
    seed: int | None = None


def compute_acoustic_flow(prev_audio: np.ndarray, cand_audio: np.ndarray) -> np.ndarray:
    """
    Compute smooth musical transition score in [0, 1] between previous track [5] and candidates [M, 5].
    Expected raw columns: [danceability, energy, valence, tempo, key]
    """
    d_energy = np.abs(cand_audio[:, 1] - prev_audio[1])
    d_valence = np.abs(cand_audio[:, 2] - prev_audio[2])
    d_dance = np.abs(cand_audio[:, 0] - prev_audio[0])

    # Harmonic BPM distance: treat 2x / 0.5x tempo as rhythmically compatible
    prev_bpm = max(float(prev_audio[3]), 40.0)
    cand_bpm = np.maximum(cand_audio[:, 3], 40.0)
    bpm_diff = np.minimum.reduce(
        [
            np.abs(cand_bpm - prev_bpm),
            np.abs(cand_bpm * 2.0 - prev_bpm),
            np.abs(cand_bpm * 0.5 - prev_bpm),
        ]
    )
    bpm_score = np.exp(-bpm_diff / 28.0)

    # Circle-of-fifths harmonic compatibility
    prev_key = int(prev_audio[4]) % 12
    cand_keys = np.mod(cand_audio[:, 4].astype(np.int32), 12)
    wheel_dist = np.abs(CIRCLE_OF_FIFTHS_POS[cand_keys] - CIRCLE_OF_FIFTHS_POS[prev_key])
    wheel_dist = np.minimum(wheel_dist, 12 - wheel_dist)  # in [0..6]
    key_score = 1.0 - (wheel_dist / 6.0)

    mood_score = 1.0 - np.clip(0.45 * d_energy + 0.35 * d_valence + 0.20 * d_dance, 0.0, 1.0)
    return (0.50 * mood_score + 0.35 * bpm_score + 0.15 * key_score).astype(np.float32)


def evaluate_sequence_metrics(
    ordered_pos: list[int],
    embeddings: np.ndarray,
    artist_ids: np.ndarray,
    genre_ids: np.ndarray,
    raw_audio: np.ndarray,
    relevance_scores: np.ndarray,
    is_discovery: np.ndarray,
    window: int = 5,
) -> dict[str, float]:
    """Compute quantitative shuffle quality metrics on an ordered sequence of pool positions."""
    n = len(ordered_pos)
    if n <= 1:
        return {
            "intra_list_diversity": 0.0,
            "local_window_diversity": 0.0,
            "min_artist_gap": float(n),
            "mean_artist_gap": float(n),
            "artist_clump_rate": 0.0,
            "genre_switch_rate": 0.0,
            "acoustic_smoothness": 1.0,
            "mean_energy_jump": 0.0,
            "mean_bpm_jump": 0.0,
            "early_relevance_ndcg": float(relevance_scores[ordered_pos[0]]) if n == 1 else 0.0,
            "discovery_pacing_std": 0.0,
        }

    seq_emb = embeddings[ordered_pos]
    seq_art = artist_ids[ordered_pos]
    seq_gen = genre_ids[ordered_pos]
    seq_aud = raw_audio[ordered_pos]
    seq_rel = relevance_scores[ordered_pos]
    seq_disc = is_discovery[ordered_pos].astype(np.float32)

    # 1. Global & Local Intra-List Diversity (1 - cosine similarity)
    sim_mat = np.clip(seq_emb @ seq_emb.T, -1.0, 1.0)
    tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    ild = float(np.mean(1.0 - sim_mat[tri_mask]))

    local_divs = []
    for k in range(1, n):
        start = max(0, k - window)
        local_divs.append(float(np.mean(1.0 - sim_mat[k, start:k])))
    local_ild = float(np.mean(local_divs))

    # 2. Artist spacing & clump rate (same artist within <= 2 steps)
    last_seen: dict[int, int] = {}
    gaps: list[int] = []
    clumps = 0
    for pos_idx, a in enumerate(seq_art):
        a_int = int(a)
        if a_int in last_seen:
            gap = pos_idx - last_seen[a_int]
            gaps.append(gap)
            if gap <= 2:
                clumps += 1
        last_seen[a_int] = pos_idx

    min_artist_gap = float(min(gaps)) if gaps else float(n)
    mean_artist_gap = float(np.mean(gaps)) if gaps else float(n)
    artist_clump_rate = float(clumps / max(1, n - 1))

    # 3. Genre switch rate (consecutive tracks from different genres)
    genre_switches = int(np.sum(seq_gen[1:] != seq_gen[:-1]))
    genre_switch_rate = float(genre_switches / (n - 1))

    # 4. Acoustic transition smoothness & jumps
    flows = []
    energy_jumps = []
    bpm_jumps = []
    for k in range(1, n):
        flows.append(float(compute_acoustic_flow(seq_aud[k - 1], seq_aud[k : k + 1])[0]))
        energy_jumps.append(abs(float(seq_aud[k, 1] - seq_aud[k - 1, 1])))
        bpm_jumps.append(abs(float(seq_aud[k, 3] - seq_aud[k - 1, 3])))

    # 5. Discounted session relevance (front-loaded enjoyment without drop-off)
    discounts = 1.0 / np.log2(np.arange(2, n + 2, dtype=np.float32))
    ideal_rel = np.sort(seq_rel)[::-1]
    denom = float(np.sum(np.maximum(ideal_rel, 0.0) * discounts)) + 1e-8
    early_rel = float(np.sum(np.maximum(seq_rel, 0.0) * discounts) / denom)

    # 6. Discovery pacing uniformity (rolling window std; lower = smoother interleaving)
    if n >= 6:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        rolling_disc = np.convolve(seq_disc, kernel, mode="valid")
        disc_std = float(np.std(rolling_disc))
    else:
        disc_std = 0.0

    return {
        "intra_list_diversity": round(ild, 4),
        "local_window_diversity": round(local_ild, 4),
        "min_artist_gap": round(min_artist_gap, 2),
        "mean_artist_gap": round(mean_artist_gap, 2),
        "artist_clump_rate": round(artist_clump_rate, 4),
        "genre_switch_rate": round(genre_switch_rate, 4),
        "acoustic_smoothness": round(float(np.mean(flows)), 4),
        "mean_energy_jump": round(float(np.mean(energy_jumps)), 4),
        "mean_bpm_jump": round(float(np.mean(bpm_jumps)), 2),
        "session_ndcg": round(early_rel, 4),
        "discovery_pacing_std": round(disc_std, 4),
    }


class AdaptiveShuffleEngine:
    """Sequences a pool of candidate/playlist tracks using Adaptive Shuffle or baselines."""

    def sequence(
        self,
        embeddings: np.ndarray,
        artist_ids: np.ndarray,
        genre_ids: np.ndarray,
        raw_audio: np.ndarray,
        relevance_scores: np.ndarray,
        is_discovery: np.ndarray,
        config: ShuffleConfig,
        artist_fatigue: dict[int, float] | None = None,
        genre_fatigue: dict[int, float] | None = None,
    ) -> dict[str, Any]:
        """
        Inputs are aligned arrays of length M (pool size):
        - embeddings      : [M, D] L2-normalized track vectors
        - artist_ids      : [M] integer artist IDs
        - genre_ids       : [M] integer genre IDs
        - raw_audio       : [M, 5] [danceability, energy, valence, tempo, key]
        - relevance_scores: [M] user affinity scores normalized to [0, 1]
        - is_discovery    : [M] bool array (True if unseen track for user)
        """
        m = len(embeddings)
        if m == 0:
            return {"ordered_positions": [], "step_details": [], "metrics": {}}

        rng = np.random.default_rng(config.seed)
        mode = config.mode.lower().strip()

        if mode == "fisher_yates":
            order = rng.permutation(m).tolist()
            details = [{"step": idx, "reason": "Uniform random draw (Fisher-Yates)"} for idx in range(m)]
        elif mode == "greedy_relevance":
            order = np.argsort(-relevance_scores).tolist()
            details = [{"step": idx, "reason": "Sorted strictly by predicted relevance"} for idx in range(m)]
        elif mode == "artist_spaced":
            order = self._artist_spaced_baseline(artist_ids, rng)
            details = [{"step": idx, "reason": "Proportional artist-interval spacing heuristic"} for idx in range(m)]
        else:
            order, details = self._adaptive_probabilistic_sequence(
                embeddings=embeddings,
                artist_ids=artist_ids,
                genre_ids=genre_ids,
                raw_audio=raw_audio,
                relevance_scores=relevance_scores,
                is_discovery=is_discovery,
                config=config,
                rng=rng,
                artist_fatigue=artist_fatigue or {},
                genre_fatigue=genre_fatigue or {},
            )

        metrics = evaluate_sequence_metrics(
            ordered_pos=order,
            embeddings=embeddings,
            artist_ids=artist_ids,
            genre_ids=genre_ids,
            raw_audio=raw_audio,
            relevance_scores=relevance_scores,
            is_discovery=is_discovery,
        )
        return {
            "mode": mode,
            "ordered_positions": order,
            "step_details": details,
            "metrics": metrics,
        }

    def _artist_spaced_baseline(self, artist_ids: np.ndarray, rng: np.random.Generator) -> list[int]:
        """Spotify-style 2014 proportional artist spacing heuristic with random offset."""
        buckets: dict[int, list[int]] = {}
        for idx, a in enumerate(artist_ids):
            buckets.setdefault(int(a), []).append(idx)

        positioned: list[tuple[float, int]] = []
        for tracks in buckets.values():
            rng.shuffle(tracks)
            k = len(tracks)
            step = 1.0 / k
            offset = float(rng.uniform(0.0, step))
            for i, pos_idx in enumerate(tracks):
                jitter = float(rng.normal(0.0, 0.03 * step))
                key_pos = offset + i * step + jitter
                positioned.append((key_pos, pos_idx))

        positioned.sort(key=lambda t: t[0])
        return [p[1] for p in positioned]

    def _adaptive_probabilistic_sequence(
        self,
        embeddings: np.ndarray,
        artist_ids: np.ndarray,
        genre_ids: np.ndarray,
        raw_audio: np.ndarray,
        relevance_scores: np.ndarray,
        is_discovery: np.ndarray,
        config: ShuffleConfig,
        rng: np.random.Generator,
        artist_fatigue: dict[int, float],
        genre_fatigue: dict[int, float],
    ) -> tuple[list[int], list[dict[str, Any]]]:
        m = len(embeddings)
        remaining = np.ones(m, dtype=bool)
        order: list[int] = []
        details: list[dict[str, Any]] = []

        # Normalize relevance to [0, 1] inside pool
        r_min, r_max = float(relevance_scores.min()), float(relevance_scores.max())
        norm_rel = (relevance_scores - r_min) / (r_max - r_min + 1e-6)

        # Static user-session fatigue penalties
        static_fatigue = np.array(
            [
                0.6 * artist_fatigue.get(int(artist_ids[i]), 0.0)
                + 0.4 * genre_fatigue.get(int(genre_ids[i]), 0.0)
                for i in range(m)
            ],
            dtype=np.float32,
        )

        sim_matrix = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
        disc_count_so_far = 0

        for step in range(m):
            cand_indices = np.flatnonzero(remaining)
            if len(cand_indices) == 1:
                chosen = int(cand_indices[0])
                order.append(chosen)
                details.append(
                    {
                        "step": step,
                        "relevance": round(float(norm_rel[chosen]), 3),
                        "flow": 1.0,
                        "diversity": 1.0,
                        "penalty": 0.0,
                        "reason": "Final track in queue",
                    }
                )
                break

            rel_term = norm_rel[cand_indices] - static_fatigue[cand_indices]

            if step == 0:
                # Anchor start of session on an engaging, familiar or high-relevance track
                start_scores = config.relevance_weight * rel_term + 0.25 * (~is_discovery[cand_indices]).astype(np.float32)
                chosen_local = self._boltzmann_sample(start_scores, max(config.entropy_temperature, 0.12), rng)
                chosen = int(cand_indices[chosen_local])
                remaining[chosen] = False
                order.append(chosen)
                if is_discovery[chosen]:
                    disc_count_so_far += 1
                details.append(
                    {
                        "step": 0,
                        "relevance": round(float(norm_rel[chosen]), 3),
                        "flow": 1.0,
                        "diversity": 1.0,
                        "penalty": 0.0,
                        "reason": "Strong session opener aligned with taste",
                    }
                )
                continue

            prev = order[-1]
            window_hist = order[max(0, step - 6) : step]

            # 1. Acoustic transition flow from previous track
            flow_term = compute_acoustic_flow(raw_audio[prev], raw_audio[cand_indices])

            # 2. Max-Min Diversity (distance to closest track in recent window)
            recent_sims = sim_matrix[np.ix_(cand_indices, window_hist)]
            max_recent_sim = recent_sims.max(axis=1)
            div_term = 1.0 - max_recent_sim  # higher when distinct from all recent W tracks

            # 3. Exponentially decaying Artist & Genre Cooldown Penalty + Hard Cooldown Guard
            rep_penalty = np.zeros(len(cand_indices), dtype=np.float32)
            hard_artist_block = np.zeros(len(cand_indices), dtype=bool)
            cand_art = artist_ids[cand_indices]
            cand_gen = genre_ids[cand_indices]

            lookback = min(step, max(config.artist_cooldown + 4, 8))
            for dist in range(1, lookback + 1):
                hist_pos = order[step - dist]
                same_art = cand_art == artist_ids[hist_pos]
                same_gen = cand_gen == genre_ids[hist_pos]

                if dist <= config.artist_cooldown:
                    hard_artist_block |= same_art
                rep_penalty += same_art.astype(np.float32) * (1.65 * (0.62 ** (dist - 1)))

                if dist <= config.genre_cooldown:
                    rep_penalty += same_gen.astype(np.float32) * (0.75 * (0.55 ** (dist - 1)))
                else:
                    rep_penalty += same_gen.astype(np.float32) * (0.20 * (0.55 ** (dist - 1)))

            # Only enforce hard artist block if at least one candidate passes it
            if not hard_artist_block.all():
                rep_penalty = np.where(hard_artist_block, rep_penalty + 25.0, rep_penalty)

            # 4. Smooth Discovery vs. Familiarity Pacing
            current_disc_ratio = disc_count_so_far / step
            need_discovery = config.discovery_target - current_disc_ratio
            pacing_bonus = np.where(
                is_discovery[cand_indices],
                np.clip(need_discovery * 1.2, -0.35, 0.45),
                np.clip(-need_discovery * 1.2, -0.35, 0.45),
            ).astype(np.float32)

            total_utility = (
                config.relevance_weight * rel_term
                + config.flow_weight * flow_term
                + config.diversity_weight * div_term
                + pacing_bonus
                - config.diversity_weight * rep_penalty
            )
            if not hard_artist_block.all():
                total_utility = np.where(hard_artist_block, -1e6, total_utility)

            chosen_local = self._boltzmann_sample(total_utility, config.entropy_temperature, rng)
            chosen = int(cand_indices[chosen_local])
            remaining[chosen] = False
            order.append(chosen)
            if is_discovery[chosen]:
                disc_count_so_far += 1

            reasons = []
            if flow_term[chosen_local] >= 0.72:
                reasons.append("Smooth BPM/mood transition")
            if div_term[chosen_local] >= 0.45:
                reasons.append("Fresh sonic palette")
            if is_discovery[chosen]:
                reasons.append("Discovery pacing")
            elif norm_rel[chosen] >= 0.70:
                reasons.append("High taste affinity")
            if not reasons:
                reasons.append("Balanced adaptive transition")

            details.append(
                {
                    "step": step,
                    "relevance": round(float(norm_rel[chosen]), 3),
                    "flow": round(float(flow_term[chosen_local]), 3),
                    "diversity": round(float(div_term[chosen_local]), 3),
                    "penalty": round(float(rep_penalty[chosen_local]), 3),
                    "reason": " · ".join(reasons[:2]),
                }
            )

        return order, details

    @staticmethod
    def _boltzmann_sample(utilities: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
        temp = max(float(temperature), 0.02)
        # Restrict sampling to top-15 candidates to prevent extreme tail misfires at high entropy
        if len(utilities) > 15:
            top_idx = np.argpartition(utilities, -15)[-15:]
            sub_u = utilities[top_idx]
            scaled = (sub_u - sub_u.max()) / temp
            probs = np.exp(np.clip(scaled, -30.0, 0.0))
            probs /= probs.sum()
            return int(top_idx[rng.choice(len(top_idx), p=probs)])

        scaled = (utilities - utilities.max()) / temp
        probs = np.exp(np.clip(scaled, -30.0, 0.0))
        probs /= probs.sum()
        return int(rng.choice(len(utilities), p=probs))
