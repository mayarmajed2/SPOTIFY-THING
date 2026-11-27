"""
Hybrid Music Recommendation Engine.

Unifies:
- Shared Multimodal Track Embeddings + exact inner-product retrieval
- Collaborative Filtering (Item-KNN Co-occurrence + NeuMF Long-Term Profile)
- Dual-Horizon Session Transformer (Short-Term + Long-Term Gated Context)
- Familiarity vs. Discovery Pool Controller + Self-Information Novelty
- Real-time Online Contextual Bandit Feedback Adaptation + Human-Readable Explanations
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def reciprocal_rank_score(scores: np.ndarray, k_rrf: int = 60, positive_only: bool = False) -> np.ndarray:
    """
    Convert arbitrary score vector into bounded, outlier-proof Reciprocal Rank scores in [0, 1].
    Prevents sparse CF spikes from hijacking full-catalog rankings when fused with dense embeddings.
    """
    n = len(scores)
    out = np.zeros(n, dtype=np.float32)
    if n == 0:
        return out
    if positive_only:
        valid = np.flatnonzero(scores > 0.0)
        if len(valid) == 0:
            return out
        order = valid[np.argsort(-scores[valid])]
    else:
        order = np.argsort(-scores)
    ranks = np.arange(1, len(order) + 1, dtype=np.float32)
    out[order] = (k_rrf + 1.0) / (k_rrf + ranks)
    return out


class ExactInnerProductIndex:
    """SIMD/BLAS-accelerated exact Inner Product vector index (drop-in for IndexFlatIP)."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.atleast_2d(queries).astype(np.float32)
        sims = q @ self.matrix.T
        k = min(k, sims.shape[1])
        top_part = np.argpartition(sims, -k, axis=1)[:, -k:]
        row_idx = np.arange(len(q))[:, None]
        top_sims = sims[row_idx, top_part]
        order = np.argsort(-top_sims, axis=1)
        return top_sims[row_idx, order], top_part[row_idx, order]

from backend.app.engine.adaptive_shuffle import AdaptiveShuffleEngine, ShuffleConfig
from backend.app.engine.feedback_store import SQLiteFeedbackStore
from backend.app.engine.online_feedback import FeedbackEvent, UserFeedbackState
from backend.app.models.collaborative import ItemCooccurrenceCF, NeuralCollaborativeFiltering
from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import (
    PAD_ACTION_ID,
    PAD_TIME_BUCKET,
    DualHorizonSessionTransformer,
    bucketize_time_deltas,
)


class HybridRecommendationEngine:
    def __init__(
        self,
        processed_dir: Path,
        artifacts_dir: Path,
        device: str = "cpu",
        feedback_db_path: Path | None = None,
    ) -> None:
        self.processed_dir = processed_dir
        self.artifacts_dir = artifacts_dir
        self.device = torch.device(device)
        self.feedback_store = SQLiteFeedbackStore(
            feedback_db_path if feedback_db_path is not None else (artifacts_dir / "feedback_state.db")
        )

        # Load catalog & metadata
        self.tracks_df = pd.read_parquet(processed_dir / "tracks.parquet")
        self.playlists_df = pd.read_parquet(processed_dir / "playlists.parquet")
        self.interactions_df = pd.read_parquet(processed_dir / "interactions.parquet")
        self.stats = json.loads((processed_dir / "dataset_stats.json").read_text(encoding="utf-8"))

        self.num_items = len(self.tracks_df)
        self.num_artists = int(self.tracks_df["artist_idx"].max()) + 1
        self.num_genres = int(self.tracks_df["genre_idx"].max()) + 1
        self.num_users = int(self.interactions_df["user_idx"].max()) + 1

        # Pre-extract fast numpy arrays aligned by item_idx (0..num_items-1)
        self.tracks_df = self.tracks_df.sort_values("item_idx").reset_index(drop=True)
        self.artist_ids = self.tracks_df["artist_idx"].to_numpy(dtype=np.int32)
        self.genre_ids = self.tracks_df["genre_idx"].to_numpy(dtype=np.int32)
        self.genre_names = self.tracks_df["track_genre"].tolist()
        self.artist_names = self.tracks_df["primary_artist"].tolist()
        self.track_names = self.tracks_df["track_name"].tolist()
        self.raw_audio_5d = self.tracks_df[["danceability", "energy", "valence", "tempo", "key"]].to_numpy(
            dtype=np.float32
        )

        # Self-information novelty in [0, 1]: -log2(pop_prob) normalized
        pop_raw = self.tracks_df["popularity"].to_numpy(dtype=np.float32) / 100.0
        lfm_freq = np.log1p(self.tracks_df["lfm_count"].to_numpy(dtype=np.float32))
        lfm_freq /= max(float(lfm_freq.max()), 1.0)
        exposure_prob = np.clip(0.5 * pop_raw + 0.5 * lfm_freq, 0.01, 0.99)
        self_info = -np.log2(exposure_prob)
        self.item_novelty = ((self_info - self_info.min()) / (self_info.max() - self_info.min() + 1e-6)).astype(
            np.float32
        )

        # Build per-user historical trajectories for instant lookup
        self.user_histories: dict[int, dict[str, Any]] = {}
        for uid, grp in self.interactions_df.groupby("user_idx", sort=False):
            items = grp["item_idx"].to_numpy(dtype=np.int32)
            actions = grp["action_id"].to_numpy(dtype=np.int32)
            ts = grp["unix_ts"].to_numpy(dtype=np.int64)
            pos_set = set(items[actions >= 2].tolist())
            self.user_histories[int(uid)] = {
                "user_id": str(grp["user_id"].iloc[0]),
                "country": str(grp["country"].iloc[0]),
                "items": items,
                "actions": actions,
                "unix_ts": ts,
                "known_items": set(items.tolist()),
                "positive_items": pos_set,
            }

        # Load trained embeddings, exact vector index, CF, NeuMF, and Dual-Horizon Transformer
        self._load_models()
        self.shuffle_engine = AdaptiveShuffleEngine()
        self.feedback_states: dict[int, UserFeedbackState] = {}

        # Fast O(1) lookup tables and prototype embeddings for Companion Extension APIs
        self._build_companion_indexes()

    def _load_models(self) -> None:
        required_paths = [
            self.artifacts_dir / "neumf_checkpoint.pt",
            self.artifacts_dir / "transformer_checkpoint.pt",
            self.artifacts_dir / "track_embeddings.npy",
            self.artifacts_dir / "item_cf.pkl",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                "Generated model artifacts are missing: "
                + ", ".join(missing)
                + ". Run backend/training/train_and_evaluate.py after building the processed dataset."
            )

        norm_cols = self.stats["norm_audio_cols"]
        audio_tensor = torch.tensor(self.tracks_df[norm_cols].to_numpy(dtype=np.float32))
        artist_tensor = torch.tensor(self.artist_ids, dtype=torch.long)
        genre_tensor = torch.tensor(self.genre_ids, dtype=torch.long)

        self.track_encoder = MultimodalTrackEncoder(
            num_items=self.num_items,
            num_artists=self.num_artists,
            num_genres=self.num_genres,
            audio_dim=len(norm_cols),
            embed_dim=128,
        )
        self.track_encoder.set_catalog_metadata(artist_tensor, genre_tensor, audio_tensor)

        self.neumf = NeuralCollaborativeFiltering(
            num_users=self.num_users,
            track_encoder=self.track_encoder,
            embed_dim=128,
        )
        self.transformer = DualHorizonSessionTransformer(
            track_encoder=self.track_encoder,
            max_seq_len=50,
            embed_dim=128,
        )

        neumf_ckpt = self.artifacts_dir / "neumf_checkpoint.pt"
        if neumf_ckpt.exists():
            self.neumf.load_state_dict(torch.load(neumf_ckpt, map_location="cpu", weights_only=True))

        tr_ckpt = self.artifacts_dir / "transformer_checkpoint.pt"
        if tr_ckpt.exists():
            self.transformer.load_state_dict(torch.load(tr_ckpt, map_location="cpu", weights_only=True))

        self.track_encoder.to(self.device).eval()
        self.neumf.to(self.device).eval()
        self.transformer.to(self.device).eval()

        emb_path = self.artifacts_dir / "track_embeddings.npy"
        if emb_path.exists():
            self.item_embeddings = np.load(emb_path).astype(np.float32)
        else:
            self.item_embeddings = self.track_encoder.export_all_embeddings(device=self.device).numpy()

        # Build BLAS-accelerated Exact Inner Product index (vectors are L2-normalized -> cosine similarity)
        self.vector_index = ExactInnerProductIndex(self.item_embeddings)

        cf_path = self.artifacts_dir / "item_cf.pkl"
        if cf_path.exists():
            with open(cf_path, "rb") as f:
                self.item_cf: ItemCooccurrenceCF = pickle.load(f)
        else:
            self.item_cf = ItemCooccurrenceCF(self.num_items).fit(
                (
                    self.interactions_df["user_idx"].to_numpy(dtype=np.int32),
                    self.interactions_df["item_idx"].to_numpy(dtype=np.int32),
                    np.maximum(self.interactions_df["reward"].to_numpy(dtype=np.float32), 0.05),
                )
            )

    def get_feedback_state(self, user_idx: int) -> UserFeedbackState:
        if user_idx not in self.feedback_states:
            self.feedback_states[user_idx] = self.feedback_store.load_user_state(
                user_idx=user_idx, embed_dim=128
            )
        return self.feedback_states[user_idx]

    def record_user_feedback(
        self,
        user_idx: int,
        item_idx: int,
        action: str,
        played_ratio: float = 1.0,
    ) -> tuple[FeedbackEvent, UserFeedbackState]:
        fb_state = self.get_feedback_state(user_idx)
        ev = fb_state.record_feedback(
            item_idx=item_idx,
            item_emb=self.item_embeddings[item_idx],
            artist_idx=int(self.artist_ids[item_idx]),
            genre_idx=int(self.genre_ids[item_idx]),
            action=action,
            played_ratio=played_ratio,
        )
        self.feedback_store.save_user_state(fb_state)
        return ev, fb_state

    def reset_feedback_state(self, user_idx: int) -> None:
        self.feedback_store.reset_user_state(user_idx)
        self.feedback_states[user_idx] = UserFeedbackState(user_idx=user_idx, embed_dim=128)

    @torch.no_grad()
    def compute_user_representation(
        self,
        user_idx: int,
        short_term_override: float | None = None,
    ) -> dict[str, Any]:
        """Compute Long-Term, Short-Term, Gate Weight, and Online-Feedback-Adjusted User Vectors."""
        hist = self.user_histories[user_idx]
        fb_state = self.get_feedback_state(user_idx)

        # Combine historical sequence with live feedback events
        items_list = hist["items"][-50:].tolist()
        actions_list = hist["actions"][-50:].tolist()
        ts_list = hist["unix_ts"][-50:].tolist()

        for ev in fb_state.events[-30:]:
            items_list.append(ev.item_idx)
            actions_list.append(ev.action_id)
            ts_list.append(int(ev.timestamp))

        L = self.transformer.max_seq_len
        items_tail = items_list[-L:]
        actions_tail = actions_list[-L:]
        ts_tail = ts_list[-L:]

        deltas = [3600] + [max(0, int(ts_tail[k] - ts_tail[k - 1])) for k in range(1, len(ts_tail))]
        delta_buckets = bucketize_time_deltas(torch.tensor(deltas, dtype=torch.long)).tolist()

        pad_len = L - len(items_tail)
        item_seq = torch.tensor(
            [[self.num_items] * pad_len + items_tail], dtype=torch.long, device=self.device
        )
        act_seq = torch.tensor(
            [[PAD_ACTION_ID] * pad_len + actions_tail], dtype=torch.long, device=self.device
        )
        tb_seq = torch.tensor(
            [[PAD_TIME_BUCKET] * pad_len + delta_buckets], dtype=torch.long, device=self.device
        )
        u_id_t = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        u_long = self.neumf.get_long_term_user_emb(u_id_t)  # [1, 128]
        u_short_seq = self.transformer.encode_short_term_sequence(item_seq, act_seq, tb_seq)
        u_short = u_short_seq[:, -1, :]  # [1, 128]

        u_fused_t, gate_t = self.transformer.fuse_horizons(
            u_long, u_short, short_term_override=short_term_override
        )
        u_fused = u_fused_t.squeeze(0).cpu().numpy()

        # Apply real-time online feedback residual shift
        if np.linalg.norm(fb_state.delta_u) > 1e-6:
            u_final = u_fused + fb_state.delta_u
            u_final = u_final / (np.linalg.norm(u_final) + 1e-8)
        else:
            u_final = u_fused

        return {
            "u_long": u_long.squeeze(0).cpu().numpy(),
            "u_short": u_short.squeeze(0).cpu().numpy(),
            "u_fused": u_final.astype(np.float32),
            "long_term_gate": float(gate_t.mean().item()),
            "recent_items": items_tail,
            "recent_actions": actions_tail,
        }

    def recommend(
        self,
        user_idx: int,
        top_k: int = 24,
        discovery_level: float = 0.40,
        diversity_level: float = 0.50,
        novelty_boost: float = 0.30,
        short_term_weight: float | None = None,
    ) -> dict[str, Any]:
        """Generate personalized recommendations balancing familiarity, discovery, diversity & feedback."""
        rep = self.compute_user_representation(user_idx, short_term_override=short_term_weight)
        u_vec = rep["u_fused"]
        hist = self.user_histories[user_idx]
        fb_state = self.get_feedback_state(user_idx)

        known_items = hist["known_items"] | fb_state.liked_items | set(fb_state.played_items_recent)
        blocked_items = fb_state.disliked_items | set(fb_state.played_items_recent[-12:])

        # 1. Shared Embedding Cosine Similarity over the full catalog
        cos_scores = (self.item_embeddings @ u_vec).astype(np.float32)

        # 2. Item-KNN Collaborative Filtering Score from positive history
        pos_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (2, 3, 4)
        ] + list(fb_state.liked_items)
        neg_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (0, 5)
        ] + list(fb_state.disliked_items)
        cf_raw = self.item_cf.score_for_history(pos_seed, neg_seed)
        cf_rrf = reciprocal_rank_score(cf_raw, k_rrf=60, positive_only=True)
        cos_rrf = reciprocal_rank_score(cos_scores, k_rrf=60, positive_only=False)

        # 3. Contextual Bandit Exploration + Exploitation Bonus
        bandit_scores = fb_state.linucb_bonus(
            self.item_embeddings, alpha_explore=0.10 + 0.18 * discovery_level
        )

        # 4. Dynamic Artist/Genre Boosts & Fatigue Penalties from real-time feedback
        dyn_adj = np.zeros(self.num_items, dtype=np.float32)
        for a_idx, pen in fb_state.artist_fatigue.items():
            dyn_adj[self.artist_ids == a_idx] -= 0.45 * pen
        for g_idx, pen in fb_state.genre_fatigue.items():
            dyn_adj[self.genre_ids == g_idx] -= 0.30 * pen
        for a_idx, bst in fb_state.artist_boost.items():
            dyn_adj[self.artist_ids == a_idx] += 0.20 * bst
        for g_idx, bst in fb_state.genre_boost.items():
            dyn_adj[self.genre_ids == g_idx] += 0.12 * bst

        # Scale-calibrated Reciprocal Rank Fusion (RRF) between Dual-Horizon Transformer and sparse CF
        base_relevance = (
            0.55 * cos_scores
            + 0.22 * (0.78 * cos_rrf + 0.22 * cf_rrf)
            + 0.20 * bandit_scores
            + novelty_boost * 0.22 * self.item_novelty
            + dyn_adj
        )
        cf_norm = cf_rrf

        for b_it in blocked_items:
            if 0 <= b_it < self.num_items:
                base_relevance[b_it] = -999.0

        is_known_mask = np.zeros(self.num_items, dtype=bool)
        if known_items:
            is_known_mask[list(known_items)] = True

        # Split target quota between Familiarity Pool and Discovery Pool
        n_disc_target = int(round(top_k * float(np.clip(discovery_level, 0.0, 1.0))))
        n_fam_target = top_k - n_disc_target

        fam_candidates = np.flatnonzero(is_known_mask & (base_relevance > -100))
        disc_candidates = np.flatnonzero((~is_known_mask) & (base_relevance > -100))

        # Top candidate pool for MMR diversification
        fam_top = fam_candidates[np.argsort(-base_relevance[fam_candidates])[: max(120, n_fam_target * 5)]]
        disc_top = disc_candidates[np.argsort(-base_relevance[disc_candidates])[: max(250, n_disc_target * 6)]]

        selected_fam = self._mmr_select(fam_top, base_relevance, min(n_fam_target, len(fam_top)), diversity_level)
        rem_slots = top_k - len(selected_fam)
        selected_disc = self._mmr_select(
            disc_top,
            base_relevance,
            rem_slots,
            diversity_level,
            already_selected=selected_fam,
        )

        # Interleave familiar & discovery tracks by combined MMR-adjusted score
        merged = selected_fam + selected_disc
        merged.sort(key=lambda idx: float(base_relevance[idx]), reverse=True)

        tracks_out = [
            self._format_track_recommendation(
                item_idx=it,
                score=float(base_relevance[it]),
                cos_sim=float(cos_scores[it]),
                cf_score=float(cf_norm[it]),
                is_discovery=not bool(is_known_mask[it]),
                rep=rep,
                fb_state=fb_state,
            )
            for it in merged
        ]

        return {
            "user_idx": user_idx,
            "user_id": hist["user_id"],
            "long_term_gate": round(rep["long_term_gate"], 3),
            "short_term_gate": round(1.0 - rep["long_term_gate"], 3),
            "discovery_ratio_actual": round(
                sum(1 for t in tracks_out if t["is_discovery"]) / max(1, len(tracks_out)), 3
            ),
            "feedback_summary": fb_state.summary(),
            "recommendations": tracks_out,
        }

    def _mmr_select(
        self,
        pool: np.ndarray,
        scores: np.ndarray,
        k: int,
        diversity_level: float,
        already_selected: list[int] | None = None,
    ) -> list[int]:
        """Maximal Marginal Relevance (MMR) with artist/genre saturation awareness."""
        if k <= 0 or len(pool) == 0:
            return []
        lam = 1.0 - 0.65 * float(np.clip(diversity_level, 0.0, 1.0))
        s_vals = scores[pool]
        s_norm = (s_vals - s_vals.min()) / (s_vals.max() - s_vals.min() + 1e-6)

        chosen: list[int] = []
        prior = list(already_selected or [])
        artist_counts: dict[int, int] = {}
        for p in prior:
            a = int(self.artist_ids[p])
            artist_counts[a] = artist_counts.get(a, 0) + 1

        avail = np.ones(len(pool), dtype=bool)
        pool_embs = self.item_embeddings[pool]

        for _ in range(min(k, len(pool))):
            cand_pos = np.flatnonzero(avail)
            ref_items = prior + chosen
            if not ref_items:
                mmr = s_norm[cand_pos]
            else:
                ref_embs = self.item_embeddings[ref_items]
                max_sim = (pool_embs[cand_pos] @ ref_embs.T).max(axis=1)
                art_pen = np.array(
                    [0.25 * artist_counts.get(int(self.artist_ids[pool[c]]), 0) for c in cand_pos],
                    dtype=np.float32,
                )
                mmr = lam * s_norm[cand_pos] - (1.0 - lam) * max_sim - art_pen

            best_local = int(cand_pos[int(np.argmax(mmr))])
            avail[best_local] = False
            picked_item = int(pool[best_local])
            chosen.append(picked_item)
            a_id = int(self.artist_ids[picked_item])
            artist_counts[a_id] = artist_counts.get(a_id, 0) + 1

        return chosen

    def _format_track_recommendation(
        self,
        item_idx: int,
        score: float,
        cos_sim: float,
        cf_score: float,
        is_discovery: bool,
        rep: dict[str, Any],
        fb_state: UserFeedbackState,
    ) -> dict[str, Any]:
        row = self.tracks_df.iloc[item_idx]
        reasons: list[str] = []

        # Compute short-term vs long-term alignment explanation
        e_i = self.item_embeddings[item_idx]
        short_sim = float(np.dot(rep["u_short"], e_i))
        long_sim = float(np.dot(rep["u_long"], e_i))

        if is_discovery:
            if float(self.item_novelty[item_idx]) >= 0.55:
                reasons.append("Hidden-gem discovery outside mainstream")
            else:
                reasons.append("New artist/track aligned with your sonic profile")
        else:
            reasons.append("Trusted favorite from your listening history")

        if short_sim > long_sim + 0.06:
            reasons.append("Matches your immediate session mood")
        elif long_sim >= 0.35:
            reasons.append(f"Fits your long-term {row['track_genre']} taste")

        if int(row["artist_idx"]) in fb_state.artist_boost or int(row["genre_idx"]) in fb_state.genre_boost:
            reasons.append("Boosted by your recent positive feedback")
        elif cf_score >= 0.35:
            reasons.append("Strong co-listening match across similar playlists")

        return {
            "item_idx": int(item_idx),
            "spotify_track_id": str(row["spotify_track_id"]),
            "track_name": str(row["track_name"]),
            "artist_name": str(row["primary_artist"]),
            "album_name": str(row["album_name"]),
            "genre": str(row["track_genre"]),
            "popularity": int(row["popularity"]),
            "duration_ms": int(row["duration_ms"]),
            "is_discovery": is_discovery,
            "novelty": round(float(self.item_novelty[item_idx]), 3),
            "score": round(score, 4),
            "short_term_affinity": round(short_sim, 3),
            "long_term_affinity": round(long_sim, 3),
            "audio": {
                "danceability": round(float(row["danceability"]), 3),
                "energy": round(float(row["energy"]), 3),
                "valence": round(float(row["valence"]), 3),
                "tempo": round(float(row["tempo"]), 1),
                "acousticness": round(float(row["acousticness"]), 3),
                "instrumentalness": round(float(row["instrumentalness"]), 3),
            },
            "explanations": reasons[:3],
        }

    def shuffle_pool(
        self,
        user_idx: int,
        item_indices: list[int],
        config: ShuffleConfig,
    ) -> dict[str, Any]:
        """Run Adaptive Shuffle (or any baseline) on a playlist or recommendation pool."""
        rep = self.compute_user_representation(user_idx)
        u_vec = rep["u_fused"]
        fb_state = self.get_feedback_state(user_idx)
        known = self.user_histories[user_idx]["known_items"] | fb_state.liked_items

        valid_items = [int(i) for i in item_indices if 0 <= int(i) < self.num_items]
        embs = self.item_embeddings[valid_items]
        arts = self.artist_ids[valid_items]
        gens = self.genre_ids[valid_items]
        aud = self.raw_audio_5d[valid_items]
        rel = (embs @ u_vec).astype(np.float32)
        is_disc = np.array([it not in known for it in valid_items], dtype=bool)

        res = self.shuffle_engine.sequence(
            embeddings=embs,
            artist_ids=arts,
            genre_ids=gens,
            raw_audio=aud,
            relevance_scores=rel,
            is_discovery=is_disc,
            config=config,
            artist_fatigue=fb_state.artist_fatigue,
            genre_fatigue=fb_state.genre_fatigue,
        )

        ordered_tracks = []
        for step_idx, pool_pos in enumerate(res["ordered_positions"]):
            it = valid_items[pool_pos]
            row = self.tracks_df.iloc[it]
            detail = res["step_details"][step_idx]
            ordered_tracks.append(
                {
                    "position": step_idx + 1,
                    "item_idx": it,
                    "spotify_track_id": str(row["spotify_track_id"]),
                    "track_name": str(row["track_name"]),
                    "artist_name": str(row["primary_artist"]),
                    "album_name": str(row["album_name"]),
                    "genre": str(row["track_genre"]),
                    "duration_ms": int(row["duration_ms"]),
                    "is_discovery": bool(is_disc[pool_pos]),
                    "audio": {
                        "danceability": round(float(row["danceability"]), 3),
                        "energy": round(float(row["energy"]), 3),
                        "valence": round(float(row["valence"]), 3),
                        "tempo": round(float(row["tempo"]), 1),
                    },
                    "step_diagnostics": detail,
                }
            )

        return {
            "mode": res["mode"],
            "metrics": res["metrics"],
            "tracks": ordered_tracks,
        }

    # =========================================================================
    # Companion Extension Methods (Discovery, Similarity, Fallback Shuffle, Insights)
    # =========================================================================

    def _build_companion_indexes(self) -> None:
        self.spotify_id_to_idx: dict[str, int] = {}
        self.norm_title_artist_to_idx: dict[tuple[str, str], int] = {}
        self.artist_name_to_idx: dict[str, int] = {}
        self.genre_name_to_idx: dict[str, int] = {}
        self.idx_to_genre_name: dict[int, str] = {}
        self.idx_to_artist_name: dict[int, str] = {}
        self.artist_top_track: dict[int, int] = {}
        self.artist_track_counts: dict[int, int] = {}
        self.genre_track_counts: dict[int, int] = {}

        pops = self.tracks_df["popularity"].to_numpy(dtype=np.int32)
        sp_ids = self.tracks_df["spotify_track_id"].astype(str).tolist()

        for it in range(self.num_items):
            sid = sp_ids[it]
            if sid and sid not in self.spotify_id_to_idx:
                self.spotify_id_to_idx[sid] = it
            key_ta = (self.track_names[it].lower().strip(), self.artist_names[it].lower().strip())
            if key_ta not in self.norm_title_artist_to_idx:
                self.norm_title_artist_to_idx[key_ta] = it

            a_idx = int(self.artist_ids[it])
            g_idx = int(self.genre_ids[it])
            a_name = self.artist_names[it]
            g_name = self.genre_names[it]

            self.artist_name_to_idx.setdefault(a_name.lower().strip(), a_idx)
            self.genre_name_to_idx.setdefault(g_name.lower().strip(), g_idx)
            self.idx_to_artist_name.setdefault(a_idx, a_name)
            self.idx_to_genre_name.setdefault(g_idx, g_name)
            self.artist_track_counts[a_idx] = self.artist_track_counts.get(a_idx, 0) + 1
            self.genre_track_counts[g_idx] = self.genre_track_counts.get(g_idx, 0) + 1

            prev_best = self.artist_top_track.get(a_idx)
            if prev_best is None or pops[it] > pops[prev_best]:
                self.artist_top_track[a_idx] = it

        # Compute L2-normalized genre & artist prototype embeddings in R^128
        self.genre_prototypes = np.zeros((self.num_genres, 128), dtype=np.float32)
        for g_idx in range(self.num_genres):
            mask = self.genre_ids == g_idx
            if np.any(mask):
                vec = self.item_embeddings[mask].mean(axis=0)
                self.genre_prototypes[g_idx] = vec / (np.linalg.norm(vec) + 1e-8)

        self.artist_prototypes = np.zeros((self.num_artists, 128), dtype=np.float32)
        self.artist_audio_medians = np.zeros((self.num_artists, 5), dtype=np.float32)
        self.artist_primary_genre = np.zeros(self.num_artists, dtype=np.int32)
        # Fast grouped aggregation via numpy bincount / sorting
        sort_a = np.argsort(self.artist_ids, kind="mergesort")
        sorted_a = self.artist_ids[sort_a]
        uniq_a, split_idx = np.unique(sorted_a, return_index=True)
        groups = np.split(sort_a, split_idx[1:])
        for a_idx, idxs in zip(uniq_a, groups):
            ai = int(a_idx)
            vec = self.item_embeddings[idxs].mean(axis=0)
            self.artist_prototypes[ai] = vec / (np.linalg.norm(vec) + 1e-8)
            self.artist_audio_medians[ai] = np.median(self.raw_audio_5d[idxs], axis=0)
            self.artist_primary_genre[ai] = int(self.genre_ids[idxs[0]])

        self.artist_index = ExactInnerProductIndex(self.artist_prototypes)
        self.available_genres_sorted = sorted(set(self.genre_names))

    def resolve_anchor_user(
        self,
        preferred_genres: list[str] | None = None,
        preferred_artists: list[str] | None = None,
    ) -> int:
        """Select a warm-start NeuMF anchor persona aligned with onboarding genre/artist tastes."""
        want_g = {g.lower().strip() for g in (preferred_genres or []) if g.strip()}
        want_a = {a.lower().strip() for a in (preferred_artists or []) if a.strip()}
        if not want_g and not want_a:
            return 0

        best_uid = 0
        best_score = -1.0
        for uid, hist in self.user_histories.items():
            pos = hist["items"][hist["actions"] >= 2]
            if len(pos) == 0:
                continue
            sample = pos[-60:]
            g_hits = sum(1 for i in sample if self.genre_names[int(i)].lower() in want_g) if want_g else 0
            a_hits = sum(1 for i in sample if self.artist_names[int(i)].lower() in want_a) if want_a else 0
            score = (g_hits / len(sample)) + 2.0 * (a_hits / len(sample))
            if score > best_score:
                best_score = score
                best_uid = uid
        return int(best_uid)

    @torch.no_grad()
    def compute_extension_user_representation(
        self,
        feedback_slot_idx: int,
        anchor_user_idx: int = 0,
        preferred_genres: list[str] | None = None,
        short_term_override: float | None = None,
    ) -> dict[str, Any]:
        """Compute Dual-Horizon (NeuMF + Causal Transformer + Taste Prior + Isolated Bandit) user state."""
        safe_anchor = anchor_user_idx if anchor_user_idx in self.user_histories else 0
        hist = self.user_histories[safe_anchor]
        fb_state = self.get_feedback_state(feedback_slot_idx)

        items_list = hist["items"][-40:].tolist()
        actions_list = hist["actions"][-40:].tolist()
        ts_list = hist["unix_ts"][-40:].tolist()

        for ev in fb_state.events[-35:]:
            if 0 <= ev.item_idx < self.num_items:
                items_list.append(ev.item_idx)
                actions_list.append(ev.action_id)
                ts_list.append(int(ev.timestamp))

        L = self.transformer.max_seq_len
        items_tail = items_list[-L:]
        actions_tail = actions_list[-L:]
        ts_tail = ts_list[-L:]

        deltas = [3600] + [max(0, int(ts_tail[k] - ts_tail[k - 1])) for k in range(1, len(ts_tail))]
        delta_buckets = bucketize_time_deltas(torch.tensor(deltas, dtype=torch.long)).tolist()

        pad_len = L - len(items_tail)
        item_seq = torch.tensor(
            [[self.num_items] * pad_len + items_tail], dtype=torch.long, device=self.device
        )
        act_seq = torch.tensor(
            [[PAD_ACTION_ID] * pad_len + actions_tail], dtype=torch.long, device=self.device
        )
        tb_seq = torch.tensor(
            [[PAD_TIME_BUCKET] * pad_len + delta_buckets], dtype=torch.long, device=self.device
        )
        u_id_t = torch.tensor([safe_anchor], dtype=torch.long, device=self.device)

        u_long = self.neumf.get_long_term_user_emb(u_id_t)
        u_short_seq = self.transformer.encode_short_term_sequence(item_seq, act_seq, tb_seq)
        u_short = u_short_seq[:, -1, :]

        u_fused_t, gate_t = self.transformer.fuse_horizons(
            u_long, u_short, short_term_override=short_term_override
        )
        u_fused = u_fused_t.squeeze(0).cpu().numpy().astype(np.float32)

        # Blend explicit user-configured genre taste prior smoothly if provided
        if preferred_genres:
            g_vecs = [
                self.genre_prototypes[self.genre_name_to_idx[g.lower().strip()]]
                for g in preferred_genres
                if g.lower().strip() in self.genre_name_to_idx
            ]
            if g_vecs:
                prior = np.mean(g_vecs, axis=0).astype(np.float32)
                prior /= np.linalg.norm(prior) + 1e-8
                u_fused = 0.72 * u_fused + 0.28 * prior
                u_fused /= np.linalg.norm(u_fused) + 1e-8

        if np.linalg.norm(fb_state.delta_u) > 1e-6:
            u_final = u_fused + fb_state.delta_u
            u_final = (u_final / (np.linalg.norm(u_final) + 1e-8)).astype(np.float32)
        else:
            u_final = u_fused

        return {
            "u_long": u_long.squeeze(0).cpu().numpy(),
            "u_short": u_short.squeeze(0).cpu().numpy(),
            "u_fused": u_final,
            "long_term_gate": float(gate_t.mean().item()),
            "recent_items": items_tail,
            "recent_actions": actions_tail,
            "anchor_user_idx": safe_anchor,
        }

    def build_discovery_feed(
        self,
        feedback_slot_idx: int,
        anchor_user_idx: int = 0,
        preferred_genres: list[str] | None = None,
        preferred_artists: list[str] | None = None,
        top_k: int = 18,
        discovery_intensity: float = 0.40,
        diversity_level: float = 0.50,
        novelty_boost: float = 0.30,
        short_term_weight: float | None = None,
        exclude_item_indices: list[int] | None = None,
        genre_filter: str | None = None,
    ) -> dict[str, Any]:
        """Generate rich Discovery payload: tracks, new discoveries, artists, related genres & explanations."""
        rep = self.compute_extension_user_representation(
            feedback_slot_idx=feedback_slot_idx,
            anchor_user_idx=anchor_user_idx,
            preferred_genres=preferred_genres,
            short_term_override=short_term_weight,
        )
        u_vec = rep["u_fused"]
        hist = self.user_histories[rep["anchor_user_idx"]]
        fb_state = self.get_feedback_state(feedback_slot_idx)

        known_items = hist["known_items"] | fb_state.liked_items | set(fb_state.played_items_recent)
        blocked_items = (
            fb_state.disliked_items
            | set(fb_state.played_items_recent[-12:])
            | {int(x) for x in (exclude_item_indices or []) if 0 <= int(x) < self.num_items}
        )

        cos_scores = (self.item_embeddings @ u_vec).astype(np.float32)
        pos_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (2, 3, 4)
        ] + list(fb_state.liked_items)
        neg_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (0, 5)
        ] + list(fb_state.disliked_items)
        cf_raw = self.item_cf.score_for_history(pos_seed, neg_seed)
        cf_rrf = reciprocal_rank_score(cf_raw, k_rrf=60, positive_only=True)
        cos_rrf = reciprocal_rank_score(cos_scores, k_rrf=60, positive_only=False)
        bandit_scores = fb_state.linucb_bonus(
            self.item_embeddings, alpha_explore=0.10 + 0.18 * discovery_intensity
        )

        dyn_adj = np.zeros(self.num_items, dtype=np.float32)
        for a_idx, pen in fb_state.artist_fatigue.items():
            dyn_adj[self.artist_ids == a_idx] -= 0.45 * pen
        for g_idx, pen in fb_state.genre_fatigue.items():
            dyn_adj[self.genre_ids == g_idx] -= 0.30 * pen
        for a_idx, bst in fb_state.artist_boost.items():
            dyn_adj[self.artist_ids == a_idx] += 0.22 * bst
        for g_idx, bst in fb_state.genre_boost.items():
            dyn_adj[self.genre_ids == g_idx] += 0.14 * bst

        if preferred_genres:
            for g_str in preferred_genres:
                g_id = self.genre_name_to_idx.get(g_str.lower().strip())
                if g_id is not None:
                    dyn_adj[self.genre_ids == g_id] += 0.10
        if preferred_artists:
            for a_str in preferred_artists:
                a_id = self.artist_name_to_idx.get(a_str.lower().strip())
                if a_id is not None:
                    dyn_adj[self.artist_ids == a_id] += 0.16

        base_relevance = (
            0.55 * cos_scores
            + 0.22 * (0.78 * cos_rrf + 0.22 * cf_rrf)
            + 0.20 * bandit_scores
            + novelty_boost * 0.22 * self.item_novelty
            + dyn_adj
        )

        for b_it in blocked_items:
            base_relevance[b_it] = -999.0

        if genre_filter and genre_filter.lower().strip() in self.genre_name_to_idx:
            target_gid = self.genre_name_to_idx[genre_filter.lower().strip()]
            base_relevance = np.where(self.genre_ids == target_gid, base_relevance, -999.0)

        is_known_mask = np.zeros(self.num_items, dtype=bool)
        if known_items:
            is_known_mask[list(known_items)] = True

        n_disc_target = int(round(top_k * float(np.clip(discovery_intensity, 0.0, 1.0))))
        n_fam_target = top_k - n_disc_target

        fam_candidates = np.flatnonzero(is_known_mask & (base_relevance > -100))
        disc_candidates = np.flatnonzero((~is_known_mask) & (base_relevance > -100))

        fam_top = fam_candidates[np.argsort(-base_relevance[fam_candidates])[: max(120, n_fam_target * 5)]]
        disc_top = disc_candidates[np.argsort(-base_relevance[disc_candidates])[: max(250, n_disc_target * 6)]]

        selected_fam = self._mmr_select(fam_top, base_relevance, min(n_fam_target, len(fam_top)), diversity_level)
        rem_slots = top_k - len(selected_fam)
        selected_disc = self._mmr_select(
            disc_top, base_relevance, rem_slots, diversity_level, already_selected=selected_fam
        )

        merged = selected_fam + selected_disc
        merged.sort(key=lambda idx: float(base_relevance[idx]), reverse=True)

        tracks_out = []
        for it in merged:
            rec = self._format_track_recommendation(
                item_idx=it,
                score=float(base_relevance[it]),
                cos_sim=float(cos_scores[it]),
                cf_score=float(cf_rrf[it]),
                is_discovery=not bool(is_known_mask[it]),
                rep=rep,
                fb_state=fb_state,
            )
            rec["spotify_url"] = f"https://open.spotify.com/track/{rec['spotify_track_id']}"
            rec["spotify_uri"] = f"spotify:track:{rec['spotify_track_id']}"
            tracks_out.append(rec)

        new_discoveries = [t for t in tracks_out if t["is_discovery"]][:8]

        # Recommended Artists (ranked by prototype alignment + bandit boost - fatigue)
        artist_aff = (self.artist_prototypes @ u_vec).astype(np.float32)
        for a_idx, bst in fb_state.artist_boost.items():
            if 0 <= a_idx < self.num_artists:
                artist_aff[a_idx] += 0.20 * bst
        for a_idx, pen in fb_state.artist_fatigue.items():
            if 0 <= a_idx < self.num_artists:
                artist_aff[a_idx] -= 0.55 * pen

        known_artists = {int(self.artist_ids[i]) for i in known_items}
        top_artist_idxs = np.argsort(-artist_aff)[:24]
        recommended_artists = []
        seen_g_art: dict[int, int] = {}
        for a_id in top_artist_idxs:
            ai = int(a_id)
            if ai not in self.idx_to_artist_name:
                continue
            g_id = int(self.artist_primary_genre[ai])
            if seen_g_art.get(g_id, 0) >= 3:
                continue
            seen_g_art[g_id] = seen_g_art.get(g_id, 0) + 1
            rep_it = self.artist_top_track[ai]
            rep_row = self.tracks_df.iloc[rep_it]
            is_new_artist = ai not in known_artists
            recommended_artists.append(
                {
                    "artist_idx": ai,
                    "artist_name": self.idx_to_artist_name[ai],
                    "genre": self.idx_to_genre_name.get(g_id, "Unknown"),
                    "catalog_track_count": self.artist_track_counts.get(ai, 1),
                    "affinity_score": round(float(artist_aff[ai]), 3),
                    "is_discovery": is_new_artist,
                    "representative_track": {
                        "item_idx": int(rep_it),
                        "spotify_track_id": str(rep_row["spotify_track_id"]),
                        "track_name": str(rep_row["track_name"]),
                        "spotify_url": f"https://open.spotify.com/track/{rep_row['spotify_track_id']}",
                    },
                    "reason": (
                        "Unexplored artist matching your multimodal taste vector"
                        if is_new_artist
                        else "High-affinity artist reinforced by your listening history"
                    ),
                }
            )
            if len(recommended_artists) >= 8:
                break

        # Related Genres (familiar core + adjacent discovery genres)
        genre_aff = (self.genre_prototypes @ u_vec).astype(np.float32)
        for g_idx, bst in fb_state.genre_boost.items():
            if 0 <= g_idx < self.num_genres:
                genre_aff[g_idx] += 0.18 * bst
        for g_idx, pen in fb_state.genre_fatigue.items():
            if 0 <= g_idx < self.num_genres:
                genre_aff[g_idx] -= 0.45 * pen

        known_genres = {int(self.genre_ids[i]) for i in known_items}
        related_genres = []
        for g_id in np.argsort(-genre_aff)[:10]:
            gi = int(g_id)
            if gi not in self.idx_to_genre_name:
                continue
            related_genres.append(
                {
                    "genre_idx": gi,
                    "genre": self.idx_to_genre_name[gi],
                    "affinity_score": round(float(genre_aff[gi]), 3),
                    "track_count": self.genre_track_counts.get(gi, 0),
                    "is_exploratory": gi not in known_genres,
                }
            )

        actual_disc = sum(1 for t in tracks_out if t["is_discovery"])
        return {
            "recommended_tracks": tracks_out,
            "new_discoveries": new_discoveries,
            "recommended_artists": recommended_artists,
            "related_genres": related_genres,
            "familiarity_vs_discovery": {
                "target_discovery_ratio": round(float(discovery_intensity), 3),
                "actual_discovery_ratio": round(actual_disc / max(1, len(tracks_out)), 3),
                "familiar_count": len(tracks_out) - actual_disc,
                "discovery_count": actual_disc,
                "long_term_gate": round(rep["long_term_gate"], 3),
                "short_term_gate": round(1.0 - rep["long_term_gate"], 3),
            },
            "feedback_summary": fb_state.summary(),
        }

    def get_similar_tracks(
        self,
        seed_item_idx: int,
        top_k: int = 10,
        exclude_ids: set[int] | None = None,
    ) -> dict[str, Any]:
        """Retrieve nearest neighbor tracks in shared 128D multimodal space."""
        if not (0 <= seed_item_idx < self.num_items):
            raise ValueError(f"Track index {seed_item_idx} out of range.")
        seed_row = self.tracks_df.iloc[seed_item_idx]
        seed_emb = self.item_embeddings[seed_item_idx : seed_item_idx + 1]
        sims, idxs = self.vector_index.search(seed_emb, k=min(self.num_items, top_k + 25))
        excluded = (exclude_ids or set()) | {seed_item_idx}

        results = []
        for sim_val, cand_it in zip(sims[0].tolist(), idxs[0].tolist()):
            it = int(cand_it)
            if it in excluded:
                continue
            row = self.tracks_df.iloc[it]
            same_genre = str(row["track_genre"]) == str(seed_row["track_genre"])
            bpm_diff = abs(float(row["tempo"]) - float(seed_row["tempo"]))
            reason_parts = [f"{int(round(sim_val * 100))}% multimodal embedding match"]
            if same_genre:
                reason_parts.append(f"shared {row['track_genre']} roots")
            if bpm_diff <= 10.0:
                reason_parts.append("matched tempo groove")
            results.append(
                {
                    "item_idx": it,
                    "spotify_track_id": str(row["spotify_track_id"]),
                    "spotify_url": f"https://open.spotify.com/track/{row['spotify_track_id']}",
                    "spotify_uri": f"spotify:track:{row['spotify_track_id']}",
                    "track_name": str(row["track_name"]),
                    "artist_name": str(row["primary_artist"]),
                    "album_name": str(row["album_name"]),
                    "genre": str(row["track_genre"]),
                    "duration_ms": int(row["duration_ms"]),
                    "similarity": round(float(sim_val), 4),
                    "reason": " · ".join(reason_parts[:2]),
                }
            )
            if len(results) >= top_k:
                break

        return {
            "seed_track": {
                "item_idx": int(seed_item_idx),
                "spotify_track_id": str(seed_row["spotify_track_id"]),
                "track_name": str(seed_row["track_name"]),
                "artist_name": str(seed_row["primary_artist"]),
                "genre": str(seed_row["track_genre"]),
            },
            "similar_tracks": results,
        }

    def get_similar_artists(self, artist_query: str, top_k: int = 8) -> dict[str, Any]:
        """Find nearest artists in 128D multimodal prototype space."""
        q_norm = artist_query.lower().strip()
        seed_a_idx = self.artist_name_to_idx.get(q_norm)
        if seed_a_idx is None:
            for name_low, idx in self.artist_name_to_idx.items():
                if q_norm in name_low:
                    seed_a_idx = idx
                    break
        if seed_a_idx is None:
            raise KeyError(f"Artist '{artist_query}' not found in catalog.")

        seed_vec = self.artist_prototypes[seed_a_idx : seed_a_idx + 1]
        sims, idxs = self.artist_index.search(seed_vec, k=min(self.num_artists, top_k + 5))

        out = []
        for sim_val, cand_a in zip(sims[0].tolist(), idxs[0].tolist()):
            ai = int(cand_a)
            if ai == seed_a_idx or ai not in self.idx_to_artist_name:
                continue
            rep_it = self.artist_top_track[ai]
            rep_row = self.tracks_df.iloc[rep_it]
            g_id = int(self.artist_primary_genre[ai])
            out.append(
                {
                    "artist_idx": ai,
                    "artist_name": self.idx_to_artist_name[ai],
                    "genre": self.idx_to_genre_name.get(g_id, "Unknown"),
                    "similarity": round(float(sim_val), 4),
                    "catalog_track_count": self.artist_track_counts.get(ai, 1),
                    "sample_track": {
                        "item_idx": int(rep_it),
                        "spotify_track_id": str(rep_row["spotify_track_id"]),
                        "track_name": str(rep_row["track_name"]),
                        "spotify_url": f"https://open.spotify.com/track/{rep_row['spotify_track_id']}",
                    },
                }
            )
            if len(out) >= top_k:
                break

        return {
            "seed_artist": {
                "artist_idx": int(seed_a_idx),
                "artist_name": self.idx_to_artist_name[seed_a_idx],
                "genre": self.idx_to_genre_name.get(int(self.artist_primary_genre[seed_a_idx]), "Unknown"),
            },
            "similar_artists": out,
        }

    @staticmethod
    def presets_to_shuffle_config(
        presets: dict[str, Any],
        seed: int | None = None,
    ) -> ShuffleConfig:
        """Map human-readable Adaptive Shuffle Studio presets onto quantitative ShuffleConfig weights."""
        art_rep = presets.get("artist_repetition", "balanced")
        gen_div = presets.get("genre_diversity", "medium")
        coherence = presets.get("musical_coherence", "smooth")
        disc_mode = presets.get("discovery_mode", "balanced")
        rand_mode = presets.get("randomness", "balanced")
        urgency = float(presets.get("quota_urgency_weight", 0.0))

        artist_cooldown = {"strict": 6, "balanced": 4, "flexible": 2}.get(art_rep, 4)
        if art_rep == "strict" and urgency == 0.0:
            urgency = 0.55

        diversity_weight, genre_cooldown = {
            "low": (0.30, 1),
            "medium": (0.65, 2),
            "high": (1.05, 3),
        }.get(gen_div, (0.65, 2))

        flow_weight = {"smooth": 0.85, "balanced": 0.45, "variety": 0.15}.get(coherence, 0.45)
        discovery_target = {"familiar": 0.15, "balanced": 0.35, "exploratory": 0.70}.get(disc_mode, 0.35)
        entropy_temperature = {"predictable": 0.06, "balanced": 0.18, "varied": 0.42}.get(rand_mode, 0.18)

        return ShuffleConfig(
            mode="adaptive",
            diversity_weight=diversity_weight,
            flow_weight=flow_weight,
            relevance_weight=0.55,
            artist_cooldown=artist_cooldown,
            genre_cooldown=genre_cooldown,
            entropy_temperature=entropy_temperature,
            discovery_target=discovery_target,
            quota_urgency_weight=urgency,
            seed=seed,
        )

    def resolve_external_tracks(self, external_tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Match authorized Spotify / external track metadata to independent catalog indices where available,
        with deterministic metadata-only fallback (never training on or storing external Spotify Content).
        """
        resolved: list[dict[str, Any]] = []
        for idx, tr in enumerate(external_tracks):
            sid = (tr.get("spotify_track_id") or "").strip()
            t_name = str(tr.get("track_name") or "Untitled").strip()
            a_name = str(tr.get("artist_name") or "Unknown Artist").strip()
            alb_name = str(tr.get("album_name") or "Unknown Album").strip()
            dur_ms = int(tr.get("duration_ms") or 210000)

            matched_it = self.spotify_id_to_idx.get(sid) if sid else None
            if matched_it is None:
                matched_it = self.norm_title_artist_to_idx.get((t_name.lower(), a_name.lower()))

            if matched_it is not None:
                row = self.tracks_df.iloc[matched_it]
                resolved.append(
                    {
                        "pool_index": idx,
                        "matched_catalog": True,
                        "feature_source": "independent_catalog",
                        "item_idx": int(matched_it),
                        "spotify_track_id": sid or str(row["spotify_track_id"]),
                        "track_name": t_name or str(row["track_name"]),
                        "artist_name": a_name or str(row["primary_artist"]),
                        "album_name": alb_name if alb_name != "Unknown Album" else str(row["album_name"]),
                        "genre": str(row["track_genre"]),
                        "duration_ms": dur_ms or int(row["duration_ms"]),
                        "artist_id": int(self.artist_ids[matched_it]),
                        "genre_id": int(self.genre_ids[matched_it]),
                        "embedding": self.item_embeddings[matched_it],
                        "raw_audio": self.raw_audio_5d[matched_it],
                    }
                )
            else:
                known_a = self.artist_name_to_idx.get(a_name.lower())
                req_genre = (tr.get("genre") or "").lower().strip()
                known_g = self.genre_name_to_idx.get(req_genre)
                if known_a is not None:
                    emb = self.artist_prototypes[known_a]
                    aud = self.artist_audio_medians[known_a]
                    gid = known_g if known_g is not None else int(self.artist_primary_genre[known_a])
                    g_label = self.idx_to_genre_name.get(gid, tr.get("genre") or "Unclassified")
                    feat_src = "artist_catalog_prior"
                    aid = int(known_a)
                else:
                    # Deterministic artist hash ID ensures exact artist spacing even without audio features
                    aid = 1_000_000 + (abs(hash(a_name.lower())) % 900_000)
                    if known_g is not None:
                        emb = self.genre_prototypes[known_g]
                        gid = int(known_g)
                        g_label = self.idx_to_genre_name.get(gid, req_genre)
                    else:
                        # Neutral unit vector orthogonalized deterministically by artist hash
                        rng_fb = np.random.default_rng(aid)
                        emb = rng_fb.normal(0, 1, size=128).astype(np.float32)
                        emb /= np.linalg.norm(emb) + 1e-8
                        gid = 500_000 + (aid % 50)
                        g_label = tr.get("genre") or "External / Unclassified"
                    aud = np.array([0.55, 0.60, 0.50, 120.0, 0.0], dtype=np.float32)
                    feat_src = "metadata_only_fallback"

                resolved.append(
                    {
                        "pool_index": idx,
                        "matched_catalog": False,
                        "feature_source": feat_src,
                        "item_idx": None,
                        "spotify_track_id": sid,
                        "track_name": t_name,
                        "artist_name": a_name,
                        "album_name": alb_name,
                        "genre": g_label,
                        "duration_ms": dur_ms,
                        "artist_id": aid,
                        "genre_id": gid,
                        "embedding": emb,
                        "raw_audio": aud,
                    }
                )
        return resolved

    def generate_companion_shuffle(
        self,
        feedback_slot_idx: int,
        anchor_user_idx: int,
        preferred_genres: list[str] | None,
        source_type: str,
        playlist_pid: int | None,
        source_name: str | None,
        pool_size: int,
        external_tracks: list[dict[str, Any]] | None,
        presets: dict[str, Any],
        include_baselines: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate Adaptive Shuffle sequence with original order comparison, distributions & export formats."""
        from backend.app.engine.adaptive_shuffle import evaluate_sequence_metrics

        rep = self.compute_extension_user_representation(
            feedback_slot_idx=feedback_slot_idx,
            anchor_user_idx=anchor_user_idx,
            preferred_genres=preferred_genres,
        )
        u_vec = rep["u_fused"]
        fb_state = self.get_feedback_state(feedback_slot_idx)
        known = self.user_histories[rep["anchor_user_idx"]]["known_items"] | fb_state.liked_items

        if source_type == "authorized_tracks" and external_tracks:
            resolved_pool = self.resolve_external_tracks(external_tracks[:pool_size])
            resolved_name = source_name or "Authorized Track Selection"
        elif source_type == "catalog_playlist" and playlist_pid is not None:
            match = self.playlists_df[self.playlists_df["pid"] == playlist_pid]
            if len(match) == 0:
                raise KeyError(f"Playlist {playlist_pid} not found.")
            idxs = [int(i) for i in match.iloc[0]["item_indices"][:pool_size] if 0 <= int(i) < self.num_items]
            resolved_name = source_name or str(match.iloc[0]["name"])
            resolved_pool = self.resolve_external_tracks(
                [
                    {
                        "spotify_track_id": str(self.tracks_df.iloc[i]["spotify_track_id"]),
                        "track_name": str(self.tracks_df.iloc[i]["track_name"]),
                        "artist_name": str(self.tracks_df.iloc[i]["primary_artist"]),
                        "album_name": str(self.tracks_df.iloc[i]["album_name"]),
                        "duration_ms": int(self.tracks_df.iloc[i]["duration_ms"]),
                    }
                    for i in idxs
                ]
            )
        else:
            disc_feed = self.build_discovery_feed(
                feedback_slot_idx=feedback_slot_idx,
                anchor_user_idx=anchor_user_idx,
                preferred_genres=preferred_genres,
                top_k=pool_size,
                discovery_intensity= {"familiar": 0.15, "balanced": 0.35, "exploratory": 0.70}.get(
                    presets.get("discovery_mode", "balanced"), 0.35
                ),
                diversity_level=0.35,
            )
            resolved_name = source_name or "Personalized Discovery Mix"
            resolved_pool = self.resolve_external_tracks(disc_feed["recommended_tracks"])

        m = len(resolved_pool)
        if m == 0:
            raise ValueError("Selected track collection is empty.")

        embs = np.vstack([r["embedding"] for r in resolved_pool]).astype(np.float32)
        arts = np.array([r["artist_id"] for r in resolved_pool], dtype=np.int32)
        gens = np.array([r["genre_id"] for r in resolved_pool], dtype=np.int32)
        aud = np.vstack([r["raw_audio"] for r in resolved_pool]).astype(np.float32)
        rel = (embs @ u_vec).astype(np.float32)
        is_disc = np.array(
            [(r["item_idx"] not in known) if r["item_idx"] is not None else True for r in resolved_pool],
            dtype=bool,
        )

        base_seed = seed if seed is not None else int(np.random.randint(1, 1_000_000))
        cfg = self.presets_to_shuffle_config(presets, seed=base_seed)

        adaptive_res = self.shuffle_engine.sequence(
            embeddings=embs,
            artist_ids=arts,
            genre_ids=gens,
            raw_audio=aud,
            relevance_scores=rel,
            is_discovery=is_disc,
            config=cfg,
            artist_fatigue=fb_state.artist_fatigue,
            genre_fatigue=fb_state.genre_fatigue,
        )
        orig_metrics = evaluate_sequence_metrics(
            ordered_pos=list(range(m)),
            embeddings=embs,
            artist_ids=arts,
            genre_ids=gens,
            raw_audio=aud,
            relevance_scores=rel,
            is_discovery=is_disc,
        )

        def _format_seq(order_positions: list[int], step_details: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
            out = []
            for pos_idx, p in enumerate(order_positions):
                item = resolved_pool[p]
                has_audio = item["feature_source"] != "metadata_only_fallback"
                sid = item["spotify_track_id"]
                out.append(
                    {
                        "position": pos_idx + 1,
                        "original_position": p + 1,
                        "item_idx": item["item_idx"],
                        "spotify_track_id": sid,
                        "spotify_url": f"https://open.spotify.com/track/{sid}" if sid else None,
                        "spotify_uri": f"spotify:track:{sid}" if sid else None,
                        "track_name": item["track_name"],
                        "artist_name": item["artist_name"],
                        "album_name": item["album_name"],
                        "genre": item["genre"],
                        "duration_ms": item["duration_ms"],
                        "is_discovery": bool(is_disc[p]),
                        "feature_source": item["feature_source"],
                        "audio": (
                            {
                                "danceability": round(float(item["raw_audio"][0]), 3),
                                "energy": round(float(item["raw_audio"][1]), 3),
                                "valence": round(float(item["raw_audio"][2]), 3),
                                "tempo": round(float(item["raw_audio"][3]), 1),
                            }
                            if has_audio
                            else None
                        ),
                        "transition": step_details[pos_idx] if step_details else None,
                    }
                )
            return out

        original_tracks = _format_seq(list(range(m)))
        generated_tracks = _format_seq(adaptive_res["ordered_positions"], adaptive_res["step_details"])

        # Artist & Genre distributions
        art_counts: dict[str, int] = {}
        gen_counts: dict[str, int] = {}
        for r in resolved_pool:
            art_counts[r["artist_name"]] = art_counts.get(r["artist_name"], 0) + 1
            gen_counts[r["genre"]] = gen_counts.get(r["genre"], 0) + 1

        artist_distribution = [
            {"artist": k, "count": v, "share": round(v / m, 3)}
            for k, v in sorted(art_counts.items(), key=lambda kv: -kv[1])[:10]
        ]
        genre_distribution = [
            {"genre": k, "count": v, "share": round(v / m, 3)}
            for k, v in sorted(gen_counts.items(), key=lambda kv: -kv[1])[:10]
        ]

        catalog_matched_count = sum(1 for r in resolved_pool if r["matched_catalog"])
        disc_count = int(is_disc.sum())

        baselines = {}
        if include_baselines:
            fy = self.shuffle_engine.sequence(
                embs, arts, gens, aud, rel, is_disc, ShuffleConfig(mode="fisher_yates", seed=base_seed)
            )
            sp = self.shuffle_engine.sequence(
                embs, arts, gens, aud, rel, is_disc, ShuffleConfig(mode="artist_spaced", seed=base_seed)
            )
            baselines = {
                "original_order": orig_metrics,
                "fisher_yates": fy["metrics"],
                "artist_spaced": sp["metrics"],
            }

        # Build functional export/handoff payloads
        uris = [t["spotify_uri"] for t in generated_tracks if t["spotify_uri"]]
        urls = [t["spotify_url"] for t in generated_tracks if t["spotify_url"]]
        m3u_lines = ["#EXTM3U", f"#PLAYLIST:{resolved_name} (SPOTIFY-THING Adaptive Shuffle)"]
        csv_lines = ["position,track_name,artist_name,album_name,genre,spotify_url"]
        for t in generated_tracks:
            sec = max(1, int(round(t["duration_ms"] / 1000)))
            m3u_lines.append(f"#EXTINF:{sec},{t['artist_name']} - {t['track_name']}")
            m3u_lines.append(t["spotify_url"] or f"spotify:search:{t['artist_name']} {t['track_name']}")
            safe_t = t["track_name"].replace('"', '""')
            safe_a = t["artist_name"].replace('"', '""')
            safe_al = t["album_name"].replace('"', '""')
            csv_lines.append(
                f'{t["position"]},"{safe_t}","{safe_a}","{safe_al}","{t["genre"]}",{t["spotify_url"] or ""}'
            )

        return {
            "source_name": resolved_name,
            "source_type": source_type,
            "pool_size": m,
            "seed": base_seed,
            "applied_config": {
                "presets": presets,
                "diversity_weight": cfg.diversity_weight,
                "flow_weight": cfg.flow_weight,
                "artist_cooldown": cfg.artist_cooldown,
                "genre_cooldown": cfg.genre_cooldown,
                "entropy_temperature": cfg.entropy_temperature,
                "discovery_target": cfg.discovery_target,
                "quota_urgency_weight": cfg.quota_urgency_weight,
            },
            "feature_coverage": {
                "catalog_audio_matched": catalog_matched_count,
                "fallback_tracks": m - catalog_matched_count,
                "audio_coverage_ratio": round(catalog_matched_count / m, 3),
                "note": (
                    "All tracks matched independent multimodal audio & embedding features."
                    if catalog_matched_count == m
                    else f"{catalog_matched_count}/{m} tracks matched catalog audio features; remaining tracks sequenced using artist/metadata spacing fallback."
                ),
            },
            "original_tracks": original_tracks,
            "generated_tracks": generated_tracks,
            "metrics": adaptive_res["metrics"],
            "baselines": baselines,
            "distributions": {
                "artists": artist_distribution,
                "genres": genre_distribution,
                "discovery": {
                    "familiar_count": m - disc_count,
                    "discovery_count": disc_count,
                    "discovery_ratio": round(disc_count / m, 3),
                },
                "repetition": {
                    "unique_artists": len(art_counts),
                    "repeated_artists_count": sum(1 for c in art_counts.values() if c > 1),
                    "original_clump_rate": orig_metrics["artist_clump_rate"],
                    "generated_clump_rate": adaptive_res["metrics"]["artist_clump_rate"],
                    "min_artist_gap": adaptive_res["metrics"]["min_artist_gap"],
                    "mean_artist_gap": adaptive_res["metrics"]["mean_artist_gap"],
                },
            },
            "handoff": {
                "direct_spotify_shuffle_override": False,
                "limitation_notice": (
                    "SPOTIFY-THING generates an independent sequence. Spotify remains the playback platform; "
                    "use Open in Spotify, Copy Track URIs, Export M3U/CSV, or Create Authorized Spotify Playlist when permitted."
                ),
                "spotify_uris": uris,
                "spotify_urls": urls,
                "m3u": "\n".join(m3u_lines),
                "csv": "\n".join(csv_lines),
            },
        }

    def compute_companion_insights(
        self,
        feedback_slot_idx: int,
        anchor_user_idx: int,
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compute transparent Music Insights strictly from permitted data, explicitly separating:
        - `observed`            : Explicit user feedback events & user-selected preferences in SPOTIFY-THING
        - `inferred_preference` : Learned LinUCB bandit weights, fatigue memory & taste radar shifts
        - `model_prediction`    : NeuMF + Dual-Horizon Transformer predicted affinities & recommendation diversity
        Never invents listening-time history or unrecorded playback counts.
        """
        fb_state = self.get_feedback_state(feedback_slot_idx)
        pref_genres = preferences.get("preferred_genres") or []
        rep = self.compute_extension_user_representation(
            feedback_slot_idx=feedback_slot_idx,
            anchor_user_idx=anchor_user_idx,
            preferred_genres=pref_genres,
        )
        u_vec = rep["u_fused"]

        # 1. OBSERVED: User's actual recorded feedback actions & selected genres inside SPOTIFY-THING
        obs_genre_counts: dict[str, int] = {}
        obs_artist_counts: dict[str, int] = {}
        pos_events = [ev for ev in fb_state.events if ev.reward > 0]
        neg_events = [ev for ev in fb_state.events if ev.reward < 0]

        for ev in pos_events:
            g_name = self.idx_to_genre_name.get(ev.genre_idx, "Unknown")
            a_name = self.idx_to_artist_name.get(ev.artist_idx, "Unknown")
            obs_genre_counts[g_name] = obs_genre_counts.get(g_name, 0) + 1
            obs_artist_counts[a_name] = obs_artist_counts.get(a_name, 0) + 1

        observed_section = {
            "provenance": "observed",
            "description": "Directly recorded feedback interactions and explicit settings in SPOTIFY-THING (no fabricated listening-time history).",
            "total_feedback_actions": len(fb_state.events),
            "positive_actions": len(pos_events),
            "negative_actions": len(neg_events),
            "liked_tracks_count": len(fb_state.liked_items),
            "dismissed_or_disliked_count": len(fb_state.disliked_items),
            "explicit_preferred_genres": pref_genres,
            "explicit_preferred_artists": preferences.get("preferred_artists") or [],
            "observed_top_genres": [
                {"genre": g, "positive_signals": c}
                for g, c in sorted(obs_genre_counts.items(), key=lambda kv: -kv[1])[:6]
            ],
            "observed_top_artists": [
                {"artist": a, "positive_signals": c}
                for a, c in sorted(obs_artist_counts.items(), key=lambda kv: -kv[1])[:6]
            ],
        }

        # 2. INFERRED PREFERENCES: Bandit drift, boosted/fatigued artists & genres, sonic profile
        boosted_genres = [
            {"genre": self.idx_to_genre_name.get(gid, str(gid)), "boost_weight": round(val, 3)}
            for gid, val in sorted(fb_state.genre_boost.items(), key=lambda kv: -kv[1])[:5]
            if gid in self.idx_to_genre_name
        ]
        fatigued_genres = [
            {"genre": self.idx_to_genre_name.get(gid, str(gid)), "fatigue_penalty": round(val, 3)}
            for gid, val in sorted(fb_state.genre_fatigue.items(), key=lambda kv: -kv[1])[:5]
            if gid in self.idx_to_genre_name
        ]
        boosted_artists = [
            {"artist": self.idx_to_artist_name.get(aid, str(aid)), "boost_weight": round(val, 3)}
            for aid, val in sorted(fb_state.artist_boost.items(), key=lambda kv: -kv[1])[:5]
            if aid in self.idx_to_artist_name
        ]
        fatigued_artists = [
            {"artist": self.idx_to_artist_name.get(aid, str(aid)), "fatigue_penalty": round(val, 3)}
            for aid, val in sorted(fb_state.artist_fatigue.items(), key=lambda kv: -kv[1])[:5]
            if aid in self.idx_to_artist_name
        ]

        # Cosine drift between baseline anchor taste (u_long) and adapted session vector (u_fused)
        u_long_norm = rep["u_long"] / (np.linalg.norm(rep["u_long"]) + 1e-8)
        preference_stability = float(np.clip(np.dot(u_long_norm, u_vec), -1.0, 1.0))

        inferred_section = {
            "provenance": "inferred_preference",
            "description": "Preferences inferred from LinUCB contextual bandit updates, EMA residual shift, and decaying fatigue memory.",
            "bandit_residual_magnitude": round(float(np.linalg.norm(fb_state.delta_u)), 4),
            "preference_drift_score": round(float(1.0 - preference_stability), 4),
            "long_term_horizon_weight": round(rep["long_term_gate"], 3),
            "short_term_session_weight": round(1.0 - rep["long_term_gate"], 3),
            "configured_discovery_intensity": round(float(preferences.get("discovery_intensity", 0.40)), 3),
            "boosted_genres": boosted_genres,
            "fatigued_genres": fatigued_genres,
            "boosted_artists": boosted_artists,
            "fatigued_artists": fatigued_artists,
        }

        # 3. MODEL PREDICTIONS: Predicted top genres/artists, recommendation diversity (ILD), repeated artist exposure
        disc_sample = self.build_discovery_feed(
            feedback_slot_idx=feedback_slot_idx,
            anchor_user_idx=anchor_user_idx,
            preferred_genres=pref_genres,
            top_k=20,
            discovery_intensity=float(preferences.get("discovery_intensity", 0.40)),
            diversity_level=float(preferences.get("diversity_level", 0.50)),
            novelty_boost=float(preferences.get("novelty_boost", 0.30)),
        )
        rec_tracks = disc_sample["recommended_tracks"]
        rec_idxs = [t["item_idx"] for t in rec_tracks]
        if len(rec_idxs) >= 2:
            sub_emb = self.item_embeddings[rec_idxs]
            sim_m = np.clip(sub_emb @ sub_emb.T, -1.0, 1.0)
            tri = np.triu(np.ones((len(rec_idxs), len(rec_idxs)), dtype=bool), k=1)
            rec_ild = float(np.mean(1.0 - sim_m[tri]))
        else:
            rec_ild = 0.0

        rec_artists = [t["artist_name"] for t in rec_tracks]
        unique_rec_artists = len(set(rec_artists))
        repeated_artist_exposure = round(1.0 - (unique_rec_artists / max(1, len(rec_artists))), 3)

        radar_cols = ["danceability", "energy", "valence", "acousticness", "instrumentalness"]
        rec_df = self.tracks_df.iloc[rec_idxs] if rec_idxs else self.tracks_df.head(10)
        predicted_sonic_profile = {c: round(float(rec_df[c].mean()), 3) for c in radar_cols}

        predictions_section = {
            "provenance": "model_prediction",
            "description": "Forward-looking predictions generated by Multimodal NeuMF + Dual-Horizon Transformer + MMR.",
            "predicted_favorite_genres": disc_sample["related_genres"][:6],
            "predicted_favorite_artists": [
                {
                    "artist": a["artist_name"],
                    "genre": a["genre"],
                    "affinity_score": a["affinity_score"],
                    "is_discovery": a["is_discovery"],
                }
                for a in disc_sample["recommended_artists"][:6]
            ],
            "recommendation_diversity_ild": round(rec_ild, 4),
            "unique_artists_in_top20": unique_rec_artists,
            "repeated_artist_exposure_rate": repeated_artist_exposure,
            "familiarity_vs_discovery": disc_sample["familiarity_vs_discovery"],
            "predicted_sonic_profile": predicted_sonic_profile,
        }

        return {
            "observed": observed_section,
            "inferred_preferences": inferred_section,
            "model_predictions": predictions_section,
        }
