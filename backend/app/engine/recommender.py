"""
Hybrid Music Recommendation Engine.

Unifies:
- Shared Multimodal Track Embeddings + FAISS Inner-Product Retrieval
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
from backend.app.engine.online_feedback import UserFeedbackState
from backend.app.models.collaborative import ItemCooccurrenceCF, NeuralCollaborativeFiltering
from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import (
    PAD_ACTION_ID,
    PAD_TIME_BUCKET,
    DualHorizonSessionTransformer,
    bucketize_time_deltas,
)


class HybridRecommendationEngine:
    def __init__(self, processed_dir: Path, artifacts_dir: Path, device: str = "cpu") -> None:
        self.processed_dir = processed_dir
        self.artifacts_dir = artifacts_dir
        self.device = torch.device(device)

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

        # Load trained embeddings, FAISS index, CF, NeuMF, and Dual-Horizon Transformer
        self._load_models()
        self.shuffle_engine = AdaptiveShuffleEngine()
        self.feedback_states: dict[int, UserFeedbackState] = {}

    def _load_models(self) -> None:
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
            self.feedback_states[user_idx] = UserFeedbackState(user_idx=user_idx, embed_dim=128)
        return self.feedback_states[user_idx]

    def reset_feedback_state(self, user_idx: int) -> None:
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

        # 1. Shared Embedding Cosine Similarity via FAISS over full catalog
        cos_scores = (self.item_embeddings @ u_vec).astype(np.float32)

        # 2. Item-KNN Collaborative Filtering Score from positive history
        pos_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (2, 3, 4)
        ] + list(fb_state.liked_items)
        neg_seed = [
            it for it, act in zip(rep["recent_items"], rep["recent_actions"]) if act in (0, 5)
        ] + list(fb_state.disliked_items)
        cf_raw = self.item_cf.score_for_history(pos_seed, neg_seed)
        cf_norm = cf_raw / (float(cf_raw.max()) + 1e-6)

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

        base_relevance = (
            0.55 * cos_scores
            + 0.25 * cf_norm
            + 0.20 * bandit_scores
            + novelty_boost * 0.22 * self.item_novelty
            + dyn_adj
        )

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
