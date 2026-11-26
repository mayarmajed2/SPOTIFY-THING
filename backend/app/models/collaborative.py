"""
Collaborative Filtering & Neural Collaborative Filtering (NeuMF) Models.

Includes:
1. Popularity & Playlist/Session Co-occurrence CF Baseline (Item-KNN sparse cosine)
2. BPRMatrixFactorization: Classic latent factor Collaborative Filtering
3. NeuralCollaborativeFiltering (NeuMF): Combines GMF (elementwise user-item interaction)
   with a deep MLP over user embeddings and shared MultimodalTrackEncoder vectors,
   producing aligned long-term user preference representations in R^128.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models.embeddings import MultimodalTrackEncoder


class ItemCooccurrenceCF:
    """Fast sparse Item-KNN Collaborative Filtering fusing user sessions & Spotify playlists."""

    def __init__(self, num_items: int, top_k_neighbors: int = 80) -> None:
        self.num_items = num_items
        self.top_k_neighbors = top_k_neighbors
        self.sim_matrix: sp.csr_matrix | None = None
        self.item_popularity: np.ndarray = np.zeros(num_items, dtype=np.float32)

    def fit(
        self,
        user_item_pairs: tuple[np.ndarray, np.ndarray, np.ndarray],
        playlist_sequences: list[list[int]] | None = None,
    ) -> "ItemCooccurrenceCF":
        users, items, weights = user_item_pairs
        rows_list = [users]
        cols_list = [items]
        vals_list = [np.clip(weights, 0.1, 2.0)]

        next_row_id = int(users.max()) + 1 if len(users) > 0 else 0
        if playlist_sequences:
            pl_rows, pl_cols, pl_vals = [], [], []
            for offset, seq in enumerate(playlist_sequences):
                rid = next_row_id + offset
                for it in seq:
                    if 0 <= it < self.num_items:
                        pl_rows.append(rid)
                        pl_cols.append(it)
                        pl_vals.append(1.0)
            if pl_rows:
                rows_list.append(np.array(pl_rows, dtype=np.int32))
                cols_list.append(np.array(pl_cols, dtype=np.int32))
                vals_list.append(np.array(pl_vals, dtype=np.float32))
                next_row_id += len(playlist_sequences)

        all_r = np.concatenate(rows_list)
        all_c = np.concatenate(cols_list)
        all_v = np.concatenate(vals_list)

        mat = sp.csr_matrix((all_v, (all_r, all_c)), shape=(next_row_id, self.num_items), dtype=np.float32)
        # Binary/BM25-style sublinear TF scaling
        mat.data = np.log1p(mat.data)

        pop = np.asarray(mat.sum(axis=0)).ravel()
        self.item_popularity = (pop / (pop.max() + 1e-6)).astype(np.float32)

        # Normalize columns (items) to compute sparse item-item cosine similarity
        col_norms = np.sqrt(np.asarray(mat.power(2).sum(axis=0)).ravel()) + 1e-8
        normed = mat @ sp.diags(1.0 / col_norms)
        sim = (normed.T @ normed).tocsr()
        sim.setdiag(0.0)
        sim.eliminate_zeros()

        # Prune to top_k_neighbors per item to keep memory tiny & inference sub-millisecond
        indptr = [0]
        indices = []
        data = []
        for i in range(self.num_items):
            start, end = sim.indptr[i], sim.indptr[i + 1]
            row_idx = sim.indices[start:end]
            row_dat = sim.data[start:end]
            if len(row_dat) > self.top_k_neighbors:
                top_pos = np.argpartition(row_dat, -self.top_k_neighbors)[-self.top_k_neighbors:]
                row_idx = row_idx[top_pos]
                row_dat = row_dat[top_pos]
            indices.append(row_idx)
            data.append(row_dat)
            indptr.append(indptr[-1] + len(row_dat))

        self.sim_matrix = sp.csr_matrix(
            (
                np.concatenate(data) if data else np.array([], dtype=np.float32),
                np.concatenate(indices) if indices else np.array([], dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(self.num_items, self.num_items),
        )
        return self

    def score_for_history(self, pos_items: list[int], neg_items: list[int] | None = None) -> np.ndarray:
        assert self.sim_matrix is not None, "Call fit() first."
        if not pos_items:
            return self.item_popularity.copy()
        vec = np.zeros(self.num_items, dtype=np.float32)
        # Recent items carry higher weight
        recent = pos_items[-40:]
        n = len(recent)
        for idx, it in enumerate(recent):
            if 0 <= it < self.num_items:
                decay = 0.5 + 0.5 * ((idx + 1) / n)
                vec[it] += decay
        scores = np.asarray(vec @ self.sim_matrix).ravel()
        if neg_items:
            neg_vec = np.zeros(self.num_items, dtype=np.float32)
            for it in neg_items[-15:]:
                if 0 <= it < self.num_items:
                    neg_vec[it] = 0.4
            scores -= np.asarray(neg_vec @ self.sim_matrix).ravel()
        return scores


class BPRMatrixFactorization(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization baseline."""

    def __init__(self, num_users: int, num_items: int, factors: int = 64) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, factors)
        self.item_emb = nn.Embedding(num_items, factors)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        if item_ids.dim() == 2 and u.dim() == 2:
            # [B, D] x [B, K, D] -> [B, K]
            it = self.item_emb(item_ids)
            b = self.item_bias(item_ids).squeeze(-1)
            return (u.unsqueeze(1) * it).sum(dim=-1) + b
        it = self.item_emb(item_ids)
        b = self.item_bias(item_ids).squeeze(-1)
        return (u * it).sum(dim=-1) + b


class NeuralCollaborativeFiltering(nn.Module):
    """
    Hybrid Neural Collaborative Filtering (NeuMF: GMF + Multimodal MLP + Aligned Taste Head).

    Simultaneously learns:
    - Non-linear GMF + MLP interaction scoring for NeuMF collaborative filtering
    - Aligned Long-Term User Embedding u_long in R^128 inside the shared MultimodalTrackEncoder space
    """

    def __init__(
        self,
        num_users: int,
        track_encoder: MultimodalTrackEncoder,
        gmf_dim: int = 64,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.track_encoder = track_encoder
        self.embed_dim = embed_dim

        # GMF branch embeddings
        self.user_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_gmf = nn.Embedding(track_encoder.num_items + 1, gmf_dim, padding_idx=track_encoder.num_items)

        # Long-term user taste representation in R^128 (shared space with MultimodalTrackEncoder)
        self.user_long_emb = nn.Embedding(num_users, embed_dim)

        # Deep MLP interaction tower over [u_long, e_i, u_long * e_i]
        self.mlp_tower = nn.Sequential(
            nn.Linear(embed_dim * 3, 192),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(192, 64),
            nn.GELU(),
        )

        self.neu_head = nn.Linear(gmf_dim + 64, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_gmf.weight, std=0.05)
        nn.init.normal_(self.item_gmf.weight[:-1], std=0.05)
        nn.init.normal_(self.user_long_emb.weight, std=0.05)

    def get_long_term_user_emb(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_long_emb(user_ids), p=2, dim=-1)

    def score_pairs(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both NeuMF interaction logit and cosine similarity in shared R^128 space.
        Supports user_ids: [B], item_ids: [B] or [B, K].
        """
        u_gmf = self.user_gmf(user_ids)
        u_long = self.get_long_term_user_emb(user_ids)
        e_item = self.track_encoder(item_ids)
        i_gmf = self.item_gmf(item_ids)

        if item_ids.dim() == 2:
            B, K = item_ids.shape
            u_gmf_exp = u_gmf.unsqueeze(1).expand(B, K, -1)
            u_long_exp = u_long.unsqueeze(1).expand(B, K, -1)
            gmf_feat = u_gmf_exp * i_gmf
            mlp_in = torch.cat([u_long_exp, e_item, u_long_exp * e_item], dim=-1)
            mlp_feat = self.mlp_tower(mlp_in)
            neumf_logit = self.neu_head(torch.cat([gmf_feat, mlp_feat], dim=-1)).squeeze(-1)
            shared_cos = (u_long_exp * e_item).sum(dim=-1)
            return neumf_logit, shared_cos

        gmf_feat = u_gmf * i_gmf
        mlp_in = torch.cat([u_long, e_item, u_long * e_item], dim=-1)
        mlp_feat = self.mlp_tower(mlp_in)
        neumf_logit = self.neu_head(torch.cat([gmf_feat, mlp_feat], dim=-1)).squeeze(-1)
        shared_cos = (u_long * e_item).sum(dim=-1)
        return neumf_logit, shared_cos
