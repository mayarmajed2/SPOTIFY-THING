"""
Shared Multimodal Track Representation Space (R^128).

Combines collaborative track ID embeddings with 12D standardized Spotify audio
features, genre embeddings, and artist embeddings. Trained with modality dropout
(dropping item IDs 15% of the time) so cold-start / discovery tracks align seamlessly
with the collaborative embedding space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalTrackEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_artists: int,
        num_genres: int,
        audio_dim: int = 12,
        embed_dim: int = 128,
        id_dropout_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.id_dropout_prob = id_dropout_prob

        # Pad index = num_items (used by sequential transformer)
        self.item_emb = nn.Embedding(num_items + 1, 64, padding_idx=num_items)
        self.artist_emb = nn.Embedding(num_artists + 1, 32, padding_idx=num_artists)
        self.genre_emb = nn.Embedding(num_genres + 1, 16, padding_idx=num_genres)

        self.audio_mlp = nn.Sequential(
            nn.Linear(audio_dim, 48),
            nn.GELU(),
            nn.Linear(48, 32),
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(64 + 32 + 16 + 32, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Register buffers for fast index-only lookup during training & serving
        self.register_buffer("item_to_artist", torch.zeros(num_items + 1, dtype=torch.long))
        self.register_buffer("item_to_genre", torch.zeros(num_items + 1, dtype=torch.long))
        self.register_buffer("item_audio_feats", torch.zeros(num_items + 1, audio_dim, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight[:-1], mean=0.0, std=0.05)
        nn.init.normal_(self.artist_emb.weight[:-1], mean=0.0, std=0.05)
        nn.init.normal_(self.genre_emb.weight[:-1], mean=0.0, std=0.05)

    def set_catalog_metadata(
        self,
        artist_indices: torch.Tensor,
        genre_indices: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> None:
        """Populate static item -> (artist, genre, audio) lookup buffers."""
        n = artist_indices.shape[0]
        assert n == self.num_items, f"Expected {self.num_items} items, got {n}"
        self.item_to_artist[:n] = artist_indices.long()
        self.item_to_artist[n] = self.artist_emb.padding_idx
        self.item_to_genre[:n] = genre_indices.long()
        self.item_to_genre[n] = self.genre_emb.padding_idx
        self.item_audio_feats[:n] = audio_features.float()
        self.item_audio_feats[n] = 0.0

    def forward(self, item_ids: torch.Tensor, cold_start: bool = False) -> torch.Tensor:
        """Encode arbitrary tensor of item_ids [...] -> [..., embed_dim] L2-normalized."""
        pad_mask = item_ids == self.num_items
        id_vec = self.item_emb(item_ids)

        if cold_start:
            id_vec = torch.zeros_like(id_vec)
        elif self.training and self.id_dropout_prob > 0.0:
            drop_mask = (torch.rand(item_ids.shape, device=item_ids.device) < self.id_dropout_prob).unsqueeze(-1)
            id_vec = id_vec.masked_fill(drop_mask, 0.0)

        artist_vec = self.artist_emb(self.item_to_artist[item_ids])
        genre_vec = self.genre_emb(self.item_to_genre[item_ids])
        audio_vec = self.audio_mlp(self.item_audio_feats[item_ids])

        concat = torch.cat([id_vec, artist_vec, genre_vec, audio_vec], dim=-1)
        out = self.fusion_proj(concat)
        out = F.normalize(out, p=2, dim=-1)
        if pad_mask.any():
            out = out.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return out

    @torch.no_grad()
    def export_all_embeddings(self, batch_size: int = 4096, device: torch.device | str = "cpu") -> torch.Tensor:
        """Export [num_items, embed_dim] matrix for exact vector indexing."""
        self.eval()
        embs = []
        for start in range(0, self.num_items, batch_size):
            end = min(start + batch_size, self.num_items)
            ids = torch.arange(start, end, device=device, dtype=torch.long)
            embs.append(self.forward(ids).cpu())
        return torch.cat(embs, dim=0)
