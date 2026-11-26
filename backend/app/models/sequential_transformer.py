"""
Dual-Horizon Causal Session Transformer (Short-Term + Long-Term Preference Fusion).

Captures:
- Sequential listening trajectories (up to L=50 recent interactions)
- Implicit/Explicit action types (early_skip, partial_play, complete_play, repeat_listen, favorite_listen)
- Log-bucketed inter-event time deltas (in-session flow vs. cross-session shift)
- Gated fusion between stable Long-Term User Taste (u_long) and Short-Term Session Context (u_short)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.models.embeddings import MultimodalTrackEncoder

# Action IDs: 0=early_skip, 1=partial, 2=complete, 3=repeat, 4=favorite, 5=dislike_explicit, 6=pad
PAD_ACTION_ID = 6
# Time delta buckets: 0(<1m), 1(1-5m), 2(5-30m), 3(30m-6h), 4(6h-48h), 5(>48h), 6(pad)
PAD_TIME_BUCKET = 6


def bucketize_time_deltas(delta_seconds: torch.Tensor) -> torch.Tensor:
    """Map elapsed seconds between consecutive plays into discrete temporal context buckets."""
    buckets = torch.full_like(delta_seconds, 5, dtype=torch.long)
    buckets = torch.where(delta_seconds < 172800, 4, buckets)
    buckets = torch.where(delta_seconds < 21600, 3, buckets)
    buckets = torch.where(delta_seconds < 1800, 2, buckets)
    buckets = torch.where(delta_seconds < 300, 1, buckets)
    buckets = torch.where(delta_seconds < 60, 0, buckets)
    return buckets


class DualHorizonSessionTransformer(nn.Module):
    def __init__(
        self,
        track_encoder: MultimodalTrackEncoder,
        max_seq_len: int = 50,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
        temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.track_encoder = track_encoder
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.pad_item_id = track_encoder.num_items
        self.temperature = temperature

        self.action_emb = nn.Embedding(7, embed_dim, padding_idx=PAD_ACTION_ID)
        self.time_emb = nn.Embedding(7, embed_dim, padding_idx=PAD_TIME_BUCKET)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.short_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Gated fusion combining Long-Term user profile u_long and Short-Term Transformer context u_short
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.action_emb.weight[:-1], std=0.03)
        nn.init.normal_(self.time_emb.weight[:-1], std=0.03)
        nn.init.normal_(self.pos_emb.weight, std=0.03)

    def encode_short_term_sequence(
        self,
        item_seq: torch.Tensor,
        action_seq: torch.Tensor,
        time_bucket_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode causal sequence states.
        Inputs: [B, L] left-padded or right-padded sequences (pad_item_id = num_items).
        Returns: [B, L, D] L2-normalized causal short-term representations at every step.
        """
        B, L = item_seq.shape
        device = item_seq.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        track_vecs = self.track_encoder(item_seq)
        act_vecs = self.action_emb(action_seq)
        time_vecs = self.time_emb(time_bucket_seq)
        pos_vecs = self.pos_emb(positions)

        x = self.input_drop(self.input_norm(track_vecs + act_vecs + time_vecs + pos_vecs))

        causal_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=device), diagonal=1)
        pad_mask = item_seq == self.pad_item_id

        # Prevent NaN if an entire row were padding
        all_pad = pad_mask.all(dim=1)
        if all_pad.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad, -1] = False

        h = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)
        u_short = F.normalize(self.short_proj(h), p=2, dim=-1)
        return u_short

    def fuse_horizons(
        self,
        u_long: torch.Tensor,
        u_short: torch.Tensor,
        short_term_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse u_long ([B, D] or [B, L, D]) with u_short ([B, D] or [B, L, D]).
        Returns (u_fused, gate_long_weight).
        """
        if u_long.dim() == 2 and u_short.dim() == 3:
            u_long = u_long.unsqueeze(1).expand_as(u_short)

        gate_in = torch.cat([u_long, u_short, u_long * u_short], dim=-1)
        gate = self.fusion_gate(gate_in)  # weight on u_long
        if short_term_override is not None:
            # Allow user/UI to steer balance between stable taste and immediate session mood
            alpha = float(max(0.0, min(1.0, short_term_override)))
            u_fused = F.normalize((1.0 - alpha) * u_long + alpha * u_short, p=2, dim=-1)
            return u_fused, torch.full_like(gate, 1.0 - alpha)

        u_fused = F.normalize(gate * u_long + (1.0 - gate) * u_short, p=2, dim=-1)
        return u_fused, gate
