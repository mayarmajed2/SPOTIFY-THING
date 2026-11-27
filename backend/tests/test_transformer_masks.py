"""Causal Masking & Padding Invariance Tests for DualHorizonSessionTransformer."""

from __future__ import annotations

import torch

from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import (
    PAD_ACTION_ID,
    PAD_TIME_BUCKET,
    DualHorizonSessionTransformer,
)


def test_transformer_causal_mask_blocks_future_information() -> None:
    torch.manual_seed(0)
    num_items = 40
    enc = MultimodalTrackEncoder(num_items=num_items, num_artists=8, num_genres=4, audio_dim=12, embed_dim=64)
    enc.set_catalog_metadata(
        artist_indices=torch.randint(0, 8, (num_items,)),
        genre_indices=torch.randint(0, 4, (num_items,)),
        audio_features=torch.randn(num_items, 12),
    )
    tr = DualHorizonSessionTransformer(track_encoder=enc, max_seq_len=10, embed_dim=64, num_heads=4, num_layers=2, dropout=0.0)
    tr.eval()

    seq_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    seq_b = seq_a.clone()
    # Mutate future tokens at positions 6..9
    seq_b[0, 6:] = torch.tensor([31, 32, 33, 34], dtype=torch.long)

    acts = torch.full((1, 10), 2, dtype=torch.long)
    tbs = torch.full((1, 10), 1, dtype=torch.long)

    with torch.no_grad():
        out_a = tr.encode_short_term_sequence(seq_a, acts, tbs)
        out_b = tr.encode_short_term_sequence(seq_b, acts, tbs)

    # Positions 0..5 MUST be strictly identical (zero future leakage across causal mask!)
    assert torch.allclose(out_a[:, :6, :], out_b[:, :6, :], atol=1e-6)
    # Position 9 MUST differ because token 9 changed
    assert not torch.allclose(out_a[:, 9, :], out_b[:, 9, :], atol=1e-3)


def test_transformer_handles_left_padding_and_all_padding_without_nan() -> None:
    torch.manual_seed(1)
    num_items = 30
    enc = MultimodalTrackEncoder(num_items=num_items, num_artists=5, num_genres=3, audio_dim=12, embed_dim=64)
    enc.set_catalog_metadata(
        artist_indices=torch.randint(0, 5, (num_items,)),
        genre_indices=torch.randint(0, 3, (num_items,)),
        audio_features=torch.randn(num_items, 12),
    )
    tr = DualHorizonSessionTransformer(track_encoder=enc, max_seq_len=8, embed_dim=64, num_heads=4, num_layers=2, dropout=0.0)
    tr.eval()

    items = torch.tensor(
        [
            [num_items, num_items, num_items, 1, 2, 3, 4, 5],
            [num_items] * 8,  # edge case: completely padded row
        ],
        dtype=torch.long,
    )
    acts = torch.tensor(
        [
            [PAD_ACTION_ID, PAD_ACTION_ID, PAD_ACTION_ID, 2, 2, 3, 2, 2],
            [PAD_ACTION_ID] * 8,
        ],
        dtype=torch.long,
    )
    tbs = torch.tensor(
        [
            [PAD_TIME_BUCKET, PAD_TIME_BUCKET, PAD_TIME_BUCKET, 1, 1, 2, 1, 1],
            [PAD_TIME_BUCKET] * 8,
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        out = tr.encode_short_term_sequence(items, acts, tbs)
    assert out.shape == (2, 8, 64)
    assert not torch.isnan(out).any()
