"""Unit & Property Tests for Adaptive Shuffle, Online Bandit Feedback, and Neural Architectures."""

from __future__ import annotations

import numpy as np
import torch

from backend.app.engine.adaptive_shuffle import AdaptiveShuffleEngine, ShuffleConfig
from backend.app.engine.online_feedback import UserFeedbackState
from backend.app.models.collaborative import NeuralCollaborativeFiltering
from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import DualHorizonSessionTransformer


def test_adaptive_shuffle_reduces_artist_clumping_and_smooths_flow() -> None:
    rng = np.random.default_rng(7)
    m = 36
    # Construct a realistic playlist where 6 artists each have 6 songs (high clump risk!)
    artist_ids = np.repeat(np.arange(6, dtype=np.int32), 6)
    genre_ids = np.tile(np.arange(6, dtype=np.int32), 6)
    embs = rng.normal(0, 1, size=(m, 128)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    raw_audio = np.column_stack(
        [
            rng.uniform(0.2, 0.9, size=m),
            rng.uniform(0.1, 0.95, size=m),
            rng.uniform(0.1, 0.95, size=m),
            rng.uniform(75.0, 165.0, size=m),
            rng.integers(0, 12, size=m),
        ]
    ).astype(np.float32)
    rel = rng.uniform(0.2, 1.0, size=m).astype(np.float32)
    is_disc = (np.arange(m) % 3 == 0)

    eng = AdaptiveShuffleEngine()

    clumps_rand, clumps_adapt = [], []
    smooth_rand, smooth_adapt = [], []
    for seed in range(20):
        r_res = eng.sequence(embs, artist_ids, genre_ids, raw_audio, rel, is_disc, ShuffleConfig(mode="fisher_yates", seed=seed))
        a_res = eng.sequence(
            embs,
            artist_ids,
            genre_ids,
            raw_audio,
            rel,
            is_disc,
            ShuffleConfig(mode="adaptive", artist_cooldown=4, flow_weight=0.6, diversity_weight=0.8, seed=seed),
        )
        assert sorted(r_res["ordered_positions"]) == list(range(m))
        assert sorted(a_res["ordered_positions"]) == list(range(m))
        clumps_rand.append(r_res["metrics"]["artist_clump_rate"])
        clumps_adapt.append(a_res["metrics"]["artist_clump_rate"])
        smooth_rand.append(r_res["metrics"]["acoustic_smoothness"])
        smooth_adapt.append(a_res["metrics"]["acoustic_smoothness"])

    assert np.mean(clumps_adapt) < np.mean(clumps_rand) * 0.35
    assert np.mean(smooth_adapt) > np.mean(smooth_rand)

    # Verify original adaptive mode is preserved identically when quota_urgency_weight=0.0
    orig_a = eng.sequence(embs, artist_ids, genre_ids, raw_audio, rel, is_disc, ShuffleConfig(mode="adaptive", seed=42))
    orig_b = eng.sequence(
        embs, artist_ids, genre_ids, raw_audio, rel, is_disc,
        ShuffleConfig(mode="adaptive", quota_urgency_weight=0.0, seed=42),
    )
    assert orig_a["ordered_positions"] == orig_b["ordered_positions"]
    assert orig_a["metrics"]["sequence_validity"] == 1.0

    # Verify experimental quota urgency produces a valid permutation
    exp_res = eng.sequence(
        embs, artist_ids, genre_ids, raw_audio, rel, is_disc,
        ShuffleConfig(mode="adaptive", quota_urgency_weight=0.75, seed=42),
    )
    assert sorted(exp_res["ordered_positions"]) == list(range(m))
    assert exp_res["metrics"]["sequence_validity"] == 1.0


def test_online_feedback_bandit_shifts_affinity_directionally() -> None:
    rng = np.random.default_rng(11)
    e_rock = rng.normal(0, 1, size=128).astype(np.float32)
    e_rock /= np.linalg.norm(e_rock)
    e_jazz = -e_rock.copy()  # opposite direction in embedding space

    fb = UserFeedbackState(user_idx=0, embed_dim=128)
    fb.record_feedback(item_idx=1, item_emb=e_rock, artist_idx=10, genre_idx=2, action="early_skip", played_ratio=0.08)
    fb.record_feedback(item_idx=2, item_emb=e_jazz, artist_idx=20, genre_idx=5, action="like", played_ratio=1.0)

    cands = np.vstack([e_rock, e_jazz])
    bonuses = fb.linucb_bonus(cands, alpha_explore=0.0)
    assert bonuses[1] > bonuses[0]
    assert np.dot(fb.delta_u, e_jazz) > 0.0
    assert np.dot(fb.delta_u, e_rock) < 0.0
    assert 10 in fb.artist_fatigue
    assert 20 in fb.artist_boost


def test_multimodal_neumf_and_transformer_shapes_and_cold_start() -> None:
    enc = MultimodalTrackEncoder(num_items=50, num_artists=10, num_genres=5, audio_dim=12, embed_dim=128)
    enc.set_catalog_metadata(
        artist_indices=torch.randint(0, 10, (50,)),
        genre_indices=torch.randint(0, 5, (50,)),
        audio_features=torch.randn(50, 12),
    )
    neumf = NeuralCollaborativeFiltering(num_users=8, track_encoder=enc, embed_dim=128)
    tr = DualHorizonSessionTransformer(track_encoder=enc, max_seq_len=12, embed_dim=128)

    u = torch.tensor([0, 1], dtype=torch.long)
    items_2d = torch.randint(0, 50, (2, 5))
    logits, cos = neumf.score_pairs(u, items_2d)
    assert logits.shape == (2, 5)
    assert cos.shape == (2, 5)

    cold_vecs = enc(torch.tensor([0, 1, 2]), cold_start=True)
    assert cold_vecs.shape == (3, 128)
    assert torch.allclose(cold_vecs.norm(dim=-1), torch.ones(3), atol=1e-4)

    seq_items = torch.randint(0, 50, (2, 12))
    seq_items[0, :4] = 50  # left padding
    seq_acts = torch.randint(0, 5, (2, 12))
    seq_acts[0, :4] = 6
    seq_tb = torch.randint(0, 6, (2, 12))
    seq_tb[0, :4] = 6

    u_short = tr.encode_short_term_sequence(seq_items, seq_acts, seq_tb)[:, -1, :]
    u_long = neumf.get_long_term_user_emb(u)
    u_fused, gate = tr.fuse_horizons(u_long, u_short)
    assert u_fused.shape == (2, 128)
    assert gate.shape == (2, 128)
    assert not torch.isnan(u_fused).any()

    short_weight = 0.75
    overridden, overridden_gate = tr.fuse_horizons(u_long, u_short, short_term_override=short_weight)
    expected = torch.nn.functional.normalize(
        (1.0 - short_weight) * u_long + short_weight * u_short,
        p=2,
        dim=-1,
    )
    assert torch.allclose(overridden, expected, atol=1e-5)
    assert torch.allclose(overridden_gate, torch.full_like(overridden_gate, 1.0 - short_weight), atol=1e-6)
