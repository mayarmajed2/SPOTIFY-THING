"""Behavioral Stress Tests for Online Feedback Bandit & Fatigue Memory (Phase E)."""

from __future__ import annotations

import numpy as np

from backend.app.engine.online_feedback import UserFeedbackState


def test_feedback_no_overreaction_and_fatigue_decay() -> None:
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, size=128).astype(np.float32)
    e /= np.linalg.norm(e)
    e_other = rng.normal(0, 1, size=128).astype(np.float32)
    e_other /= np.linalg.norm(e_other)

    fb = UserFeedbackState(user_idx=0, embed_dim=128)
    # 1 early skip should create modest residual (< 0.30), not saturate cap (0.65)
    fb.record_feedback(1, e, artist_idx=10, genre_idx=2, action="early_skip", played_ratio=0.1)
    norm_after_1 = float(np.linalg.norm(fb.delta_u))
    assert 0.05 < norm_after_1 < 0.30
    init_fatigue = fb.genre_fatigue[2]

    # Subsequent neutral plays decay fatigue smoothly
    for k in range(6):
        fb.record_feedback(100 + k, e_other, artist_idx=50 + k, genre_idx=9, action="complete_play", played_ratio=0.95)
    assert fb.genre_fatigue.get(2, 0.0) < init_fatigue * 0.65


def test_contradictory_feedback_and_preference_reversal() -> None:
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, size=128).astype(np.float32)
    e /= np.linalg.norm(e)

    fb = UserFeedbackState(user_idx=1, embed_dim=128)
    fb.record_feedback(5, e, artist_idx=7, genre_idx=3, action="like", played_ratio=1.0)
    assert 5 in fb.liked_items and 5 not in fb.disliked_items

    # Explicit dislike reverses membership cleanly
    fb.record_feedback(5, e, artist_idx=7, genre_idx=3, action="dislike", played_ratio=0.1)
    assert 5 in fb.disliked_items and 5 not in fb.liked_items
    assert 7 in fb.artist_fatigue
