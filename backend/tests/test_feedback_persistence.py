"""Tests for SQLite Feedback Persistence, Restart Recovery, User Isolation, and CORS Config."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from backend.app.engine.feedback_store import SQLiteFeedbackStore
from backend.app.main import _parse_cors_origins


def test_feedback_persists_across_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "feedback_test.db"
    rng = np.random.default_rng(42)
    e1 = rng.normal(0, 1, size=128).astype(np.float32)
    e1 /= np.linalg.norm(e1)
    e2 = rng.normal(0, 1, size=128).astype(np.float32)
    e2 /= np.linalg.norm(e2)

    # Session 1: record feedback and persist
    store1 = SQLiteFeedbackStore(db_path)
    state1 = store1.load_user_state(user_idx=7)
    state1.record_feedback(10, e1, artist_idx=100, genre_idx=5, action="like", played_ratio=1.0)
    state1.record_feedback(20, e2, artist_idx=200, genre_idx=9, action="early_skip", played_ratio=0.12)
    store1.save_user_state(state1)
    expected_bonus = state1.linucb_bonus(np.vstack([e1, e2]), alpha_explore=0.15)

    # Simulate backend restart with a brand new store instance on the same SQLite file
    store2 = SQLiteFeedbackStore(db_path)
    recovered = store2.load_user_state(user_idx=7)

    assert recovered.liked_items == {10}
    assert recovered.skipped_items == {20}
    assert 100 in recovered.artist_boost
    assert 200 in recovered.artist_fatigue
    assert len(recovered.events) == 2
    assert np.allclose(recovered.A_diag, state1.A_diag, atol=1e-6)
    assert np.allclose(recovered.b, state1.b, atol=1e-6)
    assert np.allclose(recovered.delta_u, state1.delta_u, atol=1e-6)
    assert np.allclose(
        recovered.linucb_bonus(np.vstack([e1, e2]), alpha_explore=0.15),
        expected_bonus,
        atol=1e-6,
    )


def test_feedback_user_isolation_and_reset(tmp_path: Path) -> None:
    db_path = tmp_path / "isolation.db"
    store = SQLiteFeedbackStore(db_path)
    rng = np.random.default_rng(7)
    vec = rng.normal(0, 1, size=128).astype(np.float32)
    vec /= np.linalg.norm(vec)

    u1 = store.load_user_state(1)
    u1.record_feedback(5, vec, artist_idx=11, genre_idx=2, action="like")
    store.save_user_state(u1)

    u2 = store.load_user_state(2)
    u2.record_feedback(9, vec, artist_idx=22, genre_idx=4, action="dislike")
    store.save_user_state(u2)

    assert store.list_persisted_users() == [1, 2]
    assert store.load_user_state(1).liked_items == {5}
    assert store.load_user_state(2).disliked_items == {9}
    assert store.load_user_state(2).liked_items == set()

    # Resetting user 1 must not touch user 2
    store.reset_user_state(1)
    assert store.list_persisted_users() == [2]
    assert store.load_user_state(1).liked_items == set()
    assert store.load_user_state(2).disliked_items == {9}


def test_concurrent_feedback_updates_are_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    store = SQLiteFeedbackStore(db_path)
    rng = np.random.default_rng(99)
    vec = rng.normal(0, 1, size=128).astype(np.float32)
    vec /= np.linalg.norm(vec)
    errors: list[Exception] = []

    def worker(uid: int) -> None:
        try:
            for step in range(8):
                st = store.load_user_state(uid)
                st.record_feedback(step, vec, artist_idx=uid * 10 + step, genre_idx=step % 4, action="complete_play")
                store.save_user_state(st)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(u,)) for u in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    for uid in range(4):
        assert len(store.load_user_state(uid).events) == 8


def test_cors_origins_env_config(monkeypatch) -> None:
    monkeypatch.delenv("RESONANCE_CORS_ORIGINS", raising=False)
    assert _parse_cors_origins() == ["http://localhost:3000", "http://127.0.0.1:3000"]

    monkeypatch.setenv("RESONANCE_CORS_ORIGINS", "https://resonance.example.com/, http://localhost:3000 ")
    assert _parse_cors_origins() == ["https://resonance.example.com", "http://localhost:3000"]
