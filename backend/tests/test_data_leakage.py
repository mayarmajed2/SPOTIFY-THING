"""Automated Data Leakage Prevention Tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.data_pipeline.build_verified_dataset import AUDIO_FEATURE_COLS
from backend.training.splits_and_sampling import (
    build_zero_leakage_splits,
    compute_training_only_novelty,
    fit_transform_audio_features,
)


def _make_synthetic_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    n_items = 60
    tracks = pd.DataFrame(
        {
            "item_idx": np.arange(n_items, dtype=np.int32),
            "artist_idx": np.arange(n_items, dtype=np.int32) % 10,
            "genre_idx": np.arange(n_items, dtype=np.int32) % 5,
            "is_synthetic": [i >= 45 for i in range(n_items)],
            **{col: rng.uniform(0.0, 1.0, size=n_items).astype(np.float32) for col in AUDIO_FEATURE_COLS},
        }
    )

    inter_rows = []
    for u in range(8):
        for step in range(15):
            # Items 0..39 in train/val; item 55 ONLY appears as final test target!
            it = (u * 3 + step) % 40 if step < 14 else 55
            inter_rows.append(
                {
                    "user_idx": u,
                    "item_idx": it,
                    "event_timestamp": 10000 + step * 500,
                    "event_type": "track_listened_event",
                }
            )
    interactions = pd.DataFrame(inter_rows)

    pl_rows = []
    for pid in range(20):
        split = "train" if pid < 14 else ("val" if pid < 17 else "test")
        pl_rows.append(
            {
                "pid": pid,
                "name": f"PL {pid}",
                "playlist_split": split,
                "num_tracks": 12,
                "item_indices": [(pid + k) % 40 for k in range(12)],
            }
        )
    playlists = pd.DataFrame(pl_rows)
    return tracks, interactions, playlists


def test_playlist_splits_are_strictly_disjoint() -> None:
    tracks, inter, playlists = _make_synthetic_dataset()
    splits = build_zero_leakage_splits(tracks, inter, playlists)
    tr = set(splits["train_playlists"]["pid"])
    va = set(splits["val_playlists"]["pid"])
    te = set(splits["test_playlists"]["pid"])
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_chronological_splits_prevent_future_context_leakage() -> None:
    tracks, inter, playlists = _make_synthetic_dataset()
    splits = build_zero_leakage_splits(tracks, inter, playlists)

    for case in splits["val_cases"] + splits["test_cases"]:
        assert case["max_hist_ts"] <= case["target_ts"]
        # Held-out test-only item 55 must NEVER appear in train_items
    assert 55 not in set(splits["train_items"].tolist())


def test_audio_scaler_ignores_test_only_and_synthetic_tracks() -> None:
    tracks, inter, playlists = _make_synthetic_dataset()
    train_set = set(range(40))
    _, stats_before = fit_transform_audio_features(tracks, train_set)

    # Perturb both a synthetic track (idx 50) and a test-only track (idx 42) massively
    tracks_perturbed = tracks.copy()
    tracks_perturbed.loc[tracks_perturbed["item_idx"] == 50, "energy"] = 9999.0
    tracks_perturbed.loc[tracks_perturbed["item_idx"] == 42, "energy"] = -9999.0
    _, stats_after = fit_transform_audio_features(tracks_perturbed, train_set)

    assert stats_before["energy"] == stats_after["energy"]


def test_training_popularity_ignores_future_test_interactions() -> None:
    tracks, inter, playlists = _make_synthetic_dataset()
    splits = build_zero_leakage_splits(tracks, inter, playlists)
    pop_before, nov_before = splits["train_popularity"], splits["train_novelty"]

    # Item 55 only appeared in test targets -> must have 0 training popularity!
    assert pop_before[55] == 0.0
    # Recomputing with identical train items is invariant to test set
    pop_check, nov_check = compute_training_only_novelty(
        len(tracks),
        splits["train_items"],
        [list(s) for s in splits["train_playlists"]["item_indices"]],
    )
    assert np.allclose(pop_check[55], 0.0)
    assert nov_before[55] == nov_before.max()
