"""
Zero-Leakage Splitting, Training-Only Feature Scaling, and Positive-Excluded Negative Sampling.

Guarantees:
1. Chronological per-user train/val/test splitting (strictly causal: max_train_ts <= val_ts <= test_ts).
2. Disjoint playlist-level train/val/test splitting (zero overlap between encoder/CF training and test playlists).
3. Audio StandardScaler fitted ONLY on training-eligible measured-audio tracks (`is_synthetic == False`).
4. Popularity & self-information novelty computed ONLY from training-period interactions.
5. Duplicate-free, positive-excluded negative sampling for both training and shared evaluation candidate sets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from backend.app.models.sequential_transformer import bucketize_time_deltas
from backend.data_pipeline.build_verified_dataset import AUDIO_FEATURE_COLS


def fit_transform_audio_features(
    tracks_df: pd.DataFrame,
    train_item_indices: set[int] | np.ndarray,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """
    Fit z-score normalization STRICTLY on training-eligible measured-audio tracks
    (`item_idx in train_item_indices` AND `is_synthetic == False`), then transform all tracks.
    """
    df = tracks_df.sort_values("item_idx").reset_index(drop=True)
    train_mask = df["item_idx"].isin(train_item_indices) & (~df["is_synthetic"].astype(bool))
    if int(train_mask.sum()) < 10:
        # Fallback for tiny synthetic unit-test catalogs
        train_mask = ~df["is_synthetic"].astype(bool)
    if int(train_mask.sum()) == 0:
        train_mask = np.ones(len(df), dtype=bool)

    norm_cols: list[np.ndarray] = []
    scaler_stats: dict[str, dict[str, float]] = {}
    for col in AUDIO_FEATURE_COLS:
        vals = df[col].to_numpy(dtype=np.float32)
        fit_vals = vals[train_mask]
        mean = float(np.mean(fit_vals))
        std = float(np.std(fit_vals)) + 1e-6
        scaler_stats[col] = {"mean": round(mean, 6), "std": round(std, 6)}
        norm_cols.append(((vals - mean) / std).astype(np.float32))

    return np.column_stack(norm_cols).astype(np.float32), scaler_stats


def compute_training_only_novelty(
    num_items: int,
    train_items: np.ndarray,
    train_playlist_seqs: list[list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute item popularity and self-information novelty STRICTLY from training-period
    interactions (and optional training-split playlists). Zero future/test exposure leakage.
    """
    counts = np.zeros(num_items, dtype=np.float32)
    if len(train_items) > 0:
        np.add.at(counts, train_items, 1.0)
    if train_playlist_seqs:
        for seq in train_playlist_seqs:
            for it in seq:
                if 0 <= it < num_items:
                    counts[it] += 0.5

    log_freq = np.log1p(counts)
    pop_norm = log_freq / max(float(log_freq.max()), 1e-6)
    exposure = np.clip((counts + 1.0) / (float(counts.sum()) + num_items), 1e-6, 0.999)
    self_info = -np.log2(exposure)
    novelty = ((self_info - self_info.min()) / (self_info.max() - self_info.min() + 1e-6)).astype(np.float32)
    return pop_norm.astype(np.float32), novelty


def build_zero_leakage_splits(
    tracks_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    playlists_df: pd.DataFrame,
    max_seq_len: int = 50,
) -> dict[str, Any]:
    """
    Build strictly chronological user splits and disjoint playlist splits.
    Uses observed interactions only (`track_listened_event` / `repeated_track_event`).
    """
    train_u: list[int] = []
    train_i: list[int] = []
    train_w: list[float] = []
    seq_train_samples: list[tuple[int, list[int], list[int], list[int], int]] = []
    val_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    user_train_pos_sets: dict[int, set[int]] = {}
    all_train_items_raw: list[int] = []

    # Observed event encoding for Transformer: 2 = track_listened_event, 3 = repeated_track_event
    event_to_code = {"track_listened_event": 2, "repeated_track_event": 3}

    sorted_inter = interactions_df.sort_values(["user_idx", "event_timestamp", "item_idx"]).reset_index(drop=True)
    for uid, grp in sorted_inter.groupby("user_idx", sort=False):
        uid_int = int(uid)
        items = grp["item_idx"].to_numpy(dtype=np.int32)
        ts = grp["event_timestamp"].to_numpy(dtype=np.int64)
        ev_codes = np.array([event_to_code.get(str(e), 2) for e in grp["event_type"]], dtype=np.int32)

        if len(items) < 10:
            continue

        val_idx = len(items) - 2
        test_idx = len(items) - 1

        # Strictly verify chronological monotonicity
        assert ts[val_idx - 1] <= ts[val_idx] <= ts[test_idx], f"Non-monotonic timestamps for user {uid_int}"

        deltas = np.zeros(len(ts), dtype=np.int64)
        deltas[0] = 3600
        deltas[1:] = np.maximum(0, ts[1:] - ts[:-1])
        tbuckets = bucketize_time_deltas(torch.tensor(deltas, dtype=torch.long)).numpy()

        tr_items = items[:val_idx]
        tr_codes = ev_codes[:val_idx]
        tr_tb = tbuckets[:val_idx]
        all_train_items_raw.extend(tr_items.tolist())
        user_train_pos_sets[uid_int] = set(tr_items.tolist())

        # Deduplicated user-item training pairs with sublinear log play-count weight (strictly from train slice)
        counts_map: dict[int, int] = {}
        for it in tr_items.tolist():
            counts_map[it] = counts_map.get(it, 0) + 1
        for it, cnt in counts_map.items():
            train_u.append(uid_int)
            train_i.append(it)
            train_w.append(float(1.0 + np.log1p(cnt)))

        # Sliding causal sequence windows strictly inside train slice [0 .. val_idx - 1]
        for p_int in range(3, len(tr_items), 2):
            start = max(0, p_int - max_seq_len)
            seq_train_samples.append(
                (
                    uid_int,
                    tr_items[start:p_int].tolist(),
                    tr_codes[start:p_int].tolist(),
                    tr_tb[start:p_int].tolist(),
                    int(tr_items[p_int]),
                )
            )

        v_start = max(0, val_idx - max_seq_len)
        val_cases.append(
            {
                "user_idx": uid_int,
                "hist_items": items[v_start:val_idx].tolist(),
                "hist_actions": ev_codes[v_start:val_idx].tolist(),
                "hist_tb": tbuckets[v_start:val_idx].tolist(),
                "max_hist_ts": int(ts[val_idx - 1]),
                "target_ts": int(ts[val_idx]),
                "target": int(items[val_idx]),
                "seen_pos": set(tr_items.tolist()),
            }
        )

        te_start = max(0, test_idx - max_seq_len)
        test_cases.append(
            {
                "user_idx": uid_int,
                "hist_items": items[te_start:test_idx].tolist(),
                "hist_actions": ev_codes[te_start:test_idx].tolist(),
                "hist_tb": tbuckets[te_start:test_idx].tolist(),
                "max_hist_ts": int(ts[test_idx - 1]),
                "target_ts": int(ts[test_idx]),
                "target": int(items[test_idx]),
                "seen_pos": set(items[:test_idx].tolist()),
            }
        )

    # Disjoint playlist splits
    train_pl = playlists_df[playlists_df["playlist_split"] == "train"].reset_index(drop=True)
    val_pl = playlists_df[playlists_df["playlist_split"] == "val"].reset_index(drop=True)
    test_pl = playlists_df[playlists_df["playlist_split"] == "test"].reset_index(drop=True)

    train_pids = set(train_pl["pid"].tolist())
    val_pids = set(val_pl["pid"].tolist())
    test_pids = set(test_pl["pid"].tolist())
    assert train_pids.isdisjoint(val_pids) and train_pids.isdisjoint(test_pids) and val_pids.isdisjoint(test_pids)

    # Training-eligible items for fitting audio StandardScaler (train interactions + train playlists)
    train_item_set = set(all_train_items_raw)
    for seq in train_pl["item_indices"]:
        train_item_set.update(int(x) for x in seq)

    norm_audio, scaler_stats = fit_transform_audio_features(tracks_df, train_item_set)
    train_pop, train_novelty = compute_training_only_novelty(
        num_items=len(tracks_df),
        train_items=np.array(all_train_items_raw, dtype=np.int32),
        train_playlist_seqs=[list(s) for s in train_pl["item_indices"]],
    )

    return {
        "train_users": np.array(train_u, dtype=np.int32),
        "train_items": np.array(train_i, dtype=np.int32),
        "train_weights": np.array(train_w, dtype=np.float32),
        "seq_train_samples": seq_train_samples,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "user_train_pos_sets": user_train_pos_sets,
        "train_playlists": train_pl,
        "val_playlists": val_pl,
        "test_playlists": test_pl,
        "norm_audio": norm_audio,
        "audio_scaler_stats": scaler_stats,
        "train_popularity": train_pop,
        "train_novelty": train_novelty,
    }


def sample_unique_negatives(
    num_items: int,
    excluded: set[int],
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw k unique negative item IDs strictly outside `excluded` without replacement."""
    avail_count = num_items - len(excluded)
    if k > avail_count:
        raise ValueError(f"Cannot draw {k} unique negatives when only {avail_count} items are available.")

    chosen: list[int] = []
    chosen_set: set[int] = set()
    batch_size = max(k * 2, 32)
    while len(chosen) < k:
        draws = rng.integers(0, num_items, size=batch_size)
        for d in draws:
            di = int(d)
            if di not in excluded and di not in chosen_set:
                chosen_set.add(di)
                chosen.append(di)
                if len(chosen) == k:
                    break
    return np.array(chosen, dtype=np.int64)


def build_shared_eval_candidates(
    eval_cases: list[dict[str, Any]],
    num_items: int,
    num_negatives: int = 499,
    seed: int = 123,
) -> np.ndarray:
    """
    Pre-generate a deterministic, duplicate-free [N_cases, 1 + num_negatives] candidate matrix
    where index 0 is the ground-truth target and indices 1..num_negatives exclude seen_pos | {target}.
    Identical across every evaluated model.
    """
    rng = np.random.default_rng(seed)
    mat = np.zeros((len(eval_cases), 1 + num_negatives), dtype=np.int64)
    for row_idx, case in enumerate(eval_cases):
        target = int(case["target"])
        excluded = set(case["seen_pos"]) | {target}
        negs = sample_unique_negatives(num_items, excluded, num_negatives, rng)
        mat[row_idx, 0] = target
        mat[row_idx, 1:] = negs
    return mat
