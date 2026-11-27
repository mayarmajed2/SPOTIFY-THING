"""Tests for Positive-Excluded, Duplicate-Free Negative Sampling & Shared Eval Candidates."""

from __future__ import annotations

import numpy as np

from backend.training.splits_and_sampling import (
    build_shared_eval_candidates,
    sample_unique_negatives,
)


def test_sample_unique_negatives_excludes_positives_and_duplicates() -> None:
    rng = np.random.default_rng(42)
    excluded = {0, 1, 2, 5, 10, 15, 20}
    negs = sample_unique_negatives(num_items=100, excluded=excluded, k=30, rng=rng)
    assert len(negs) == 30
    assert len(set(negs.tolist())) == 30
    assert set(negs.tolist()).isdisjoint(excluded)


def test_shared_eval_candidates_are_deterministic_and_duplicate_free() -> None:
    eval_cases = [
        {"user_idx": 0, "target": 12, "seen_pos": {1, 2, 3, 4, 5}},
        {"user_idx": 1, "target": 88, "seen_pos": {10, 20, 30, 88}},
    ]
    c1 = build_shared_eval_candidates(eval_cases, num_items=200, num_negatives=50, seed=123)
    c2 = build_shared_eval_candidates(eval_cases, num_items=200, num_negatives=50, seed=123)

    assert np.array_equal(c1, c2)
    assert c1.shape == (2, 51)
    for row_idx, case in enumerate(eval_cases):
        row = c1[row_idx].tolist()
        assert row[0] == case["target"]
        negs = row[1:]
        assert len(set(negs)) == 50
        assert case["target"] not in negs
        assert set(negs).isdisjoint(case["seen_pos"])
