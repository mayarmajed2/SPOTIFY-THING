"""Unit & Integrity Tests for Data Pipeline v2 (Entity Resolution, Provenance, Sessions, Schema)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backend.data_pipeline.build_verified_dataset import (
    extract_version_tag,
    normalize_exact,
    normalize_packaging_stripped,
    sessionize_timestamps,
)


def test_version_aware_entity_resolution_preserves_distinct_recordings() -> None:
    # Live, remix, and acoustic versions must NOT collapse to 'studio'
    assert extract_version_tag("Hotel California") == "studio"
    assert extract_version_tag("Hotel California (Live)") == "live"
    assert extract_version_tag("Hotel California - Acoustic Version") == "acoustic"
    assert extract_version_tag("Blinding Lights (Chromatics Remix)") == "remix"
    assert extract_version_tag("Creep (2009 Remaster)") == "studio"

    # Packaging suffixes are stripped only while keeping core title intact
    assert normalize_packaging_stripped("Bohemian Rhapsody - Remastered 2011") == "bohemian rhapsody"
    assert normalize_exact("Bohemian Rhapsody (Live)") != normalize_exact("Bohemian Rhapsody")


def test_sessionize_timestamps_respects_configurable_gap() -> None:
    df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 0, 1, 1],
            "event_timestamp": [1000, 1200, 4000, 4150, 500, 5000],
        }
    )
    # With 1800s gap: user 0 has 2 sessions ([1000,1200], [4000,4150]); user 1 has 2 sessions
    s_1800 = sessionize_timestamps(df, gap_seconds=1800).tolist()
    assert s_1800 == [1, 1, 2, 2, 1, 2]

    # With 3600s gap: user 0 stays in 1 session (2800s gap < 3600s); user 1 still splits (4500s > 3600s)
    s_3600 = sessionize_timestamps(df, gap_seconds=3600).tolist()
    assert s_3600 == [1, 1, 1, 1, 1, 2]


@pytest.mark.requires_dataset
def test_processed_v2_schema_and_provenance_integrity() -> None:
    out_dir = Path("data/processed_v2")
    if not (out_dir / "tracks_v2.parquet").exists():
        pytest.skip("data/processed_v2 not built yet")

    tracks = pd.read_parquet(out_dir / "tracks_v2.parquet")
    inter = pd.read_parquet(out_dir / "interactions_v2.parquet")
    pl = pd.read_parquet(out_dir / "playlists_v2.parquet")

    # Required provenance columns on tracks
    for col in ["audio_feature_source", "audio_feature_type", "audio_feature_confidence", "is_synthetic"]:
        assert col in tracks.columns
    assert not tracks["spotify_track_id"].duplicated().any()
    assert (tracks.loc[~tracks["is_synthetic"], "audio_feature_confidence"] == 1.0).all()
    assert (tracks.loc[tracks["is_synthetic"], "audio_feature_confidence"] <= 0.50).all()
    # Verify NO pre-fitted full-catalog z-score columns exist in tracks_v2
    assert not any(c.startswith("norm_") for c in tracks.columns)

    # Required observed-only schema on interactions (NO manufactured duration/skips/likes!)
    assert "played_ratio" not in inter.columns
    assert "action_id" not in inter.columns
    assert set(inter["event_type"].unique()).issubset({"track_listened_event", "repeated_track_event"})
    assert inter["is_observed"].all()
    assert (~inter["is_inferred"]).all()
    assert (inter["match_confidence"] >= 0.85).all()

    # Verify causal ordering and no exact duplicate timestamps per (user, item, ts)
    assert not inter.duplicated(subset=["user_idx", "item_idx", "event_timestamp"]).any()

    # Verify strict non-overlapping playlist splits
    assert set(pl["playlist_split"].unique()) == {"train", "val", "test"}
    tr_pids = set(pl.loc[pl["playlist_split"] == "train", "pid"])
    va_pids = set(pl.loc[pl["playlist_split"] == "val", "pid"])
    te_pids = set(pl.loc[pl["playlist_split"] == "test", "pid"])
    assert tr_pids.isdisjoint(va_pids)
    assert tr_pids.isdisjoint(te_pids)
    assert va_pids.isdisjoint(te_pids)
