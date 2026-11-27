"""
Scientifically Defensible Data Pipeline v2 (Zero-Leakage, Provenance-Aware).

Writes to `data/processed_v2/` without modifying `data/raw/` or legacy `data/processed/`.

Key Corrections over Legacy `download_and_build.py`:
1. Portable PyArrow/Pandas ETL (no native DuckDB DLL dependency).
2. Version-aware entity resolution preserving live/remix/acoustic/demo/instrumental tags
   with explicit match tiers (`exact_normalized`, `packaging_stripped_high`, `ambiguous`, `unmatched`).
3. Explicit audio feature provenance (`audio_feature_source`, `audio_feature_type`,
   `audio_feature_confidence`, `is_synthetic`) and NO Gaussian noise injection.
4. No full-catalog z-score normalization or full-timeline popularity leakage in ETL.
5. Honest observed-only interaction schema (`track_listened_event`, strictly causal
   `repeated_track_event`) — zero manufactured play durations, skips, or explicit likes.
6. Configurable sessionization with built-in threshold sensitivity analysis (15m/30m/60m/120m)
   and deterministic 70/15/15 playlist-level train/val/test split.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AUDIO_FEATURE_COLS = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "loudness",
    "key",
    "mode",
    "explicit_num",
]

# Semantic version qualifiers that distinguish distinct musical recordings (MUST NOT be silently merged)
VERSION_QUALIFIER_RE = re.compile(
    r"\b(live|remix|acoustic|unplugged|demo|instrumental|karaoke|a\s*cappella|orchestral|dub|club\s*mix)\b",
    re.IGNORECASE,
)

# Packaging/release suffixes safe to strip for high-confidence fallback matching
PACKAGING_SUFFIX_RE = re.compile(
    r"\s*[\(\[\-]\s*(feat\.?|ft\.?|remaster(?:ed)?|deluxe|bonus\s*track|explicit|clean|album\s*version|single\s*version|\d{4}\s*remaster).*$",
    re.IGNORECASE,
)


def extract_version_tag(track_name: Any) -> str:
    """Extract canonical semantic version tag ('studio' if none) to prevent live/remix collisions."""
    if not isinstance(track_name, str):
        return "studio"
    matches = sorted({m.group(1).lower().replace(" ", "_") for m in VERSION_QUALIFIER_RE.finditer(track_name)})
    return "+".join(matches) if matches else "studio"


def normalize_exact(text: Any) -> str:
    """Strict alphanumeric lowercase normalization preserving all words."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def normalize_packaging_stripped(text: Any) -> str:
    """Strip only non-semantic packaging suffixes (remaster/deluxe/feat), preserving version words."""
    if not isinstance(text, str):
        return ""
    cleaned = PACKAGING_SUFFIX_RE.sub("", text.lower().strip())
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def sessionize_timestamps(
    df: pd.DataFrame,
    user_col: str = "user_idx",
    ts_col: str = "event_timestamp",
    gap_seconds: int = 1800,
) -> pd.Series:
    """Assign sequential session IDs per user whenever inter-event inactivity exceeds gap_seconds."""
    prev_ts = df.groupby(user_col)[ts_col].shift(1)
    is_new = prev_ts.isna() | ((df[ts_col] - prev_ts) > gap_seconds)
    return is_new.groupby(df[user_col]).cumsum().astype(np.int32)


def run_session_sensitivity(df: pd.DataFrame, thresholds: list[int] | None = None) -> dict[str, dict[str, float]]:
    """Measure session count & length sensitivity across inactivity gap thresholds."""
    thresholds = thresholds or [900, 1800, 3600, 7200]
    results: dict[str, dict[str, float]] = {}
    for gap in thresholds:
        sids = sessionize_timestamps(df, gap_seconds=gap)
        sess_sizes = df.groupby(["user_idx", sids]).size()
        results[f"{gap}s"] = {
            "num_sessions": int(len(sess_sizes)),
            "mean_events_per_session": round(float(sess_sizes.mean()), 2),
            "median_events_per_session": round(float(sess_sizes.median()), 2),
            "pct_sessions_ge_5_events": round(float((sess_sizes >= 5).mean() * 100.0), 2),
        }
    return results


def parse_mpd_slices(mpd_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Parse local Spotify MPD JSON slices into unique track table and ordered playlists."""
    mpd_files = sorted(mpd_dir.glob("data/mpd.slice.*.json"))
    if not mpd_files:
        raise FileNotFoundError(f"No MPD slice JSON files found under {mpd_dir / 'data'}")

    track_records: dict[str, dict[str, Any]] = {}
    playlist_rows: list[dict[str, Any]] = []
    raw_playlists = 0
    dropped_short = 0

    for fpath in mpd_files:
        with open(fpath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for pl in payload.get("playlists", []):
            raw_playlists += 1
            pid = int(pl["pid"])
            pl_name = str(pl.get("name", f"Playlist {pid}")).strip()
            tracks = pl.get("tracks", [])
            t_ids: list[str] = []
            for tr in tracks:
                uri = str(tr.get("track_uri", ""))
                if not uri.startswith("spotify:track:"):
                    continue
                tid = uri.split(":")[-1]
                t_ids.append(tid)
                if tid not in track_records:
                    track_records[tid] = {
                        "spotify_track_id": tid,
                        "track_name": str(tr.get("track_name", "")).strip(),
                        "artist_name": str(tr.get("artist_name", "")).strip(),
                        "album_name": str(tr.get("album_name", "")).strip(),
                        "duration_ms": int(tr.get("duration_ms", 210000)),
                        "mpd_count": 1,
                    }
                else:
                    track_records[tid]["mpd_count"] += 1

            if len(t_ids) < 8:
                dropped_short += 1
                continue
            playlist_rows.append(
                {
                    "pid": pid,
                    "name": pl_name,
                    "num_tracks_raw": len(t_ids),
                    "modified_at": int(pl.get("modified_at", 0)),
                    "spotify_track_ids": t_ids,
                }
            )

    stats = {
        "mpd_slice_files": len(mpd_files),
        "raw_playlists": raw_playlists,
        "dropped_playlists_lt_8": dropped_short,
        "retained_playlists": len(playlist_rows),
        "unique_mpd_tracks": len(track_records),
    }
    return pd.DataFrame(list(track_records.values())), pd.DataFrame(playlist_rows), stats


def build_verified_dataset(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/processed_v2"),
    session_gap_seconds: int = 1800,
    min_user_events: int = 25,
    max_events_per_user: int = 500,
    max_catalog_size: int = 40000,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute end-to-end provenance-aware ETL and write Parquet outputs to data/processed_v2/."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    lineage: dict[str, Any] = {"config": {"session_gap_seconds": session_gap_seconds, "seed": seed}}

    # ------------------------------------------------------------------
    # Stage 1: Spotify Tracks Audio Features (Measured Track Audio)
    # ------------------------------------------------------------------
    sp_csv = raw_dir / "spotify_tracks" / "dataset.csv"
    sp_raw = pd.read_csv(sp_csv)
    n_sp_raw = len(sp_raw)
    sp_df = sp_raw.dropna(subset=["track_id", "track_name", "artists"]).copy()
    sp_df["spotify_track_id"] = sp_df["track_id"].astype(str).str.strip()
    sp_df["primary_artist"] = sp_df["artists"].astype(str).apply(lambda x: x.split(";")[0].strip())
    sp_df["explicit_num"] = sp_df["explicit"].astype(float)

    sp_df = sp_df.sort_values("popularity", ascending=False)
    genre_agg = sp_df.groupby("spotify_track_id")["track_genre"].apply(lambda g: list(dict.fromkeys(g))).rename("all_genres")
    sp_dedup = sp_df.drop_duplicates(subset=["spotify_track_id"], keep="first").merge(
        genre_agg, on="spotify_track_id", how="left"
    )
    sp_dedup["artist_exact"] = sp_dedup["primary_artist"].map(normalize_exact)
    sp_dedup["track_exact"] = sp_dedup["track_name"].map(normalize_exact)
    sp_dedup["track_stripped"] = sp_dedup["track_name"].map(normalize_packaging_stripped)
    sp_dedup["version_tag"] = sp_dedup["track_name"].map(extract_version_tag)

    # Explicit audio provenance for measured Spotify Audio CSV tracks
    sp_dedup["audio_feature_source"] = "maharshipandya_spotify_csv"
    sp_dedup["audio_feature_type"] = "third_party_measured_spotify_audio"
    sp_dedup["audio_feature_confidence"] = 1.0
    sp_dedup["is_synthetic"] = False

    lineage["stage_1_spotify_tracks"] = {
        "input_rows": n_sp_raw,
        "dropped_null_rows": n_sp_raw - len(sp_df),
        "deduplicated_multi_genre_rows": len(sp_df) - len(sp_dedup),
        "output_unique_tracks": len(sp_dedup),
    }

    # ------------------------------------------------------------------
    # Stage 2: Spotify MPD Slices + Deterministic Playlist Split (70/15/15)
    # ------------------------------------------------------------------
    mpd_tracks_df, playlists_df, mpd_stats = parse_mpd_slices(raw_dir / "spotify_mpd")
    lineage["stage_2_spotify_mpd"] = mpd_stats

    # Deterministic 70/15/15 playlist-level split BEFORE any track selection or feature estimation
    rng_split = np.random.default_rng(seed)
    perm = rng_split.permutation(len(playlists_df))
    n_pl = len(playlists_df)
    n_tr_pl = int(0.70 * n_pl)
    n_va_pl = int(0.15 * n_pl)
    split_labels = np.empty(n_pl, dtype=object)
    split_labels[perm[:n_tr_pl]] = "train"
    split_labels[perm[n_tr_pl : n_tr_pl + n_va_pl]] = "val"
    split_labels[perm[n_tr_pl + n_va_pl :]] = "test"
    playlists_df["playlist_split"] = split_labels

    # Compute MPD track counts STRICTLY from train playlists to prevent test playlist leakage!
    train_mpd_counts: dict[str, int] = {}
    for tids in playlists_df.loc[playlists_df["playlist_split"] == "train", "spotify_track_ids"]:
        for tid in tids:
            train_mpd_counts[tid] = train_mpd_counts.get(tid, 0) + 1
    mpd_tracks_df["train_mpd_count"] = mpd_tracks_df["spotify_track_id"].map(train_mpd_counts).fillna(0).astype(int)
    mpd_tracks_df["artist_exact"] = mpd_tracks_df["artist_name"].map(normalize_exact)
    mpd_tracks_df["track_exact"] = mpd_tracks_df["track_name"].map(normalize_exact)
    mpd_tracks_df["track_stripped"] = mpd_tracks_df["track_name"].map(normalize_packaging_stripped)
    mpd_tracks_df["version_tag"] = mpd_tracks_df["track_name"].map(extract_version_tag)

    # Artist-level deterministic estimates (NO Gaussian noise!) for frequent train-MPD tracks missing measured audio
    artist_priors = (
        sp_dedup.groupby("artist_exact")
        .agg(
            prior_genre=("track_genre", "first"),
            prior_popularity=("popularity", "mean"),
            prior_track_count=("spotify_track_id", "count"),
            **{f"prior_{col}": (col, "mean") for col in AUDIO_FEATURE_COLS},
        )
        .reset_index()
    )

    sp_merged = sp_dedup.copy()
    sp_merged["train_mpd_count"] = sp_merged["spotify_track_id"].map(train_mpd_counts).fillna(0).astype(int)

    extra_mpd = mpd_tracks_df[
        (~mpd_tracks_df["spotify_track_id"].isin(set(sp_merged["spotify_track_id"])))
        & (mpd_tracks_df["train_mpd_count"] >= 4)
    ].merge(artist_priors, on="artist_exact", how="inner")

    keep_cols = [
        "spotify_track_id", "primary_artist", "artists", "album_name", "track_name",
        "popularity", "duration_ms", "track_genre", "all_genres",
        "artist_exact", "track_exact", "track_stripped", "version_tag",
        "train_mpd_count",
        "audio_feature_source", "audio_feature_type", "audio_feature_confidence", "is_synthetic",
        *AUDIO_FEATURE_COLS,
    ]

    if len(extra_mpd) > 0:
        # Confidence scales slightly with number of measured tracks by the artist, capped at 0.50 (never 1.0!)
        est_conf = np.clip(0.25 + 0.05 * np.log1p(extra_mpd["prior_track_count"].to_numpy()), 0.25, 0.50)
        extra_rows = pd.DataFrame(
            {
                "spotify_track_id": extra_mpd["spotify_track_id"],
                "primary_artist": extra_mpd["artist_name"],
                "artists": extra_mpd["artist_name"],
                "album_name": extra_mpd["album_name"],
                "track_name": extra_mpd["track_name"],
                "popularity": np.clip(extra_mpd["prior_popularity"].round(), 15, 95).astype(int),
                "duration_ms": extra_mpd["duration_ms"].clip(30000, 900000),
                "track_genre": extra_mpd["prior_genre"],
                "all_genres": extra_mpd["prior_genre"].apply(lambda g: [g]),
                "artist_exact": extra_mpd["artist_exact"],
                "track_exact": extra_mpd["track_exact"],
                "track_stripped": extra_mpd["track_stripped"],
                "version_tag": extra_mpd["version_tag"],
                "train_mpd_count": extra_mpd["train_mpd_count"],
                "audio_feature_source": "artist_mean_prior_from_spotify_csv",
                "audio_feature_type": "artist_level_estimate",
                "audio_feature_confidence": np.round(est_conf, 3),
                "is_synthetic": True,
            }
        )
        for col in AUDIO_FEATURE_COLS:
            vals = extra_mpd[f"prior_{col}"].to_numpy(dtype=np.float64)
            extra_rows[col] = np.round(vals).astype(float) if col in ("key", "mode", "explicit_num") else vals
        candidate_catalog = pd.concat([sp_merged[keep_cols], extra_rows[keep_cols]], ignore_index=True)
    else:
        candidate_catalog = sp_merged[keep_cols].copy()

    candidate_catalog = candidate_catalog[
        (candidate_catalog["artist_exact"] != "") & (candidate_catalog["track_exact"] != "")
    ].drop_duplicates(subset=["spotify_track_id"], keep="first").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stage 3: Version-Aware Entity Resolution Lookup Tables
    # ------------------------------------------------------------------
    # Tier 1: Exact normalized (artist_exact, track_exact, version_tag)
    exact_counts = candidate_catalog.groupby(["artist_exact", "track_exact", "version_tag"])["spotify_track_id"].transform("count")
    exact_unique = candidate_catalog[exact_counts == 1][
        ["artist_exact", "track_exact", "version_tag", "spotify_track_id"]
    ].rename(columns={"spotify_track_id": "exact_tid"})
    n_exact_ambiguous = int((exact_counts > 1).sum())

    # Tier 2: Packaging-stripped (artist_exact, track_stripped, version_tag) — version_tag MUST still match!
    strip_counts = candidate_catalog.groupby(["artist_exact", "track_stripped", "version_tag"])["spotify_track_id"].transform("count")
    strip_unique = candidate_catalog[strip_counts == 1][
        ["artist_exact", "track_stripped", "version_tag", "spotify_track_id"]
    ].rename(columns={"spotify_track_id": "strip_tid"})
    n_strip_ambiguous = int((strip_counts > 1).sum())

    exact_map = dict(
        zip(
            zip(exact_unique["artist_exact"], exact_unique["track_exact"], exact_unique["version_tag"]),
            exact_unique["exact_tid"],
        )
    )
    strip_map = dict(
        zip(
            zip(strip_unique["artist_exact"], strip_unique["track_stripped"], strip_unique["version_tag"]),
            strip_unique["strip_tid"],
        )
    )
    exact_ambig_keys = set(
        zip(
            candidate_catalog.loc[exact_counts > 1, "artist_exact"],
            candidate_catalog.loc[exact_counts > 1, "track_exact"],
            candidate_catalog.loc[exact_counts > 1, "version_tag"],
        )
    )

    # ------------------------------------------------------------------
    # Stage 4: Stream & Match Last.fm 1K Parquets (Portable PyArrow/Pandas)
    # ------------------------------------------------------------------
    lfm_files = [
        raw_dir / "lastfm_1k" / "train.gz.parquet",
        raw_dir / "lastfm_1k" / "valid.gz.parquet",
        raw_dir / "lastfm_1k" / "test.gz.parquet",
    ]
    matched_chunks: list[pd.DataFrame] = []
    er_stats = {
        "raw_lfm_events": 0,
        "dropped_null_lfm_events": 0,
        "matched_exact_normalized": 0,
        "matched_packaging_stripped_high": 0,
        "excluded_ambiguous": 0,
        "unmatched": 0,
    }

    for fpath in lfm_files:
        df_chunk = pd.read_parquet(
            fpath,
            columns=["user_id", "timestamp", "artist_name", "track_name", "country"],
        )
        er_stats["raw_lfm_events"] += len(df_chunk)
        df_chunk = df_chunk.dropna(subset=["user_id", "timestamp", "artist_name", "track_name"])
        er_stats["dropped_null_lfm_events"] += er_stats["raw_lfm_events"] - len(df_chunk) - sum(
            v for k, v in er_stats.items() if k in ("matched_exact_normalized", "matched_packaging_stripped_high", "excluded_ambiguous", "unmatched", "dropped_null_lfm_events")
        )

        # Deduplicate distinct (artist_name, track_name) pairs inside chunk for ultra-fast resolution
        unique_pairs = df_chunk[["artist_name", "track_name"]].drop_duplicates().copy()
        a_ex = unique_pairs["artist_name"].map(normalize_exact)
        t_ex = unique_pairs["track_name"].map(normalize_exact)
        t_st = unique_pairs["track_name"].map(normalize_packaging_stripped)
        v_tg = unique_pairs["track_name"].map(extract_version_tag)

        resolved_tids: list[str | None] = []
        resolved_tiers: list[str] = []
        resolved_confs: list[float] = []

        for ae, te, ts, vt in zip(a_ex, t_ex, t_st, v_tg):
            k_ex = (ae, te, vt)
            if k_ex in exact_map:
                resolved_tids.append(exact_map[k_ex])
                resolved_tiers.append("exact_normalized")
                resolved_confs.append(1.0)
            elif k_ex in exact_ambig_keys:
                resolved_tids.append(None)
                resolved_tiers.append("ambiguous")
                resolved_confs.append(0.4)
            else:
                k_st = (ae, ts, vt)
                if k_st in strip_map:
                    resolved_tids.append(strip_map[k_st])
                    resolved_tiers.append("packaging_stripped_high")
                    resolved_confs.append(0.85)
                else:
                    resolved_tids.append(None)
                    resolved_tiers.append("unmatched")
                    resolved_confs.append(0.0)

        unique_pairs["spotify_track_id"] = resolved_tids
        unique_pairs["match_tier"] = resolved_tiers
        unique_pairs["match_confidence"] = resolved_confs

        merged = df_chunk.merge(unique_pairs, on=["artist_name", "track_name"], how="left")
        vc = merged["match_tier"].value_counts()
        er_stats["matched_exact_normalized"] += int(vc.get("exact_normalized", 0))
        er_stats["matched_packaging_stripped_high"] += int(vc.get("packaging_stripped_high", 0))
        er_stats["excluded_ambiguous"] += int(vc.get("ambiguous", 0))
        er_stats["unmatched"] += int(vc.get("unmatched", 0))

        good = merged[merged["match_confidence"] >= 0.85][
            ["user_id", "timestamp", "country", "spotify_track_id", "match_tier", "match_confidence"]
        ].copy()
        good["country"] = good["country"].astype(object).fillna("Unknown").astype(str)
        matched_chunks.append(good)

    matched_events = pd.concat(matched_chunks, ignore_index=True)
    total_matched = len(matched_events)
    er_stats["total_reliable_matches"] = total_matched
    er_stats["reliable_match_rate_pct"] = round(total_matched / max(1, er_stats["raw_lfm_events"]) * 100.0, 2)
    er_stats["catalog_exact_ambiguous_rows"] = n_exact_ambiguous
    er_stats["catalog_strip_ambiguous_rows"] = n_strip_ambiguous
    lineage["stage_3_entity_resolution"] = er_stats

    # ------------------------------------------------------------------
    # Stage 5: Active Catalog Selection (WITHOUT using future test events!)
    # ------------------------------------------------------------------
    # Select catalog based purely on measured audio availability, train MPD co-occurrence,
    # and genre breadth (zero Last.fm test-period popularity leakage!)
    candidate_catalog["catalog_priority"] = (
        candidate_catalog["train_mpd_count"] * 5.0
        + (~candidate_catalog["is_synthetic"]).astype(float) * 20.0
        + candidate_catalog["popularity"] * 0.15
    )
    # Also ensure tracks appearing in matched Last.fm events (within first 80% chronological timestamps) are included
    matched_events["ts_dt"] = pd.to_datetime(matched_events["timestamp"], utc=True)
    matched_events["event_timestamp"] = (matched_events["ts_dt"].astype("int64") // 10**9).astype(np.int64)
    cutoff_ts = int(np.percentile(matched_events["event_timestamp"].to_numpy(), 80))
    early_lfm_counts = matched_events.loc[matched_events["event_timestamp"] <= cutoff_ts, "spotify_track_id"].value_counts()
    candidate_catalog["early_lfm_count"] = candidate_catalog["spotify_track_id"].map(early_lfm_counts).fillna(0).astype(int)
    candidate_catalog["catalog_priority"] += candidate_catalog["early_lfm_count"] * 3.0

    genre_top = candidate_catalog.sort_values("catalog_priority", ascending=False).groupby("track_genre").head(120)
    remaining_slots = max(0, max_catalog_size - len(genre_top))
    rest_top = (
        candidate_catalog[~candidate_catalog["spotify_track_id"].isin(set(genre_top["spotify_track_id"]))]
        .sort_values("catalog_priority", ascending=False)
        .head(remaining_slots)
    )
    active_tracks = (
        pd.concat([genre_top, rest_top], ignore_index=True)
        .sort_values(["catalog_priority", "spotify_track_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    active_tracks["item_idx"] = np.arange(len(active_tracks), dtype=np.int32)
    unique_artists = {a: i for i, a in enumerate(active_tracks["primary_artist"].unique())}
    unique_genres = {g: i for i, g in enumerate(sorted(active_tracks["track_genre"].unique()))}
    active_tracks["artist_idx"] = active_tracks["primary_artist"].map(unique_artists).astype(np.int32)
    active_tracks["genre_idx"] = active_tracks["track_genre"].map(unique_genres).astype(np.int32)
    tid_to_idx = dict(zip(active_tracks["spotify_track_id"], active_tracks["item_idx"]))

    # ------------------------------------------------------------------
    # Stage 6: Deduplication, Session Construction & Observed-Only Schema
    # ------------------------------------------------------------------
    events = matched_events[matched_events["spotify_track_id"].isin(tid_to_idx)].copy()
    events["item_idx"] = events["spotify_track_id"].map(tid_to_idx).astype(np.int32)
    events = events.sort_values(["user_id", "event_timestamp", "item_idx"]).reset_index(drop=True)

    n_before_dup = len(events)
    # Remove exact identical timestamp scrobble duplicates (same user, track, second)
    events = events.drop_duplicates(subset=["user_id", "item_idx", "event_timestamp"], keep="first").reset_index(drop=True)
    n_dropped_dups = n_before_dup - len(events)

    user_counts = events["user_id"].value_counts()
    valid_users = sorted(user_counts[user_counts >= min_user_events].index.tolist())
    n_dropped_low_act_users = int(events["user_id"].nunique()) - len(valid_users)
    events = events[events["user_id"].isin(set(valid_users))].reset_index(drop=True)
    user_map = {u: idx for idx, u in enumerate(valid_users)}
    events["user_idx"] = events["user_id"].map(user_map).astype(np.int32)

    # Strictly causal repeat flag: cumcount() > 0 depends ONLY on prior plays strictly before t!
    prior_play_count = events.groupby(["user_idx", "item_idx"]).cumcount()
    events["event_type"] = np.where(prior_play_count > 0, "repeated_track_event", "track_listened_event")
    events["event_source"] = "lastfm_1k_scrobble"
    events["event_confidence"] = events["match_confidence"].astype(np.float32)
    events["is_observed"] = True
    events["is_inferred"] = False

    # Keep most recent max_events_per_user events per user (recording exact dropped count)
    n_before_trunc = len(events)
    events = events.groupby("user_idx", group_keys=False).tail(max_events_per_user).reset_index(drop=True)
    n_truncated_old = n_before_trunc - len(events)

    # Sessionization + sensitivity analysis
    events["session_id"] = sessionize_timestamps(events, gap_seconds=session_gap_seconds)
    sensitivity = run_session_sensitivity(events)

    interactions_v2 = events[
        [
            "user_idx",
            "user_id",
            "country",
            "item_idx",
            "spotify_track_id",
            "event_timestamp",
            "session_id",
            "event_type",
            "event_source",
            "event_confidence",
            "is_observed",
            "is_inferred",
            "match_tier",
            "match_confidence",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Stage 7: Filter Playlists to Active Catalog (Preserving Split!)
    # ------------------------------------------------------------------
    clean_pl_rows: list[dict[str, Any]] = []
    for row in playlists_df.itertuples(index=False):
        mapped = [tid_to_idx[t] for t in row.spotify_track_ids if t in tid_to_idx]
        seen: set[int] = set()
        dedup_mapped = [x for x in mapped if not (x in seen or seen.add(x))]
        if len(dedup_mapped) >= 12:
            clean_pl_rows.append(
                {
                    "pid": int(row.pid),
                    "name": row.name,
                    "playlist_split": row.playlist_split,
                    "num_tracks": len(dedup_mapped),
                    "item_indices": dedup_mapped[:150],
                }
            )
    playlists_v2 = pd.DataFrame(clean_pl_rows)

    # Save v2 Parquet artifacts (leaving legacy data/processed/ completely untouched!)
    active_tracks.to_parquet(out_dir / "tracks_v2.parquet", index=False)
    interactions_v2.to_parquet(out_dir / "interactions_v2.parquet", index=False)
    playlists_v2.to_parquet(out_dir / "playlists_v2.parquet", index=False)

    pl_split_counts = playlists_v2["playlist_split"].value_counts().to_dict()
    event_type_dist = interactions_v2["event_type"].value_counts().to_dict()

    lineage["stage_4_sessions_and_output"] = {
        "dropped_exact_duplicate_timestamps": n_dropped_dups,
        "dropped_users_lt_min_events": n_dropped_low_act_users,
        "truncated_older_events": n_truncated_old,
        "final_users": int(interactions_v2["user_idx"].nunique()),
        "final_interactions": int(len(interactions_v2)),
        "event_type_distribution": {str(k): int(v) for k, v in event_type_dist.items()},
        "session_sensitivity": sensitivity,
        "final_tracks": int(len(active_tracks)),
        "measured_audio_tracks": int((~active_tracks["is_synthetic"]).sum()),
        "estimated_audio_tracks": int(active_tracks["is_synthetic"].sum()),
        "final_playlists_by_split": {str(k): int(v) for k, v in pl_split_counts.items()},
        "build_seconds": round(time.time() - t0, 2),
    }

    (out_dir / "lineage_metrics.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(json.dumps(lineage, indent=2))
    return lineage


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provenance-aware v2 music dataset.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed_v2"))
    parser.add_argument("--session-gap", type=int, default=1800)
    args = parser.parse_args()
    build_verified_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir, session_gap_seconds=args.session_gap)


if __name__ == "__main__":
    main()
