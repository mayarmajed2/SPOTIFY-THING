"""
Automated Dataset Downloader & Unified Music Catalog / Session ETL Pipeline.

Downloads three public, ungated Hugging Face datasets:
1. maharshipandya/spotify-tracks-dataset (114k tracks, 125 genres, 12 audio features)
2. jaxliu/Spotify_Million_Playlist_Dataset_Challenge (configurable MPD slices, default 10 = 10,000 playlists)
3. matthewfranglen/lastfm-1k (multi-year timestamped user listening histories)

Builds unified processed Parquet artifacts in `data/processed/`:
- tracks.parquet        : Canonical Spotify-ID catalog with normalized 12D audio vectors & genre metadata
- interactions.parquet  : Chronological user-track events with timestamp-derived implicit feedback (complete, early_skip, repeat, favorite)
- playlists.parquet     : Real user-curated Spotify playlists for Shuffle & Continuation benchmarking + UI demo
- dataset_stats.json    : Summary statistics for reporting & UI diagnostics
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

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

MPD_SLICE_NAMES = [
    "data/mpd.slice.0-999.json",
    "data/mpd.slice.1000-1999.json",
    "data/mpd.slice.2000-2999.json",
    "data/mpd.slice.3000-3999.json",
    "data/mpd.slice.4000-4999.json",
    "data/mpd.slice.5000-5999.json",
    "data/mpd.slice.6000-6999.json",
    "data/mpd.slice.7000-7999.json",
    "data/mpd.slice.8000-8999.json",
    "data/mpd.slice.9000-9999.json",
]


def normalize_text(s: Any) -> str:
    """Normalize artist or track names for cross-dataset entity resolution."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    # Strip feat./remaster/version parentheticals for cleaner Last.fm <-> Spotify matching
    s = re.sub(r"\s*[\(\[\-]\s*(feat\.?|ft\.?|remaster|live|radio edit|version|deluxe|bonus).*$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def download_datasets(raw_dir: Path, num_mpd_slices: int = 10) -> dict[str, Any]:
    """Download public datasets from Hugging Face Hub into data/raw/."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {}

    print("[1/3] Downloading maharshipandya/spotify-tracks-dataset (19.2 MB)...")
    paths["spotify_tracks_csv"] = Path(
        hf_hub_download(
            repo_id="maharshipandya/spotify-tracks-dataset",
            filename="dataset.csv",
            repo_type="dataset",
            local_dir=raw_dir / "spotify_tracks",
        )
    )

    print(f"[2/3] Downloading {num_mpd_slices} slices of Spotify Million Playlist Dataset...")
    mpd_files: list[Path] = []
    repo_files = list_repo_files(
        repo_id="jaxliu/Spotify_Million_Playlist_Dataset_Challenge",
        repo_type="dataset",
    )
    slice_candidates = sorted([f for f in repo_files if f.startswith("data/mpd.slice.") and f.endswith(".json")])[
        :num_mpd_slices
    ]
    for idx, fname in enumerate(slice_candidates, 1):
        p = Path(
            hf_hub_download(
                repo_id="jaxliu/Spotify_Million_Playlist_Dataset_Challenge",
                filename=fname,
                repo_type="dataset",
                local_dir=raw_dir / "spotify_mpd",
            )
        )
        mpd_files.append(p)
        print(f"      Downloaded slice {idx}/{num_mpd_slices}: {fname}")
    paths["mpd_slices"] = mpd_files

    print("[3/3] Downloading matthewfranglen/lastfm-1k (train/valid/test parquet)...")
    lfm_files: list[Path] = []
    for split_file in ["train.gz.parquet", "valid.gz.parquet", "test.gz.parquet"]:
        p = Path(
            hf_hub_download(
                repo_id="matthewfranglen/lastfm-1k",
                filename=split_file,
                repo_type="dataset",
                local_dir=raw_dir / "lastfm_1k",
            )
        )
        lfm_files.append(p)
        print(f"      Downloaded Last.fm split: {split_file}")
    paths["lastfm_parquets"] = lfm_files

    return paths


def parse_mpd_slices(mpd_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract unique MPD tracks and playlist sequences from MPD JSON slices."""
    track_records: dict[str, dict[str, Any]] = {}
    playlist_rows: list[dict[str, Any]] = []
    playlist_track_rows: list[dict[str, Any]] = []

    for fpath in mpd_files:
        with open(fpath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for pl in payload.get("playlists", []):
            pid = int(pl["pid"])
            pl_name = str(pl.get("name", f"Playlist {pid}")).strip()
            tracks = pl.get("tracks", [])
            if len(tracks) < 8:
                continue
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

            if len(t_ids) >= 8:
                playlist_rows.append(
                    {
                        "pid": pid,
                        "name": pl_name,
                        "num_tracks": len(t_ids),
                        "modified_at": int(pl.get("modified_at", 0)),
                        "spotify_track_ids": t_ids,
                    }
                )
                for pos, tid in enumerate(t_ids):
                    playlist_track_rows.append({"pid": pid, "pos": pos, "spotify_track_id": tid})

    mpd_tracks_df = pd.DataFrame(list(track_records.values()))
    playlists_df = pd.DataFrame(playlist_rows)
    pl_edges_df = pd.DataFrame(playlist_track_rows)
    return mpd_tracks_df, playlists_df, pl_edges_df


def build_unified_dataset(
    paths: dict[str, Any],
    processed_dir: Path,
    max_catalog_size: int = 40000,
) -> dict[str, Any]:
    """Fuse Spotify Audio Tracks, Spotify MPD, and Last.fm 1K into clean Parquet tables."""
    t0 = time.time()
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Step A: Loading & Deduplicating Spotify Audio Features Catalog ---")
    sp_df = pd.read_csv(paths["spotify_tracks_csv"])
    sp_df = sp_df.dropna(subset=["track_id", "track_name", "artists"]).copy()
    sp_df["spotify_track_id"] = sp_df["track_id"].astype(str).str.strip()
    sp_df["primary_artist"] = sp_df["artists"].astype(str).apply(lambda x: x.split(";")[0].strip())
    sp_df["explicit_num"] = sp_df["explicit"].astype(float)

    # Collect all genres per track_id, keep highest popularity row as canonical
    sp_df = sp_df.sort_values("popularity", ascending=False)
    genre_agg = sp_df.groupby("spotify_track_id")["track_genre"].apply(lambda g: list(dict.fromkeys(g))).rename("all_genres")
    sp_dedup = sp_df.drop_duplicates(subset=["spotify_track_id"], keep="first").merge(
        genre_agg, on="spotify_track_id", how="left"
    )
    sp_dedup["artist_norm"] = sp_dedup["primary_artist"].map(normalize_text)
    sp_dedup["track_norm"] = sp_dedup["track_name"].map(normalize_text)
    sp_dedup["has_exact_audio"] = True
    print(f"Unique Spotify audio-feature tracks: {len(sp_dedup):,}")

    print("\n--- Step B: Parsing Spotify Million Playlist Dataset Slices ---")
    mpd_tracks_df, playlists_df, pl_edges_df = parse_mpd_slices(paths["mpd_slices"])
    mpd_tracks_df["artist_norm"] = mpd_tracks_df["artist_name"].map(normalize_text)
    mpd_tracks_df["track_norm"] = mpd_tracks_df["track_name"].map(normalize_text)
    print(f"Parsed {len(playlists_df):,} playlists containing {len(mpd_tracks_df):,} unique MPD tracks.")

    # Artist-level genre and audio priors from Spotify Audio dataset (for MPD tracks not in the 89.7k CSV)
    artist_priors = (
        sp_dedup.groupby("artist_norm")
        .agg(
            prior_genre=("track_genre", "first"),
            prior_popularity=("popularity", "mean"),
            **{f"prior_{col}": (col, "mean") for col in AUDIO_FEATURE_COLS},
        )
        .reset_index()
    )
    global_means = {col: float(sp_dedup[col].mean()) for col in AUDIO_FEATURE_COLS}

    # Merge MPD counts onto sp_dedup
    sp_merged = sp_dedup.merge(
        mpd_tracks_df[["spotify_track_id", "mpd_count"]],
        on="spotify_track_id",
        how="left",
    )
    sp_merged["mpd_count"] = sp_merged["mpd_count"].fillna(0).astype(int)

    # Also include frequent MPD tracks (>= 4 playlist appearances) whose artist has known Spotify priors
    extra_mpd = mpd_tracks_df[
        (~mpd_tracks_df["spotify_track_id"].isin(set(sp_merged["spotify_track_id"])))
        & (mpd_tracks_df["mpd_count"] >= 4)
    ].merge(artist_priors, on="artist_norm", how="inner")

    if len(extra_mpd) > 0:
        rng = np.random.default_rng(42)
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
                "artist_norm": extra_mpd["artist_norm"],
                "track_norm": extra_mpd["track_norm"],
                "mpd_count": extra_mpd["mpd_count"],
                "has_exact_audio": False,
            }
        )
        for col in AUDIO_FEATURE_COLS:
            base_vals = extra_mpd[f"prior_{col}"].fillna(global_means[col]).to_numpy()
            if col in ("key", "mode", "explicit_num"):
                extra_rows[col] = np.round(base_vals).astype(float)
            elif col == "tempo":
                extra_rows[col] = np.clip(base_vals + rng.normal(0, 6.0, size=len(base_vals)), 50.0, 210.0)
            elif col == "loudness":
                extra_rows[col] = np.clip(base_vals + rng.normal(0, 1.2, size=len(base_vals)), -35.0, 0.0)
            else:
                extra_rows[col] = np.clip(base_vals + rng.normal(0, 0.04, size=len(base_vals)), 0.0, 1.0)

        keep_cols = list(extra_rows.columns)
        combined_catalog = pd.concat([sp_merged[keep_cols], extra_rows], ignore_index=True)
    else:
        keep_cols = [
            "spotify_track_id", "primary_artist", "artists", "album_name", "track_name",
            "popularity", "duration_ms", "track_genre", "all_genres", "artist_norm",
            "track_norm", "mpd_count", "has_exact_audio", *AUDIO_FEATURE_COLS,
        ]
        combined_catalog = sp_merged[keep_cols].copy()

    combined_catalog = combined_catalog[
        (combined_catalog["artist_norm"] != "") & (combined_catalog["track_norm"] != "")
    ].drop_duplicates(subset=["spotify_track_id"], keep="first")

    print(f"Combined candidate Spotify catalog before Last.fm matching: {len(combined_catalog):,} tracks")

    print("\n--- Step C: Matching Last.fm 1K Chronological User Logs via DuckDB ---")
    con = duckdb.connect()
    # Canonical lookup key (artist_norm, track_norm) -> highest (has_exact_audio, mpd_count, popularity)
    lookup_df = (
        combined_catalog.sort_values(["has_exact_audio", "mpd_count", "popularity"], ascending=[False, False, False])
        .drop_duplicates(subset=["artist_norm", "track_norm"], keep="first")[
            ["spotify_track_id", "artist_norm", "track_norm", "duration_ms"]
        ]
    )
    con.register("catalog_lookup", lookup_df)

    lfm_glob = [str(p).replace("\\", "/") for p in paths["lastfm_parquets"]]
    lfm_sql_list = ", ".join(f"'{p}'" for p in lfm_glob)

    matched_events = con.execute(
        f"""
        WITH raw_lfm AS (
            SELECT
                user_id,
                timestamp AS ts,
                trim(regexp_replace(
                    regexp_replace(lower(coalesce(artist_name, '')), '\\s*[\\(\\[\\-]\\s*(feat\\.?|ft\\.?|remaster|live|radio edit|version|deluxe|bonus).*$', ''),
                    '[^a-z0-9]+', ' ', 'g'
                )) AS artist_norm,
                trim(regexp_replace(
                    regexp_replace(lower(coalesce(track_name, '')), '\\s*[\\(\\[\\-]\\s*(feat\\.?|ft\\.?|remaster|live|radio edit|version|deluxe|bonus).*$', ''),
                    '[^a-z0-9]+', ' ', 'g'
                )) AS track_norm,
                coalesce(country, 'Unknown') AS country
            FROM read_parquet([{lfm_sql_list}])
            WHERE timestamp IS NOT NULL AND artist_name IS NOT NULL AND track_name IS NOT NULL
        )
        SELECT
            r.user_id,
            r.ts,
            r.country,
            c.spotify_track_id,
            c.duration_ms
        FROM raw_lfm r
        INNER JOIN catalog_lookup c
            ON r.artist_norm = c.artist_norm AND r.track_norm = c.track_norm
        ORDER BY r.user_id, r.ts
        """
    ).df()
    print(f"Matched {len(matched_events):,} real timestamped Last.fm listening events to Spotify tracks!")

    # Count Last.fm plays per track to select the active catalog (up to max_catalog_size)
    lfm_counts = matched_events["spotify_track_id"].value_counts().rename("lfm_count")
    combined_catalog = combined_catalog.merge(lfm_counts, on="spotify_track_id", how="left")
    combined_catalog["lfm_count"] = combined_catalog["lfm_count"].fillna(0).astype(int)

    # Prioritize tracks that appear in Last.fm OR MPD, plus diverse genre coverage across all 125 genres
    combined_catalog["activity_score"] = (
        combined_catalog["lfm_count"] * 3.0
        + combined_catalog["mpd_count"] * 5.0
        + combined_catalog["popularity"] * 0.25
        + combined_catalog["has_exact_audio"].astype(float) * 10.0
    )
    # Guarantee at least 120 tracks per genre for broad discovery + top active tracks up to max_catalog_size
    genre_top = combined_catalog.sort_values("activity_score", ascending=False).groupby("track_genre").head(120)
    remaining_slots = max(0, max_catalog_size - len(genre_top))
    rest_top = (
        combined_catalog[~combined_catalog["spotify_track_id"].isin(set(genre_top["spotify_track_id"]))]
        .sort_values("activity_score", ascending=False)
        .head(remaining_slots)
    )
    active_tracks = pd.concat([genre_top, rest_top], ignore_index=True).sort_values(
        ["activity_score", "spotify_track_id"], ascending=[False, True]
    ).reset_index(drop=True)

    active_tracks["item_idx"] = np.arange(len(active_tracks), dtype=np.int32)

    # Encode artist_idx and genre_idx + standardized 12D audio features
    unique_artists = {a: i for i, a in enumerate(active_tracks["primary_artist"].unique())}
    unique_genres = {g: i for i, g in enumerate(sorted(active_tracks["track_genre"].unique()))}
    active_tracks["artist_idx"] = active_tracks["primary_artist"].map(unique_artists).astype(np.int32)
    active_tracks["genre_idx"] = active_tracks["track_genre"].map(unique_genres).astype(np.int32)

    # Standardized audio vector columns (z-score / min-max scaled cleanly for neural networks)
    norm_cols: list[str] = []
    for col in AUDIO_FEATURE_COLS:
        vals = active_tracks[col].astype(np.float32)
        mean, std = float(vals.mean()), float(vals.std()) + 1e-6
        ncol = f"norm_{col}"
        active_tracks[ncol] = ((vals - mean) / std).astype(np.float32)
        norm_cols.append(ncol)

    tid_to_idx = dict(zip(active_tracks["spotify_track_id"], active_tracks["item_idx"]))

    print("\n--- Step D: Extracting Sessions & Timestamp-Derived Implicit Feedback ---")
    matched_events = matched_events[matched_events["spotify_track_id"].isin(tid_to_idx)].copy()
    matched_events["item_idx"] = matched_events["spotify_track_id"].map(tid_to_idx).astype(np.int32)
    matched_events["ts"] = pd.to_datetime(matched_events["ts"], utc=True)
    matched_events = matched_events.sort_values(["user_id", "ts"]).reset_index(drop=True)

    # Filter users with at least 25 matched chronological listens
    user_counts = matched_events["user_id"].value_counts()
    valid_users = sorted(user_counts[user_counts >= 25].index.tolist())
    matched_events = matched_events[matched_events["user_id"].isin(set(valid_users))].reset_index(drop=True)
    user_map = {u: idx for idx, u in enumerate(valid_users)}
    matched_events["user_idx"] = matched_events["user_id"].map(user_map).astype(np.int32)

    # Compute delta to next track & previous track within each user to infer sessions & implicit skips/repeats
    matched_events["prev_ts"] = matched_events.groupby("user_idx")["ts"].shift(1)
    matched_events["next_ts"] = matched_events.groupby("user_idx")["ts"].shift(-1)
    matched_events["prev_item"] = matched_events.groupby("user_idx")["item_idx"].shift(1)
    matched_events["prev2_item"] = matched_events.groupby("user_idx")["item_idx"].shift(2)

    gap_from_prev_sec = (matched_events["ts"] - matched_events["prev_ts"]).dt.total_seconds().fillna(999999.0)
    listen_sec = (matched_events["next_ts"] - matched_events["ts"]).dt.total_seconds().fillna(240.0)

    # New session whenever user pauses > 30 minutes (1800s)
    is_new_session = (gap_from_prev_sec > 1800.0).astype(np.int32)
    matched_events["session_id"] = matched_events.groupby("user_idx")["prev_ts"].transform(
        lambda s: ((s.isna()) | ((matched_events.loc[s.index, "ts"] - s).dt.total_seconds() > 1800.0)).cumsum()
    ).astype(np.int32)

    dur_sec = np.clip(matched_events["duration_ms"].to_numpy(dtype=np.float64) / 1000.0, 60.0, 600.0)
    played_ratio = np.where(listen_sec > 1800.0, 1.0, np.clip(listen_sec / dur_sec, 0.0, 1.5))
    matched_events["played_ratio"] = played_ratio.astype(np.float32)

    # User-track total play count -> repeat / favorite signal
    ut_counts = matched_events.groupby(["user_idx", "item_idx"])["ts"].transform("count")
    is_immediate_repeat = (
        (matched_events["item_idx"] == matched_events["prev_item"])
        | (matched_events["item_idx"] == matched_events["prev2_item"])
    ) & (gap_from_prev_sec <= 1800.0)

    # Interaction action codes:
    # 0 = early_skip (< 30% played and next track starts within < 45s in same session)
    # 1 = partial_play (30% - 75% played)
    # 2 = complete_play (>= 75% played)
    # 3 = repeat_listen (immediate session replay)
    # 4 = favorite_listen (track played >= 5 times by this user & completed)
    action_id = np.full(len(matched_events), 2, dtype=np.int8)
    action_id[(played_ratio < 0.75) & (listen_sec <= 1800.0)] = 1
    action_id[(played_ratio < 0.30) & (listen_sec < 45.0)] = 0
    action_id[is_immediate_repeat.to_numpy()] = 3
    action_id[(ut_counts.to_numpy() >= 5) & (played_ratio >= 0.70)] = 4

    reward_lookup = np.array([-0.75, 0.15, 0.55, 0.85, 1.0], dtype=np.float32)
    matched_events["action_id"] = action_id
    matched_events["reward"] = reward_lookup[action_id]
    matched_events["unix_ts"] = (matched_events["ts"].astype("int64") // 10**9).astype(np.int64)

    # Keep most recent 400 events per user so every user has clean, fast, rich multi-session trajectories
    interactions_df = (
        matched_events.groupby("user_idx", group_keys=False)
        .tail(400)[
            [
                "user_idx",
                "user_id",
                "country",
                "session_id",
                "item_idx",
                "spotify_track_id",
                "unix_ts",
                "played_ratio",
                "action_id",
                "reward",
            ]
        ]
        .reset_index(drop=True)
    )

    print("\n--- Step E: Filtering Curated MPD Playlists to Active Catalog ---")
    filtered_playlists: list[dict[str, Any]] = []
    for row in playlists_df.itertuples(index=False):
        mapped = [tid_to_idx[t] for t in row.spotify_track_ids if t in tid_to_idx]
        # Deduplicate while preserving playlist order
        seen: set[int] = set()
        dedup_mapped = [x for x in mapped if not (x in seen or seen.add(x))]
        if len(dedup_mapped) >= 12:
            filtered_playlists.append(
                {
                    "pid": int(row.pid),
                    "name": row.name,
                    "num_tracks": len(dedup_mapped),
                    "item_indices": dedup_mapped[:150],
                }
            )
    clean_playlists_df = pd.DataFrame(filtered_playlists)

    # Save Parquet artifacts
    tracks_out = processed_dir / "tracks.parquet"
    interactions_out = processed_dir / "interactions.parquet"
    playlists_out = processed_dir / "playlists.parquet"

    active_tracks.to_parquet(tracks_out, index=False)
    interactions_df.to_parquet(interactions_out, index=False)
    clean_playlists_df.to_parquet(playlists_out, index=False)

    action_names = {
        0: "early_skip",
        1: "partial_play",
        2: "complete_play",
        3: "repeat_listen",
        4: "favorite_listen",
    }
    action_dist = {
        action_names[int(k)]: int(v)
        for k, v in interactions_df["action_id"].value_counts().sort_index().items()
    }

    stats = {
        "num_tracks": int(len(active_tracks)),
        "num_exact_audio_tracks": int(active_tracks["has_exact_audio"].sum()),
        "num_artists": int(active_tracks["artist_idx"].nunique()),
        "num_genres": int(active_tracks["genre_idx"].nunique()),
        "num_users": int(interactions_df["user_idx"].nunique()),
        "num_interactions": int(len(interactions_df)),
        "num_curated_playlists": int(len(clean_playlists_df)),
        "action_distribution": action_dist,
        "audio_feature_cols": AUDIO_FEATURE_COLS,
        "norm_audio_cols": norm_cols,
        "build_seconds": round(time.time() - t0, 2),
    }

    stats_path = processed_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n=== Unified Dataset Summary ===")
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and fuse music recommendation datasets.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--mpd-slices", type=int, default=10)
    parser.add_argument("--max-catalog", type=int, default=40000)
    args = parser.parse_args()

    paths = download_datasets(args.raw_dir, num_mpd_slices=args.mpd_slices)
    build_unified_dataset(paths, args.processed_dir, max_catalog_size=args.max_catalog)


if __name__ == "__main__":
    main()
