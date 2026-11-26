"""
FastAPI Backend for Resonance: Intelligent Music Recommendation & Adaptive Shuffle System.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.app.engine.adaptive_shuffle import ShuffleConfig
from backend.app.engine.recommender import HybridRecommendationEngine
from backend.app.schemas.api_models import (
    FeedbackRequest,
    RecommendRequest,
    ResetFeedbackRequest,
    ShuffleCompareRequest,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(ROOT_DIR / "data" / "processed")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT_DIR / "data" / "artifacts")))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(ROOT_DIR / "data" / "reports")))

ENGINE: HybridRecommendationEngine | None = None

ACTION_LABELS = {
    0: "early_skip",
    1: "partial_play",
    2: "complete_play",
    3: "repeat",
    4: "like",
    5: "dislike",
}


def get_engine() -> HybridRecommendationEngine:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized yet.")
    return ENGINE


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (PROCESSED_DIR / "tracks.parquet").exists():
        ENGINE = HybridRecommendationEngine(
            processed_dir=PROCESSED_DIR,
            artifacts_dir=ARTIFACTS_DIR,
            device=device,
        )
    yield


app = FastAPI(
    title="Resonance — Intelligent Music Recommendation & Adaptive Shuffle API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    eng = get_engine()
    return {
        "status": "ok",
        "device": str(eng.device),
        "num_tracks": eng.num_items,
        "num_artists": eng.num_artists,
        "num_genres": eng.num_genres,
        "num_users": eng.num_users,
        "num_playlists": len(eng.playlists_df),
    }


@app.get("/api/users")
def list_sample_users(limit: int = Query(default=24, ge=1, le=100)) -> dict[str, Any]:
    """Return diverse real listener personas for interactive exploration."""
    eng = get_engine()
    personas = []
    # Pick evenly spaced users so different taste profiles & countries are represented
    step = max(1, eng.num_users // limit)
    selected_uids = list(range(0, eng.num_users, step))[:limit]

    for uid in selected_uids:
        hist = eng.user_histories[uid]
        items = hist["items"]
        actions = hist["actions"]
        pos_items = items[actions >= 2]
        if len(pos_items) == 0:
            pos_items = items

        genres = [eng.genre_names[int(i)] for i in pos_items]
        g_vals, g_counts = np.unique(genres, return_counts=True)
        top_genres = [str(g_vals[k]) for k in np.argsort(-g_counts)[:3]]

        artists = [eng.artist_names[int(i)] for i in pos_items]
        a_vals, a_counts = np.unique(artists, return_counts=True)
        top_artists = [str(a_vals[k]) for k in np.argsort(-a_counts)[:3]]

        skip_rate = float(np.mean(actions == 0))
        personas.append(
            {
                "user_idx": uid,
                "user_id": hist["user_id"],
                "country": hist["country"],
                "total_events": int(len(items)),
                "historical_skip_rate": round(skip_rate, 3),
                "top_genres": top_genres,
                "top_artists": top_artists,
            }
        )

    return {"users": personas}


@app.get("/api/users/{user_idx}/profile")
def get_user_profile(user_idx: int) -> dict[str, Any]:
    """Detailed listener profile comparing Long-Term baseline vs. Short-Term active session taste."""
    eng = get_engine()
    if user_idx not in eng.user_histories:
        raise HTTPException(status_code=404, detail=f"User {user_idx} not found.")

    hist = eng.user_histories[user_idx]
    fb_state = eng.get_feedback_state(user_idx)
    rep = eng.compute_user_representation(user_idx)

    all_items = hist["items"]
    all_actions = hist["actions"]
    pos_items = all_items[all_actions >= 2]
    if len(pos_items) == 0:
        pos_items = all_items

    radar_cols = ["danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness"]
    long_df = eng.tracks_df.iloc[pos_items]
    long_radar = {c: round(float(long_df[c].mean()), 3) for c in radar_cols}

    short_items = [
        it for it, act in zip(rep["recent_items"][-12:], rep["recent_actions"][-12:]) if act >= 1
    ] or pos_items[-10:].tolist()
    short_df = eng.tracks_df.iloc[short_items]
    short_radar = {c: round(float(short_df[c].mean()), 3) for c in radar_cols}

    # Genre distribution
    g_vals, g_counts = np.unique(long_df["track_genre"], return_counts=True)
    top_idx = np.argsort(-g_counts)[:6]
    genre_breakdown = [
        {"genre": str(g_vals[i]), "share": round(float(g_counts[i] / len(long_df)), 3)} for i in top_idx
    ]

    # Recent timeline (last 14 historical + any live feedback events)
    recent_timeline = []
    for it, act in zip(hist["items"][-12:].tolist(), hist["actions"][-12:].tolist()):
        row = eng.tracks_df.iloc[int(it)]
        recent_timeline.append(
            {
                "item_idx": int(it),
                "spotify_track_id": str(row["spotify_track_id"]),
                "track_name": str(row["track_name"]),
                "artist_name": str(row["primary_artist"]),
                "genre": str(row["track_genre"]),
                "action": ACTION_LABELS.get(int(act), "complete_play"),
                "source": "historical",
            }
        )
    for ev in fb_state.events[-12:]:
        row = eng.tracks_df.iloc[int(ev.item_idx)]
        recent_timeline.append(
            {
                "item_idx": int(ev.item_idx),
                "spotify_track_id": str(row["spotify_track_id"]),
                "track_name": str(row["track_name"]),
                "artist_name": str(row["primary_artist"]),
                "genre": str(row["track_genre"]),
                "action": ev.action,
                "reward": round(ev.reward, 2),
                "source": "live_feedback",
            }
        )

    return {
        "user_idx": user_idx,
        "user_id": hist["user_id"],
        "country": hist["country"],
        "long_term_gate": round(rep["long_term_gate"], 3),
        "short_term_gate": round(1.0 - rep["long_term_gate"], 3),
        "long_term_radar": long_radar,
        "short_term_radar": short_radar,
        "genre_breakdown": genre_breakdown,
        "feedback_summary": fb_state.summary(),
        "recent_timeline": recent_timeline[-16:],
    }


@app.post("/api/recommend")
def generate_recommendations(req: RecommendRequest) -> dict[str, Any]:
    eng = get_engine()
    if req.user_idx not in eng.user_histories:
        raise HTTPException(status_code=404, detail=f"User {req.user_idx} not found.")
    return eng.recommend(
        user_idx=req.user_idx,
        top_k=req.top_k,
        discovery_level=req.discovery_level,
        diversity_level=req.diversity_level,
        novelty_boost=req.novelty_boost,
        short_term_weight=req.short_term_weight,
    )


@app.get("/api/playlists")
def list_curated_playlists(limit: int = Query(default=30, ge=5, le=100)) -> dict[str, Any]:
    eng = get_engine()
    # Select playlists that have multiple tracks per artist too (great stress-test for shuffle repetition!)
    rows = []
    for row in eng.playlists_df.head(limit * 3).itertuples(index=False):
        idxs = list(row.item_indices[:60])
        if len(idxs) < 16:
            continue
        arts = [eng.artist_names[i] for i in idxs]
        gens = [eng.genre_names[i] for i in idxs]
        unique_artist_ratio = len(set(arts)) / len(arts)
        top_g = list(dict.fromkeys(gens))[:3]
        rows.append(
            {
                "pid": int(row.pid),
                "name": str(row.name),
                "num_tracks": len(idxs),
                "unique_artists": len(set(arts)),
                "artist_concentration": round(1.0 - unique_artist_ratio, 3),
                "sample_genres": top_g,
                "sample_artists": list(dict.fromkeys(arts))[:4],
            }
        )
    # Prioritize realistic multi-artist playlists (20%-45% repeat artists) where anti-clumping shines
    rows.sort(
        key=lambda r: (
            0 if 0.18 <= r["artist_concentration"] <= 0.45 else 1,
            -r["artist_concentration"],
        )
    )
    return {"playlists": rows[:limit]}


@app.post("/api/shuffle/compare")
def compare_shuffle_algorithms(req: ShuffleCompareRequest) -> dict[str, Any]:
    """Run Adaptive Shuffle head-to-head against Fisher-Yates Random & Spotify Artist-Spaced on the same pool."""
    eng = get_engine()
    if req.user_idx not in eng.user_histories:
        raise HTTPException(status_code=404, detail=f"User {req.user_idx} not found.")

    source_name = "Personalized Hybrid Mix"
    if req.playlist_pid is not None:
        match = eng.playlists_df[eng.playlists_df["pid"] == req.playlist_pid]
        if len(match) == 0:
            raise HTTPException(status_code=404, detail=f"Playlist {req.playlist_pid} not found.")
        pool_indices = list(match.iloc[0]["item_indices"][: req.pool_size])
        source_name = str(match.iloc[0]["name"])
    else:
        # Build a realistic mixed pool (familiar favorites + discovery recommendations)
        rec = eng.recommend(
            user_idx=req.user_idx,
            top_k=req.pool_size,
            discovery_level=req.discovery_target,
            diversity_level=0.35,  # allow natural artist overlap so shuffle has real repetition to solve
        )
        pool_indices = [t["item_idx"] for t in rec["recommendations"]]

    base_seed = req.seed if req.seed is not None else int(np.random.randint(1, 100000))

    adaptive_res = eng.shuffle_pool(
        user_idx=req.user_idx,
        item_indices=pool_indices,
        config=ShuffleConfig(
            mode="adaptive",
            diversity_weight=req.diversity_weight,
            flow_weight=req.flow_weight,
            relevance_weight=req.relevance_weight,
            artist_cooldown=req.artist_cooldown,
            genre_cooldown=req.genre_cooldown,
            entropy_temperature=req.entropy_temperature,
            discovery_target=req.discovery_target,
            seed=base_seed,
        ),
    )
    random_res = eng.shuffle_pool(
        user_idx=req.user_idx,
        item_indices=pool_indices,
        config=ShuffleConfig(mode="fisher_yates", seed=base_seed),
    )
    spaced_res = eng.shuffle_pool(
        user_idx=req.user_idx,
        item_indices=pool_indices,
        config=ShuffleConfig(mode="artist_spaced", seed=base_seed),
    )

    return {
        "source_name": source_name,
        "pool_size": len(pool_indices),
        "seed": base_seed,
        "adaptive": adaptive_res,
        "fisher_yates": random_res,
        "artist_spaced": spaced_res,
    }


@app.post("/api/feedback")
def record_live_feedback(req: FeedbackRequest) -> dict[str, Any]:
    eng = get_engine()
    if req.user_idx not in eng.user_histories:
        raise HTTPException(status_code=404, detail=f"User {req.user_idx} not found.")
    if not (0 <= req.item_idx < eng.num_items):
        raise HTTPException(status_code=404, detail=f"Track {req.item_idx} not found.")

    fb_state = eng.get_feedback_state(req.user_idx)
    ev = fb_state.record_feedback(
        item_idx=req.item_idx,
        item_emb=eng.item_embeddings[req.item_idx],
        artist_idx=int(eng.artist_ids[req.item_idx]),
        genre_idx=int(eng.genre_ids[req.item_idx]),
        action=req.action,
        played_ratio=req.played_ratio,
    )
    row = eng.tracks_df.iloc[req.item_idx]
    return {
        "status": "recorded",
        "event": {
            "item_idx": ev.item_idx,
            "track_name": str(row["track_name"]),
            "artist_name": str(row["primary_artist"]),
            "genre": str(row["track_genre"]),
            "action": ev.action,
            "reward": round(ev.reward, 3),
        },
        "feedback_summary": fb_state.summary(),
    }


@app.post("/api/feedback/reset")
def reset_user_feedback(req: ResetFeedbackRequest) -> dict[str, Any]:
    eng = get_engine()
    eng.reset_feedback_state(req.user_idx)
    return {"status": "reset", "user_idx": req.user_idx}


@app.get("/api/catalog/search")
def search_catalog(q: str = Query(..., min_length=1), limit: int = Query(default=15, ge=1, le=50)) -> dict[str, Any]:
    eng = get_engine()
    ql = q.lower().strip()
    df = eng.tracks_df
    mask = (
        df["track_name"].str.lower().str.contains(ql, na=False, regex=False)
        | df["primary_artist"].str.lower().str.contains(ql, na=False, regex=False)
        | df["track_genre"].str.lower().str.contains(ql, na=False, regex=False)
    )
    hits = df[mask].sort_values("popularity", ascending=False).head(limit)
    results = [
        {
            "item_idx": int(r.item_idx),
            "spotify_track_id": str(r.spotify_track_id),
            "track_name": str(r.track_name),
            "artist_name": str(r.primary_artist),
            "album_name": str(r.album_name),
            "genre": str(r.track_genre),
            "popularity": int(r.popularity),
            "duration_ms": int(r.duration_ms),
        }
        for r in hits.itertuples(index=False)
    ]
    return {"results": results}


@app.get("/api/benchmarks")
def get_offline_benchmarks() -> dict[str, Any]:
    report_file = REPORTS_DIR / "evaluation_results.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not generated yet.")
    return json.loads(report_file.read_text(encoding="utf-8"))
