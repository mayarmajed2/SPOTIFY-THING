"""
Comprehensive Integration, Security, User-Isolation & ML Pipeline Tests for `/api/v1/*` Companion Extension API.

Builds a self-contained synthetic mini-catalog + real PyTorch NeuMF & Dual-Horizon Transformer checkpoints
inside `tmp_path` so this entire test suite runs deterministically in CI and locally without external datasets.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import main as app_main
from backend.app.api.v1 import sanitize_message
from backend.app.engine.extension_store import ExtensionUserStore
from backend.app.engine.recommender import HybridRecommendationEngine
from backend.app.models.collaborative import ItemCooccurrenceCF, NeuralCollaborativeFiltering
from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import DualHorizonSessionTransformer


@pytest.fixture()
def synthetic_companion_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a real PyTorch-backed mini HybridRecommendationEngine + ExtensionUserStore for `/api/v1/*` testing."""
    proc_dir = tmp_path / "processed"
    art_dir = tmp_path / "artifacts"
    proc_dir.mkdir(parents=True)
    art_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    num_items = 60
    num_artists = 12
    num_genres = 6
    num_users = 8
    genres = ["indie", "electronic", "jazz", "rock", "ambient", "pop"]
    norm_cols = [f"norm_{i}" for i in range(12)]

    rows = []
    for i in range(num_items):
        a_idx = i % num_artists
        g_idx = i % num_genres
        row = {
            "item_idx": i,
            "spotify_track_id": f"sp_track_{i:04d}",
            "track_name": f"Track {i}",
            "primary_artist": f"Artist {a_idx}",
            "album_name": f"Album {a_idx}",
            "track_genre": genres[g_idx],
            "artist_idx": a_idx,
            "genre_idx": g_idx,
            "popularity": int(rng.integers(20, 90)),
            "duration_ms": 200_000 + i * 1000,
            "lfm_count": int(rng.integers(5, 500)),
            "danceability": float(rng.uniform(0.2, 0.9)),
            "energy": float(rng.uniform(0.2, 0.95)),
            "valence": float(rng.uniform(0.15, 0.9)),
            "tempo": float(rng.uniform(80.0, 155.0)),
            "acousticness": float(rng.uniform(0.05, 0.85)),
            "instrumentalness": float(rng.uniform(0.0, 0.7)),
            "speechiness": float(rng.uniform(0.03, 0.25)),
            "liveness": float(rng.uniform(0.05, 0.4)),
            "loudness": float(rng.uniform(-14.0, -4.0)),
            "key": int(i % 12),
            "mode": int(i % 2),
            "explicit": 0,
        }
        for nc in norm_cols:
            row[nc] = float(rng.normal(0, 1))
        rows.append(row)

    pd.DataFrame(rows).to_parquet(proc_dir / "tracks.parquet", index=False)

    # Playlists with repeated artists to stress-test Adaptive Shuffle spacing
    pl_rows = [
        {"pid": 1, "name": "Multi-Artist Test Mix", "item_indices": list(range(24))},
        {"pid": 2, "name": "Late Night Groove", "item_indices": list(range(12, 40))},
    ]
    pd.DataFrame(pl_rows).to_parquet(proc_dir / "playlists.parquet", index=False)

    inter_rows = []
    for u in range(num_users):
        base_g = u % num_genres
        preferred_pool = [i for i in range(num_items) if (i % num_genres) == base_g]
        other_pool = [i for i in range(num_items) if (i % num_genres) != base_g]
        seq = preferred_pool[:8] + other_pool[:6]
        for step, it in enumerate(seq):
            act = 4 if (it % num_genres) == base_g else 2
            inter_rows.append(
                {
                    "user_idx": u,
                    "user_id": f"u_{u}",
                    "country": "US",
                    "item_idx": it,
                    "action_id": act,
                    "reward": 1.0 if act == 4 else 0.55,
                    "unix_ts": 1_700_000_000 + step * 240,
                }
            )
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_parquet(proc_dir / "interactions.parquet", index=False)

    (proc_dir / "dataset_stats.json").write_text(
        json.dumps({"norm_audio_cols": norm_cols}), encoding="utf-8"
    )

    torch.manual_seed(42)
    tracks_df = pd.read_parquet(proc_dir / "tracks.parquet")
    enc = MultimodalTrackEncoder(
        num_items=num_items,
        num_artists=num_artists,
        num_genres=num_genres,
        audio_dim=12,
        embed_dim=128,
    )
    enc.set_catalog_metadata(
        torch.tensor(tracks_df["artist_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(tracks_df["genre_idx"].to_numpy(), dtype=torch.long),
        torch.tensor(tracks_df[norm_cols].to_numpy(dtype=np.float32)),
    )
    neumf = NeuralCollaborativeFiltering(num_users=num_users, track_encoder=enc, embed_dim=128)
    transformer = DualHorizonSessionTransformer(track_encoder=enc, max_seq_len=50, embed_dim=128)

    torch.save(neumf.state_dict(), art_dir / "neumf_checkpoint.pt")
    torch.save(transformer.state_dict(), art_dir / "transformer_checkpoint.pt")
    embs = enc.export_all_embeddings(device=torch.device("cpu")).numpy()
    np.save(art_dir / "track_embeddings.npy", embs)

    item_cf = ItemCooccurrenceCF(num_items).fit(
        (
            inter_df["user_idx"].to_numpy(dtype=np.int32),
            inter_df["item_idx"].to_numpy(dtype=np.int32),
            inter_df["reward"].to_numpy(dtype=np.float32),
        )
    )
    with open(art_dir / "item_cf.pkl", "wb") as f:
        pickle.dump(item_cf, f)

    fb_db = art_dir / "feedback_test.db"
    ext_db = art_dir / "extension_test.db"

    engine = HybridRecommendationEngine(
        processed_dir=proc_dir,
        artifacts_dir=art_dir,
        device="cpu",
        feedback_db_path=fb_db,
    )
    store = ExtensionUserStore(ext_db)

    monkeypatch.setattr(app_main, "ENGINE", engine)
    monkeypatch.setattr(app_main, "EXTENSION_STORE", store)

    return {
        "client": TestClient(app_main.app),
        "proc_dir": proc_dir,
        "art_dir": art_dir,
        "fb_db": fb_db,
        "ext_db": ext_db,
    }


def test_v1_health_auth_and_token_hashing_security(synthetic_companion_env) -> None:
    client: TestClient = synthetic_companion_env["client"]
    ext_db: Path = synthetic_companion_env["ext_db"]

    # 1. Health endpoint works without auth and reports loaded PyTorch models
    h = client.get("/api/v1/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["engine_loaded"] is True
    assert body["models"]["neumf"] is True
    assert body["models"]["dual_horizon_transformer"] is True

    # 2. Protected endpoints reject missing or invalid tokens
    assert client.get("/api/v1/preferences").status_code == 401
    assert client.get("/api/v1/preferences", headers={"X-Extension-Token": "invalid_tok"}).status_code == 401

    # 3. Create session & verify raw token is NEVER stored in SQLite (only SHA-256 hash)
    sess = client.post(
        "/api/v1/auth/session",
        json={
            "display_name": "Alice",
            "preferred_genres": ["jazz", "electronic"],
            "discovery_intensity": 0.55,
            "onboarding_completed": True,
        },
    )
    assert sess.status_code == 201
    raw_token = sess.json()["session_token"]
    assert raw_token.startswith("st_ext_")

    conn = sqlite3.connect(str(ext_db))
    try:
        stored_hashes = [r[0] for r in conn.execute("SELECT token_hash FROM extension_users").fetchall()]
    finally:
        conn.close()
    assert raw_token not in stored_hashes
    assert all(len(th) == 64 for th in stored_hashes)

    # 4. Token redaction utility strips tokens from error logs
    leaked = f"Failure with {raw_token} and Bearer secret_jwt_token_value"
    sanitized = sanitize_message(leaked)
    assert raw_token not in sanitized
    assert "secret_jwt_token_value" not in sanitized

    # 5. Chrome Extension CORS preflight check succeeds for valid 32-char extension ID
    cors_resp = client.options(
        "/api/v1/discovery",
        headers={
            "Origin": "chrome-extension://abcdefghijklmnopabcdefghijklmnop",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,X-Extension-Token",
        },
    )
    assert cors_resp.status_code == 200
    assert (
        cors_resp.headers.get("access-control-allow-origin")
        == "chrome-extension://abcdefghijklmnopabcdefghijklmnop"
    )


def test_v1_strict_user_isolation_and_restart_persistence(synthetic_companion_env, monkeypatch) -> None:
    client: TestClient = synthetic_companion_env["client"]

    u1 = client.post(
        "/api/v1/auth/session",
        json={"display_name": "User One", "preferred_genres": ["jazz"], "onboarding_completed": True},
    ).json()
    u2 = client.post(
        "/api/v1/auth/session",
        json={"display_name": "User Two", "preferred_genres": ["rock"], "onboarding_completed": True},
    ).json()

    h1 = {"X-Extension-Token": u1["session_token"]}
    h2 = {"X-Extension-Token": u2["session_token"]}

    # User 1 likes item 5 and saves strict artist spacing
    assert client.post("/api/v1/feedback", headers=h1, json={"item_idx": 5, "action": "like"}).status_code == 200
    assert (
        client.put(
            "/api/v1/shuffle/preferences",
            headers=h1,
            json={
                "artist_repetition": "strict",
                "genre_diversity": "high",
                "musical_coherence": "smooth",
                "discovery_mode": "exploratory",
                "randomness": "balanced",
                "quota_urgency_weight": 0.55,
            },
        ).status_code
        == 200
    )

    # User 2 dislikes item 5 and has zero likes
    assert client.post("/api/v1/feedback", headers=h2, json={"item_idx": 5, "action": "dislike"}).status_code == 200

    fb1 = client.get("/api/v1/feedback", headers=h1).json()["feedback_summary"]
    fb2 = client.get("/api/v1/feedback", headers=h2).json()["feedback_summary"]
    assert fb1["liked_count"] == 1 and fb1["disliked_count"] == 0
    assert fb2["liked_count"] == 0 and fb2["disliked_count"] == 1

    # Simulate complete backend restart by instantiating fresh Engine & Store over the same SQLite files
    new_engine = HybridRecommendationEngine(
        processed_dir=synthetic_companion_env["proc_dir"],
        artifacts_dir=synthetic_companion_env["art_dir"],
        device="cpu",
        feedback_db_path=synthetic_companion_env["fb_db"],
    )
    new_store = ExtensionUserStore(synthetic_companion_env["ext_db"])
    monkeypatch.setattr(app_main, "ENGINE", new_engine)
    monkeypatch.setattr(app_main, "EXTENSION_STORE", new_store)

    prefs1_after = client.get("/api/v1/preferences", headers=h1).json()["preferences"]
    assert prefs1_after["shuffle_presets"]["artist_repetition"] == "strict"
    assert prefs1_after["shuffle_presets"]["discovery_mode"] == "exploratory"
    assert client.get("/api/v1/feedback", headers=h1).json()["feedback_summary"]["liked_count"] == 1


def test_v1_discovery_shuffle_fallbacks_insights_and_data_deletion(synthetic_companion_env) -> None:
    client: TestClient = synthetic_companion_env["client"]
    sess = client.post(
        "/api/v1/auth/session",
        json={"display_name": "Explorer", "preferred_genres": ["indie", "electronic"], "onboarding_completed": True},
    ).json()
    headers = {"Authorization": f"Bearer {sess['session_token']}"}

    # 1. Discovery feed + Similar tracks + Similar artists
    disc = client.post("/api/v1/discovery", headers=headers, json={"top_k": 12}).json()
    assert len(disc["recommended_tracks"]) == 12
    first_track = disc["recommended_tracks"][0]
    assert len(first_track["explanations"]) >= 1
    assert first_track["spotify_url"].startswith("https://open.spotify.com/track/")

    sim_t = client.get(f"/api/v1/discovery/similar-tracks/{first_track['item_idx']}?top_k=5", headers=headers)
    assert sim_t.status_code == 200
    assert len(sim_t.json()["similar_tracks"]) == 5

    sim_a = client.get(
        f"/api/v1/discovery/similar-artists?artist={first_track['artist_name']}&top_k=4",
        headers=headers,
    )
    assert sim_a.status_code == 200
    assert len(sim_a.json()["similar_artists"]) == 4

    # 2. Rich feedback actions & single-negative stability check
    for act in ("more_like_this", "not_interested", "less_like_this", "genre_fatigue"):
        r = client.post(
            "/api/v1/feedback",
            headers=headers,
            json={"item_idx": 3, "genre": "rock", "action": act},
        )
        assert r.status_code == 200
        assert r.json()["feedback_summary"]["residual_magnitude"] <= 0.65

    # 3. Adaptive Shuffle on mixed catalog + out-of-catalog Spotify tracks (fallback test!)
    external_tracks = [
        {"spotify_track_id": "sp_track_0001", "track_name": "Track 1", "artist_name": "Artist 1"},
        {"spotify_track_id": "sp_track_0002", "track_name": "Track 2", "artist_name": "Artist 2"},
        {"spotify_track_id": "unseen_ext_1", "track_name": "Brand New Song A", "artist_name": "Brand New Artist"},
        {"spotify_track_id": "unseen_ext_2", "track_name": "Brand New Song B", "artist_name": "Brand New Artist"},
        {"spotify_track_id": "unseen_ext_3", "track_name": "Indie Sunrise", "artist_name": "Another Indie Band", "genre": "indie"},
        {"spotify_track_id": "sp_track_0005", "track_name": "Track 5", "artist_name": "Artist 5"},
    ]
    shuf = client.post(
        "/api/v1/shuffle",
        headers=headers,
        json={
            "source_type": "authorized_tracks",
            "source_name": "Mixed Authorized Playlist",
            "tracks": external_tracks,
            "seed": 99,
        },
    )
    assert shuf.status_code == 200
    shuf_data = shuf.json()
    assert shuf_data["pool_size"] == 6
    assert shuf_data["feature_coverage"]["catalog_audio_matched"] == 3
    assert shuf_data["feature_coverage"]["fallback_tracks"] == 3
    # Verify exact permutation preservation of all original tracks
    orig_names = sorted(t["track_name"] for t in shuf_data["original_tracks"])
    gen_names = sorted(t["track_name"] for t in shuf_data["generated_tracks"])
    assert orig_names == gen_names
    assert shuf_data["handoff"]["direct_spotify_shuffle_override"] is False
    assert "#EXTM3U" in shuf_data["handoff"]["m3u"]

    # 4. Insights endpoint separates Observed, Inferred, and Model Predictions
    ins = client.get("/api/v1/insights", headers=headers)
    assert ins.status_code == 200
    ins_body = ins.json()
    assert ins_body["observed"]["provenance"] == "observed"
    assert ins_body["inferred_preferences"]["provenance"] == "inferred_preference"
    assert ins_body["model_predictions"]["provenance"] == "model_prediction"

    # 5. Complete Privacy Data Deletion (`DELETE /api/v1/users/me`) invalidates token & wipes state
    del_resp = client.delete("/api/v1/users/me", headers=headers)
    assert del_resp.status_code == 200
    assert client.get("/api/v1/users/me", headers=headers).status_code == 401
