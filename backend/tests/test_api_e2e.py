"""End-to-End API Integration Tests against Trained Artifacts."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from backend.app.main import app
from backend.app.main import ARTIFACTS_DIR


def test_api_reports_missing_artifacts_as_service_unavailable() -> None:
    required_artifacts = (
        "neumf_checkpoint.pt",
        "transformer_checkpoint.pt",
        "track_embeddings.npy",
        "item_cf.pkl",
    )
    if all((ARTIFACTS_DIR / name).exists() for name in required_artifacts):
        pytest.skip("Required model artifacts are present; this check is for incomplete deployments.")

    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]


def test_full_api_lifecycle() -> None:
    required_artifacts = (
        "neumf_checkpoint.pt",
        "transformer_checkpoint.pt",
        "track_embeddings.npy",
        "item_cf.pkl",
    )
    missing = [name for name in required_artifacts if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        pytest.skip(
            "Generated model artifacts are absent: "
            + ", ".join(missing)
            + ". Build and train the project before running the API lifecycle test."
        )

    with TestClient(app) as client:
        h = client.get("/api/health")
        assert h.status_code == 200
        h_json = h.json()
        assert h_json["num_tracks"] > 0
        assert h_json["num_users"] > 0

        users_res = client.get("/api/users?limit=5")
        assert users_res.status_code == 200
        uid = users_res.json()["users"][0]["user_idx"]

        prof_res = client.get(f"/api/users/{uid}/profile")
        assert prof_res.status_code == 200
        assert "long_term_radar" in prof_res.json()

        rec_res = client.post(
            "/api/recommend",
            json={
                "user_idx": uid,
                "top_k": 12,
                "discovery_level": 0.5,
                "diversity_level": 0.6,
                "novelty_boost": 0.3,
                "short_term_weight": None,
            },
        )
        assert rec_res.status_code == 200
        recs = rec_res.json()["recommendations"]
        assert len(recs) == 12
        first_item = recs[0]["item_idx"]

        # Record live early skip & like feedback
        fb_res = client.post(
            "/api/feedback",
            json={"user_idx": uid, "item_idx": first_item, "action": "early_skip", "played_ratio": 0.08},
        )
        assert fb_res.status_code == 200
        assert fb_res.json()["feedback_summary"]["skipped_count"] >= 1

        # Compare shuffle modes
        shuf_res = client.post(
            "/api/shuffle/compare",
            json={
                "user_idx": uid,
                "playlist_pid": None,
                "pool_size": 24,
                "diversity_weight": 0.75,
                "flow_weight": 0.55,
                "relevance_weight": 0.55,
                "artist_cooldown": 4,
                "genre_cooldown": 2,
                "entropy_temperature": 0.18,
                "discovery_target": 0.4,
            },
        )
        assert shuf_res.status_code == 200
        s_json = shuf_res.json()
        assert len(s_json["adaptive"]["tracks"]) == 24
        assert len(s_json["fisher_yates"]["tracks"]) == 24
        assert len(s_json["artist_spaced"]["tracks"]) == 24

        # Search catalog & benchmarks
        srch = client.get("/api/catalog/search?q=radiohead")
        assert srch.status_code == 200

        bm = client.get("/api/benchmarks")
        assert bm.status_code == 200
        assert "Proposed Hybrid + Discovery" in bm.json()["recommendation_benchmarks"]
