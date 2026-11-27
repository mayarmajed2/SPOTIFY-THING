"""
FastAPI `/api/v1/*` Companion Extension Router for SPOTIFY-THING.

Implements:
- `/api/v1/health`          : Engine readiness, model lifecycle status, catalog dimensions & independent mode flag
- `/api/v1/auth/*`          : Token-authenticated isolated user session creation, validation & logout
- `/api/v1/users/*`         : User profile retrieval & complete data deletion (privacy control)
- `/api/v1/preferences`     : Persistent onboarding, discovery, taste & Adaptive Shuffle Studio presets
- `/api/v1/discovery/*`     : Personalized discovery feed, similar tracks, similar artists, catalog genres/playlists
- `/api/v1/recommendations` : Direct personalized track ranking endpoint
- `/api/v1/shuffle/*`       : Adaptive Shuffle generation, algorithm comparison, fallback track handling & handoff exports
- `/api/v1/feedback/*`      : Rich feedback actions (Like, Dislike, Not interested, More/Less like this, Fatigue) + reset
- `/api/v1/insights`        : Strictly provenance-tagged Observed / Inferred / Model Prediction analytics
- `/api/v1/spotify/*`       : Capability-aware connection metadata & read-only inference track resolution
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from backend.app.engine.extension_store import ExtensionUserStore
from backend.app.engine.recommender import HybridRecommendationEngine
from backend.app.schemas.v1_models import (
    AdaptiveShuffleGenerateRequest,
    CreateSessionRequest,
    DiscoveryFeedRequest,
    ShufflePresetsSchema,
    SpotifyConnectionUpdateRequest,
    SpotifyResolveTracksRequest,
    UpdatePreferencesRequest,
    V1FeedbackRequest,
)

logger = logging.getLogger("spotify_thing.api.v1")

TOKEN_REDACT_RE = re.compile(r"(Bearer\s+[A-Za-z0-9_\-\.]+|st_ext_[A-Za-z0-9_\-]+)", re.IGNORECASE)


def sanitize_message(msg: str) -> str:
    """Ensure bearer or extension tokens are never leaked in error strings or logs."""
    return TOKEN_REDACT_RE.sub("[REDACTED_TOKEN]", str(msg))


class SlidingWindowRateLimiter:
    """Lightweight in-process per-client sliding-window rate limiter."""

    def __init__(self, max_requests: int = 180, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            dq = self._hits[client_key]
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please wait briefly before making more requests.",
                )
            dq.append(now)


def create_v1_router(
    get_engine_fn: Callable[[], HybridRecommendationEngine],
    get_store_fn: Callable[[], ExtensionUserStore],
    is_engine_ready_fn: Callable[[], bool],
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["companion-v1"])
    rate_limiter = SlidingWindowRateLimiter(max_requests=240, window_seconds=60.0)

    def enforce_rate_limit(request: Request) -> None:
        host = request.client.host if request.client else "local"
        rate_limiter.check(host)

    def get_current_user(
        request: Request,
        authorization: str | None = Header(default=None),
        x_extension_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        enforce_rate_limit(request)
        raw_token: str | None = None
        if authorization and authorization.lower().startswith("bearer "):
            raw_token = authorization[7:].strip()
        elif x_extension_token:
            raw_token = x_extension_token.strip()

        if not raw_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required (Authorization: Bearer <token> or X-Extension-Token).",
            )
        store = get_store_fn()
        user_rec = store.get_by_token(raw_token)
        if user_rec is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired extension session token.",
            )
        return user_rec

    # -------------------------------------------------------------------------
    # 1. Health & Model Lifecycle Status
    # -------------------------------------------------------------------------
    @router.get("/health")
    def v1_health(request: Request) -> dict[str, Any]:
        enforce_rate_limit(request)
        ready = is_engine_ready_fn()
        if not ready:
            return {
                "status": "degraded",
                "api_version": "1.0.0",
                "engine_loaded": False,
                "independent_mode_available": False,
                "models": {
                    "multimodal_encoder": False,
                    "neumf": False,
                    "dual_horizon_transformer": False,
                    "item_cf": False,
                    "adaptive_shuffle": True,
                    "linucb_bandit": True,
                },
            }
        eng = get_engine_fn()
        return {
            "status": "ok",
            "api_version": "1.0.0",
            "engine_loaded": True,
            "independent_mode_available": True,
            "device": str(eng.device),
            "catalog": {
                "num_tracks": eng.num_items,
                "num_artists": eng.num_artists,
                "num_genres": eng.num_genres,
                "num_playlists": len(eng.playlists_df),
            },
            "models": {
                "multimodal_encoder": True,
                "neumf": True,
                "dual_horizon_transformer": True,
                "item_cf": True,
                "adaptive_shuffle": True,
                "linucb_bandit": True,
            },
        }

    # -------------------------------------------------------------------------
    # 2. Authentication & Session Lifecycle
    # -------------------------------------------------------------------------
    @router.post("/auth/session", status_code=status.HTTP_201_CREATED)
    def create_extension_session(req: CreateSessionRequest, request: Request) -> dict[str, Any]:
        enforce_rate_limit(request)
        store = get_store_fn()
        anchor_idx = 0
        if is_engine_ready_fn() and (req.preferred_genres or req.preferred_artists):
            anchor_idx = get_engine_fn().resolve_anchor_user(req.preferred_genres, req.preferred_artists)

        raw_token, rec = store.create_session(
            display_name=req.display_name,
            anchor_user_idx=anchor_idx,
            initial_preferences={
                "preferred_genres": req.preferred_genres,
                "preferred_artists": req.preferred_artists,
                "discovery_intensity": req.discovery_intensity,
                "operating_mode": req.operating_mode,
                "onboarding_completed": req.onboarding_completed,
            },
        )
        return {
            "session_token": raw_token,
            "user": {
                "user_id": rec["user_id"],
                "display_name": rec["display_name"],
                "preferences": rec["preferences"],
                "spotify": rec["spotify"],
            },
        }

    @router.get("/auth/session")
    def validate_extension_session(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        return {
            "authenticated": True,
            "user": {
                "user_id": user["user_id"],
                "display_name": user["display_name"],
                "preferences": user["preferences"],
                "spotify": user["spotify"],
            },
        }

    @router.post("/auth/logout")
    def logout_and_revoke_session(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        store = get_store_fn()
        slot_idx = store.delete_user(user["user_id"])
        if slot_idx is not None and is_engine_ready_fn():
            get_engine_fn().reset_feedback_state(slot_idx)
        return {"status": "logged_out", "user_id": user["user_id"]}

    # -------------------------------------------------------------------------
    # 3. User Profile & Privacy Data Deletion
    # -------------------------------------------------------------------------
    @router.get("/users/me")
    def get_my_profile(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        fb_summary: dict[str, Any] = {"num_feedback_events": 0, "liked_count": 0, "disliked_count": 0}
        if is_engine_ready_fn():
            fb_summary = get_engine_fn().get_feedback_state(user["feedback_slot_idx"]).summary()
            fb_summary.pop("user_idx", None)
        return {
            "user_id": user["user_id"],
            "display_name": user["display_name"],
            "preferences": user["preferences"],
            "spotify": user["spotify"],
            "feedback_summary": fb_summary,
            "created_at": user["created_at"],
            "updated_at": user["updated_at"],
        }

    @router.delete("/users/me")
    def delete_my_account_and_data(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        store = get_store_fn()
        slot_idx = store.delete_user(user["user_id"])
        if slot_idx is not None and is_engine_ready_fn():
            get_engine_fn().reset_feedback_state(slot_idx)
        return {
            "status": "deleted",
            "user_id": user["user_id"],
            "deleted_items": ["session_token", "preferences", "spotify_connection_metadata", "bandit_feedback_state"],
        }

    # -------------------------------------------------------------------------
    # 4. User Preferences
    # -------------------------------------------------------------------------
    @router.get("/preferences")
    def get_user_preferences(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        genres = get_engine_fn().available_genres_sorted if is_engine_ready_fn() else []
        return {
            "user_id": user["user_id"],
            "preferences": user["preferences"],
            "available_catalog_genres": genres,
        }

    @router.put("/preferences")
    def update_user_preferences(
        req: UpdatePreferencesRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        store = get_store_fn()
        updates = req.model_dump(exclude_unset=True)
        new_anchor: int | None = None
        if is_engine_ready_fn() and ("preferred_genres" in updates or "preferred_artists" in updates):
            g_list = updates.get("preferred_genres")
            if g_list is None:
                g_list = user["preferences"].get("preferred_genres", [])
            a_list = updates.get("preferred_artists")
            if a_list is None:
                a_list = user["preferences"].get("preferred_artists", [])
            new_anchor = get_engine_fn().resolve_anchor_user(g_list, a_list)

        updated = store.update_preferences(
            user_id=user["user_id"],
            updates=updates,
            anchor_user_idx=new_anchor,
            display_name=req.display_name,
        )
        return {
            "status": "updated",
            "user_id": updated["user_id"],
            "display_name": updated["display_name"],
            "preferences": updated["preferences"],
        }

    # -------------------------------------------------------------------------
    # 5. Discovery & Personalized Recommendations
    # -------------------------------------------------------------------------
    @router.post("/discovery")
    def generate_discovery_session(
        req: DiscoveryFeedRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        prefs = user["preferences"]
        disc = req.discovery_intensity if req.discovery_intensity is not None else prefs.get("discovery_intensity", 0.40)
        div = req.diversity_level if req.diversity_level is not None else prefs.get("diversity_level", 0.50)
        nov = req.novelty_boost if req.novelty_boost is not None else prefs.get("novelty_boost", 0.30)
        excluded = list(set((prefs.get("dismissed_item_indices") or []) + req.exclude_item_indices))

        return eng.build_discovery_feed(
            feedback_slot_idx=user["feedback_slot_idx"],
            anchor_user_idx=user["anchor_user_idx"],
            preferred_genres=prefs.get("preferred_genres"),
            preferred_artists=prefs.get("preferred_artists"),
            top_k=req.top_k,
            discovery_intensity=float(disc),
            diversity_level=float(div),
            novelty_boost=float(nov),
            short_term_weight=prefs.get("short_term_weight"),
            exclude_item_indices=excluded,
            genre_filter=req.genre_filter,
        )

    @router.post("/recommendations")
    def get_personalized_recommendations(
        req: DiscoveryFeedRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        feed = generate_discovery_session(req, user)
        return {
            "user_id": user["user_id"],
            "recommendations": feed["recommended_tracks"],
            "familiarity_vs_discovery": feed["familiarity_vs_discovery"],
            "feedback_summary": feed["feedback_summary"],
        }

    @router.get("/discovery/similar-tracks/{item_idx}")
    def get_similar_tracks_endpoint(
        item_idx: int,
        top_k: int = Query(default=10, ge=2, le=30),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        fb_state = eng.get_feedback_state(user["feedback_slot_idx"])
        try:
            return eng.get_similar_tracks(item_idx, top_k=top_k, exclude_ids=fb_state.disliked_items)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=sanitize_message(str(exc))) from exc

    @router.get("/discovery/similar-artists")
    def get_similar_artists_endpoint(
        artist: str = Query(..., min_length=1, max_length=160),
        top_k: int = Query(default=8, ge=2, le=20),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        try:
            return eng.get_similar_artists(artist, top_k=top_k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=sanitize_message(str(exc))) from exc

    @router.get("/discovery/catalog-playlists")
    def list_companion_playlists(
        limit: int = Query(default=25, ge=5, le=80),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        rows = []
        for row in eng.playlists_df.head(limit * 3).itertuples(index=False):
            idxs = [int(i) for i in row.item_indices[:60] if 0 <= int(i) < eng.num_items]
            if len(idxs) < 12:
                continue
            arts = [eng.artist_names[i] for i in idxs]
            gens = [eng.genre_names[i] for i in idxs]
            unique_ratio = len(set(arts)) / len(arts)
            rows.append(
                {
                    "pid": int(row.pid),
                    "name": str(row.name),
                    "num_tracks": len(idxs),
                    "unique_artists": len(set(arts)),
                    "artist_concentration": round(1.0 - unique_ratio, 3),
                    "sample_genres": list(dict.fromkeys(gens))[:3],
                    "sample_artists": list(dict.fromkeys(arts))[:4],
                }
            )
        rows.sort(key=lambda r: (0 if 0.18 <= r["artist_concentration"] <= 0.45 else 1, -r["artist_concentration"]))
        return {"playlists": rows[:limit]}

    @router.get("/discovery/search")
    def search_companion_catalog(
        q: str = Query(..., min_length=1, max_length=120),
        limit: int = Query(default=15, ge=1, le=50),
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        ql = q.lower().strip()
        df = eng.tracks_df
        mask = (
            df["track_name"].str.lower().str.contains(ql, na=False, regex=False)
            | df["primary_artist"].str.lower().str.contains(ql, na=False, regex=False)
            | df["track_genre"].str.lower().str.contains(ql, na=False, regex=False)
        )
        hits = df[mask].sort_values("popularity", ascending=False).head(limit)
        return {
            "results": [
                {
                    "item_idx": int(r.item_idx),
                    "spotify_track_id": str(r.spotify_track_id),
                    "spotify_url": f"https://open.spotify.com/track/{r.spotify_track_id}",
                    "track_name": str(r.track_name),
                    "artist_name": str(r.primary_artist),
                    "album_name": str(r.album_name),
                    "genre": str(r.track_genre),
                    "popularity": int(r.popularity),
                    "duration_ms": int(r.duration_ms),
                }
                for r in hits.itertuples(index=False)
            ]
        }

    # -------------------------------------------------------------------------
    # 6. Adaptive Shuffle Studio
    # -------------------------------------------------------------------------
    @router.post("/shuffle")
    def generate_adaptive_shuffle(
        req: AdaptiveShuffleGenerateRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        store = get_store_fn()
        prefs = user["preferences"]
        active_presets = dict(prefs.get("shuffle_presets") or {})
        if req.presets is not None:
            active_presets.update(req.presets.model_dump())
            if req.save_presets:
                store.update_preferences(user["user_id"], {"shuffle_presets": active_presets})

        try:
            ext_tracks = [t.model_dump() for t in req.tracks] if req.tracks else None
            return eng.generate_companion_shuffle(
                feedback_slot_idx=user["feedback_slot_idx"],
                anchor_user_idx=user["anchor_user_idx"],
                preferred_genres=prefs.get("preferred_genres"),
                source_type=req.source_type,
                playlist_pid=req.playlist_pid,
                source_name=req.source_name,
                pool_size=req.pool_size,
                external_tracks=ext_tracks,
                presets=active_presets,
                include_baselines=req.include_baselines,
                seed=req.seed,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=sanitize_message(str(exc))) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=sanitize_message(str(exc))) from exc

    @router.put("/shuffle/preferences")
    def save_shuffle_preferences(
        presets: ShufflePresetsSchema,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        updated = get_store_fn().update_preferences(
            user["user_id"], {"shuffle_presets": presets.model_dump()}
        )
        return {
            "status": "saved",
            "shuffle_presets": updated["preferences"]["shuffle_presets"],
        }

    # -------------------------------------------------------------------------
    # 7. Feedback & Personalization Adaptation
    # -------------------------------------------------------------------------
    @router.post("/feedback")
    def submit_companion_feedback(
        req: V1FeedbackRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        eng = get_engine_fn()
        store = get_store_fn()
        slot_idx = user["feedback_slot_idx"]

        resolved_item_idx = req.item_idx
        if resolved_item_idx is None and req.spotify_track_id:
            resolved_item_idx = eng.spotify_id_to_idx.get(req.spotify_track_id.strip())

        # Support artist-level or genre-level fatigue even when no specific item_idx is passed
        if resolved_item_idx is None:
            if req.action == "artist_fatigue" and req.artist_name:
                a_id = eng.artist_name_to_idx.get(req.artist_name.lower().strip())
                if a_id is not None:
                    resolved_item_idx = eng.artist_top_track.get(a_id)
            elif req.action == "genre_fatigue" and req.genre:
                g_id = eng.genre_name_to_idx.get(req.genre.lower().strip())
                if g_id is not None:
                    matches = [i for i, g in enumerate(eng.genre_ids) if int(g) == g_id]
                    if matches:
                        resolved_item_idx = matches[0]

        if resolved_item_idx is None or not (0 <= resolved_item_idx < eng.num_items):
            raise HTTPException(
                status_code=404,
                detail="Target track/artist/genre could not be resolved in the independent catalog.",
            )

        ev, fb_state = eng.record_user_feedback(
            user_idx=slot_idx,
            item_idx=int(resolved_item_idx),
            action=req.action,
            played_ratio=req.played_ratio,
        )

        if req.action in ("not_interested", "dislike", "less_like_this"):
            dismissed = list(user["preferences"].get("dismissed_item_indices") or [])
            if int(resolved_item_idx) not in dismissed:
                dismissed.append(int(resolved_item_idx))
                store.update_preferences(user["user_id"], {"dismissed_item_indices": dismissed[-300:]})

        row = eng.tracks_df.iloc[int(resolved_item_idx)]
        summary = fb_state.summary()
        summary.pop("user_idx", None)
        return {
            "status": "recorded",
            "event": {
                "item_idx": int(ev.item_idx),
                "spotify_track_id": str(row["spotify_track_id"]),
                "track_name": str(row["track_name"]),
                "artist_name": str(row["primary_artist"]),
                "genre": str(row["track_genre"]),
                "action": ev.action,
                "reward": round(ev.reward, 3),
            },
            "feedback_summary": summary,
        }

    @router.get("/feedback")
    def get_feedback_state_endpoint(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        eng = get_engine_fn()
        fb_state = eng.get_feedback_state(user["feedback_slot_idx"])
        summary = fb_state.summary()
        summary.pop("user_idx", None)
        recent = []
        for ev in reversed(fb_state.events[-25:]):
            if 0 <= ev.item_idx < eng.num_items:
                row = eng.tracks_df.iloc[ev.item_idx]
                recent.append(
                    {
                        "item_idx": int(ev.item_idx),
                        "spotify_track_id": str(row["spotify_track_id"]),
                        "track_name": str(row["track_name"]),
                        "artist_name": str(row["primary_artist"]),
                        "genre": str(row["track_genre"]),
                        "action": ev.action,
                        "reward": round(ev.reward, 3),
                        "timestamp": ev.timestamp,
                    }
                )
        return {
            "user_id": user["user_id"],
            "feedback_summary": summary,
            "recent_events": recent,
        }

    @router.post("/feedback/reset")
    def reset_companion_feedback(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        eng = get_engine_fn()
        eng.reset_feedback_state(user["feedback_slot_idx"])
        get_store_fn().update_preferences(user["user_id"], {"dismissed_item_indices": []})
        return {"status": "reset", "user_id": user["user_id"]}

    # -------------------------------------------------------------------------
    # 8. Music Insights (Observed vs. Inferred vs. Model Predictions)
    # -------------------------------------------------------------------------
    @router.get("/insights")
    def get_music_insights(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        eng = get_engine_fn()
        return eng.compute_companion_insights(
            feedback_slot_idx=user["feedback_slot_idx"],
            anchor_user_idx=user["anchor_user_idx"],
            preferences=user["preferences"],
        )

    # -------------------------------------------------------------------------
    # 9. Capability-Aware Spotify Integration Metadata & Inference Track Resolver
    # -------------------------------------------------------------------------
    @router.get("/spotify/status")
    def get_spotify_integration_status(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        return {
            "user_id": user["user_id"],
            "spotify": user["spotify"],
            "policy_compliance": {
                "oauth_flow": "authorization_code_pkce_s256",
                "stores_spotify_tokens_on_backend": False,
                "trains_ml_on_spotify_content": False,
                "scrapes_spotify_dom": False,
                "restricted_endpoints_avoided": [
                    "GET /v1/audio-features",
                    "GET /v1/audio-analysis",
                    "GET /v1/recommendations",
                    "GET /v1/artists/{id}/related-artists",
                    "POST /v1/users/{user_id}/playlists",
                ],
            },
        }

    @router.put("/spotify/status")
    def update_spotify_integration_status(
        req: SpotifyConnectionUpdateRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        updated = get_store_fn().update_spotify_state(user["user_id"], req.model_dump())
        return {
            "status": "updated",
            "spotify": updated["spotify"],
        }

    @router.post("/spotify/disconnect")
    def disconnect_spotify_integration(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
        updated = get_store_fn().update_spotify_state(
            user["user_id"],
            {
                "connected": False,
                "account_label": None,
                "product_tier": None,
                "granted_scopes": [],
                "capabilities": {
                    "can_read_profile": False,
                    "can_read_playlists": False,
                    "can_read_playback_state": False,
                    "can_create_playlists": False,
                    "direct_queue_reorder_supported": False,
                    "native_shuffle_override_supported": False,
                },
            },
        )
        return {"status": "disconnected", "spotify": updated["spotify"]}

    @router.post("/spotify/resolve-tracks")
    def resolve_authorized_tracks_for_inference(
        req: SpotifyResolveTracksRequest,
        user: dict[str, Any] = Depends(get_current_user),
    ) -> dict[str, Any]:
        """Transiently map authorized track identifiers against independent catalog for read-only inference."""
        eng = get_engine_fn()
        resolved = eng.resolve_external_tracks([t.model_dump() for t in req.tracks])
        matched = sum(1 for r in resolved if r["matched_catalog"])
        return {
            "total_submitted": len(resolved),
            "matched_in_independent_catalog": matched,
            "fallback_count": len(resolved) - matched,
            "tracks": [
                {
                    "pool_index": r["pool_index"],
                    "matched_catalog": r["matched_catalog"],
                    "feature_source": r["feature_source"],
                    "item_idx": r["item_idx"],
                    "spotify_track_id": r["spotify_track_id"],
                    "track_name": r["track_name"],
                    "artist_name": r["artist_name"],
                    "genre": r["genre"],
                }
                for r in resolved
            ],
        }

    return router
