"""Pydantic Request/Response Contracts for the FastAPI Serving Layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    user_idx: int = Field(default=0, ge=0)
    top_k: int = Field(default=24, ge=4, le=80)
    discovery_level: float = Field(default=0.40, ge=0.0, le=1.0)
    diversity_level: float = Field(default=0.50, ge=0.0, le=1.0)
    novelty_boost: float = Field(default=0.30, ge=0.0, le=1.0)
    short_term_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class ShuffleCompareRequest(BaseModel):
    user_idx: int = Field(default=0, ge=0)
    playlist_pid: int | None = Field(default=None)
    pool_size: int = Field(default=32, ge=10, le=80)
    diversity_weight: float = Field(default=0.65, ge=0.0, le=1.5)
    flow_weight: float = Field(default=0.45, ge=0.0, le=1.5)
    relevance_weight: float = Field(default=0.55, ge=0.0, le=1.5)
    artist_cooldown: int = Field(default=4, ge=0, le=12)
    genre_cooldown: int = Field(default=2, ge=0, le=8)
    entropy_temperature: float = Field(default=0.18, ge=0.02, le=1.0)
    discovery_target: float = Field(default=0.35, ge=0.0, le=1.0)
    quota_urgency_weight: float = Field(default=0.0, ge=0.0, le=1.5)
    seed: int | None = Field(default=None)


class FeedbackRequest(BaseModel):
    user_idx: int = Field(..., ge=0)
    item_idx: int = Field(..., ge=0)
    action: str = Field(
        ...,
        description="One of: like, dislike, early_skip, partial_play, complete_play, repeat",
    )
    played_ratio: float = Field(default=1.0, ge=0.0, le=1.5)


class ResetFeedbackRequest(BaseModel):
    user_idx: int = Field(..., ge=0)
