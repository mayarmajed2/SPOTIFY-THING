"""Typed Pydantic Request & Response Contracts for `/api/v1/*` Companion Extension API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ArtistRepetitionPreset = Literal["strict", "balanced", "flexible"]
GenreDiversityPreset = Literal["low", "medium", "high"]
MusicalCoherencePreset = Literal["smooth", "balanced", "variety"]
DiscoveryModePreset = Literal["familiar", "balanced", "exploratory"]
RandomnessPreset = Literal["predictable", "balanced", "varied"]

FeedbackActionType = Literal[
    "like",
    "dislike",
    "not_interested",
    "more_like_this",
    "less_like_this",
    "artist_fatigue",
    "genre_fatigue",
    "early_skip",
    "partial_play",
    "complete_play",
    "repeat",
]


class ShufflePresetsSchema(BaseModel):
    artist_repetition: ArtistRepetitionPreset = "balanced"
    genre_diversity: GenreDiversityPreset = "medium"
    musical_coherence: MusicalCoherencePreset = "smooth"
    discovery_mode: DiscoveryModePreset = "balanced"
    randomness: RandomnessPreset = "balanced"
    quota_urgency_weight: float = Field(default=0.0, ge=0.0, le=1.5)


class CreateSessionRequest(BaseModel):
    display_name: str = Field(default="Spotify Companion Listener", min_length=1, max_length=80)
    preferred_genres: list[str] = Field(default_factory=list, max_length=20)
    preferred_artists: list[str] = Field(default_factory=list, max_length=20)
    discovery_intensity: float = Field(default=0.40, ge=0.0, le=1.0)
    operating_mode: Literal["independent", "spotify_connected"] = "independent"
    onboarding_completed: bool = False


class UpdatePreferencesRequest(BaseModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=80)
    onboarding_completed: bool | None = None
    operating_mode: Literal["independent", "spotify_connected"] | None = None
    preferred_genres: list[str] | None = Field(default=None, max_length=25)
    preferred_artists: list[str] | None = Field(default=None, max_length=25)
    discovery_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    diversity_level: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty_boost: float | None = Field(default=None, ge=0.0, le=1.0)
    short_term_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    shuffle_presets: ShufflePresetsSchema | None = None


class DiscoveryFeedRequest(BaseModel):
    top_k: int = Field(default=18, ge=4, le=60)
    discovery_intensity: float | None = Field(default=None, ge=0.0, le=1.0)
    diversity_level: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty_boost: float | None = Field(default=None, ge=0.0, le=1.0)
    exclude_item_indices: list[int] = Field(default_factory=list, max_length=200)
    genre_filter: str | None = Field(default=None, max_length=60)


class ExternalTrackInput(BaseModel):
    spotify_track_id: str | None = Field(default=None, max_length=64)
    track_name: str = Field(..., min_length=1, max_length=240)
    artist_name: str = Field(..., min_length=1, max_length=200)
    album_name: str = Field(default="Unknown Album", max_length=240)
    duration_ms: int = Field(default=210000, ge=1000, le=3600000)
    genre: str | None = Field(default=None, max_length=80)
    artwork_url: str | None = Field(default=None, max_length=512)


class AdaptiveShuffleGenerateRequest(BaseModel):
    source_type: Literal["catalog_playlist", "discovery_pool", "authorized_tracks"] = "catalog_playlist"
    playlist_pid: int | None = None
    source_name: str | None = Field(default=None, max_length=160)
    pool_size: int = Field(default=28, ge=6, le=80)
    tracks: list[ExternalTrackInput] = Field(default_factory=list, max_length=120)
    presets: ShufflePresetsSchema | None = None
    save_presets: bool = False
    include_baselines: bool = True
    seed: int | None = None


class V1FeedbackRequest(BaseModel):
    item_idx: int | None = Field(default=None, ge=0)
    spotify_track_id: str | None = Field(default=None, max_length=64)
    artist_name: str | None = Field(default=None, max_length=200)
    genre: str | None = Field(default=None, max_length=80)
    action: FeedbackActionType
    played_ratio: float = Field(default=1.0, ge=0.0, le=1.5)


class SpotifyConnectionUpdateRequest(BaseModel):
    connected: bool
    account_label: str | None = Field(default=None, max_length=80)
    product_tier: str | None = Field(default=None, max_length=40)
    granted_scopes: list[str] = Field(default_factory=list, max_length=20)
    capabilities: dict[str, bool] = Field(default_factory=dict)


class SpotifyResolveTracksRequest(BaseModel):
    tracks: list[ExternalTrackInput] = Field(..., min_length=1, max_length=120)
