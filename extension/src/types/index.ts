export type ArtistRepetitionPreset = "strict" | "balanced" | "flexible";
export type GenreDiversityPreset = "low" | "medium" | "high";
export type MusicalCoherencePreset = "smooth" | "balanced" | "variety";
export type DiscoveryModePreset = "familiar" | "balanced" | "exploratory";
export type RandomnessPreset = "predictable" | "balanced" | "varied";

export type FeedbackAction =
  | "like"
  | "dislike"
  | "not_interested"
  | "more_like_this"
  | "less_like_this"
  | "artist_fatigue"
  | "genre_fatigue"
  | "early_skip"
  | "partial_play"
  | "complete_play"
  | "repeat";

export interface ShufflePresets {
  artist_repetition: ArtistRepetitionPreset;
  genre_diversity: GenreDiversityPreset;
  musical_coherence: MusicalCoherencePreset;
  discovery_mode: DiscoveryModePreset;
  randomness: RandomnessPreset;
  quota_urgency_weight: number;
}

export interface UserPreferences {
  onboarding_completed: boolean;
  operating_mode: "independent" | "spotify_connected";
  preferred_genres: string[];
  preferred_artists: string[];
  discovery_intensity: number;
  diversity_level: number;
  novelty_boost: number;
  short_term_weight: number | null;
  shuffle_presets: ShufflePresets;
  dismissed_item_indices: number[];
}

export interface SpotifyCapabilities {
  can_read_profile: boolean;
  can_read_playlists: boolean;
  can_read_playback_state: boolean;
  can_create_playlists: boolean;
  direct_queue_reorder_supported: boolean;
  native_shuffle_override_supported: boolean;
}

export interface SpotifyConnectionState {
  connected: boolean;
  mode: "independent" | "spotify_connected";
  account_label: string | null;
  product_tier: string | null;
  granted_scopes: string[];
  capabilities: SpotifyCapabilities;
  last_verified_at: number | null;
}

export interface ExtensionUser {
  user_id: string;
  display_name: string;
  preferences: UserPreferences;
  spotify: SpotifyConnectionState;
}

export interface TrackAudioSummary {
  danceability: number;
  energy: number;
  valence: number;
  tempo: number;
  acousticness?: number;
  instrumentalness?: number;
}

export interface RecommendedTrack {
  item_idx: number;
  spotify_track_id: string;
  spotify_url: string;
  spotify_uri: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
  popularity: number;
  duration_ms: number;
  is_discovery: boolean;
  novelty: number;
  score: number;
  short_term_affinity: number;
  long_term_affinity: number;
  audio: TrackAudioSummary;
  explanations: string[];
}

export interface RecommendedArtist {
  artist_idx: number;
  artist_name: string;
  genre: string;
  catalog_track_count: number;
  affinity_score: number;
  is_discovery: boolean;
  representative_track: {
    item_idx: number;
    spotify_track_id: string;
    track_name: string;
    spotify_url: string;
  };
  reason: string;
}

export interface RelatedGenre {
  genre_idx: number;
  genre: string;
  affinity_score: number;
  track_count: number;
  is_exploratory: boolean;
}

export interface DiscoveryFeedResponse {
  recommended_tracks: RecommendedTrack[];
  new_discoveries: RecommendedTrack[];
  recommended_artists: RecommendedArtist[];
  related_genres: RelatedGenre[];
  familiarity_vs_discovery: {
    target_discovery_ratio: number;
    actual_discovery_ratio: number;
    familiar_count: number;
    discovery_count: number;
    long_term_gate: number;
    short_term_gate: number;
  };
  feedback_summary: {
    num_feedback_events: number;
    liked_count: number;
    disliked_count: number;
    skipped_count: number;
    residual_magnitude: number;
    fatigued_artists: number;
    fatigued_genres: number;
  };
}

export interface SimilarTrackItem {
  item_idx: number;
  spotify_track_id: string;
  spotify_url: string;
  spotify_uri: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
  duration_ms: number;
  similarity: number;
  reason: string;
}

export interface SimilarArtistItem {
  artist_idx: number;
  artist_name: string;
  genre: string;
  similarity: number;
  catalog_track_count: number;
  sample_track: {
    item_idx: number;
    spotify_track_id: string;
    track_name: string;
    spotify_url: string;
  };
}

export interface CatalogPlaylistSummary {
  pid: number;
  name: string;
  num_tracks: number;
  unique_artists: number;
  artist_concentration: number;
  sample_genres: string[];
  sample_artists: string[];
}

export interface ShuffleTrackEntry {
  position: number;
  original_position: number;
  item_idx: number | null;
  spotify_track_id: string;
  spotify_url: string | null;
  spotify_uri: string | null;
  track_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
  duration_ms: number;
  is_discovery: boolean;
  feature_source: "independent_catalog" | "artist_catalog_prior" | "metadata_only_fallback";
  audio: TrackAudioSummary | null;
  transition: {
    step: number;
    relevance?: number;
    flow?: number;
    diversity?: number;
    penalty?: number;
    reason: string;
  } | null;
}

export interface SequenceMetrics {
  intra_list_diversity: number;
  local_window_diversity: number;
  min_artist_gap: number;
  median_artist_gap?: number;
  mean_artist_gap: number;
  artist_clump_rate: number;
  genre_switch_rate: number;
  window_genre_diversity?: number;
  acoustic_smoothness: number;
  mean_energy_jump: number;
  mean_bpm_jump: number;
  session_ndcg?: number;
  discovery_pacing_std: number;
  sequence_validity?: number;
}

export interface AdaptiveShuffleResponse {
  source_name: string;
  source_type: "catalog_playlist" | "discovery_pool" | "authorized_tracks";
  pool_size: number;
  seed: number;
  applied_config: {
    presets: ShufflePresets;
    diversity_weight: number;
    flow_weight: number;
    artist_cooldown: number;
    genre_cooldown: number;
    entropy_temperature: number;
    discovery_target: number;
    quota_urgency_weight: number;
  };
  feature_coverage: {
    catalog_audio_matched: number;
    fallback_tracks: number;
    audio_coverage_ratio: number;
    note: string;
  };
  original_tracks: ShuffleTrackEntry[];
  generated_tracks: ShuffleTrackEntry[];
  metrics: SequenceMetrics;
  baselines: {
    original_order?: SequenceMetrics;
    fisher_yates?: SequenceMetrics;
    artist_spaced?: SequenceMetrics;
  };
  distributions: {
    artists: { artist: string; count: number; share: number }[];
    genres: { genre: string; count: number; share: number }[];
    discovery: {
      familiar_count: number;
      discovery_count: number;
      discovery_ratio: number;
    };
    repetition: {
      unique_artists: number;
      repeated_artists_count: number;
      original_clump_rate: number;
      generated_clump_rate: number;
      min_artist_gap: number;
      mean_artist_gap: number;
    };
  };
  handoff: {
    direct_spotify_shuffle_override: boolean;
    limitation_notice: string;
    spotify_uris: string[];
    spotify_urls: string[];
    m3u: string;
    csv: string;
  };
}

export interface CompanionInsightsResponse {
  observed: {
    provenance: "observed";
    description: string;
    total_feedback_actions: number;
    positive_actions: number;
    negative_actions: number;
    liked_tracks_count: number;
    dismissed_or_disliked_count: number;
    explicit_preferred_genres: string[];
    explicit_preferred_artists: string[];
    observed_top_genres: { genre: string; positive_signals: number }[];
    observed_top_artists: { artist: string; positive_signals: number }[];
  };
  inferred_preferences: {
    provenance: "inferred_preference";
    description: string;
    bandit_residual_magnitude: number;
    preference_drift_score: number;
    long_term_horizon_weight: number;
    short_term_session_weight: number;
    configured_discovery_intensity: number;
    boosted_genres: { genre: string; boost_weight: number }[];
    fatigued_genres: { genre: string; fatigue_penalty: number }[];
    boosted_artists: { artist: string; boost_weight: number }[];
    fatigued_artists: { artist: string; fatigue_penalty: number }[];
  };
  model_predictions: {
    provenance: "model_prediction";
    description: string;
    predicted_favorite_genres: RelatedGenre[];
    predicted_favorite_artists: {
      artist: string;
      genre: string;
      affinity_score: number;
      is_discovery: boolean;
    }[];
    recommendation_diversity_ild: number;
    unique_artists_in_top20: number;
    repeated_artist_exposure_rate: number;
    familiarity_vs_discovery: DiscoveryFeedResponse["familiarity_vs_discovery"];
    predicted_sonic_profile: Record<string, number>;
  };
}

export interface BackendHealthResponse {
  status: "ok" | "degraded";
  api_version: string;
  engine_loaded: boolean;
  independent_mode_available: boolean;
  device?: string;
  catalog?: {
    num_tracks: number;
    num_artists: number;
    num_genres: number;
    num_playlists: number;
  };
  models: {
    multimodal_encoder: boolean;
    neumf: boolean;
    dual_horizon_transformer: boolean;
    item_cf: boolean;
    adaptive_shuffle: boolean;
    linucb_bandit: boolean;
  };
}
