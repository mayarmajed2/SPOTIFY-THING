export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api";

export interface UserPersona {
  user_idx: number;
  user_id: string;
  country: string;
  total_events: number;
  historical_skip_rate: number;
  top_genres: string[];
  top_artists: string[];
}

export interface TimelineEvent {
  item_idx: number;
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  genre: string;
  action: string;
  reward?: number;
  source: "historical" | "live_feedback";
}

export interface FeedbackSummary {
  user_idx: number;
  num_feedback_events: number;
  liked_count: number;
  disliked_count: number;
  skipped_count: number;
  residual_magnitude: number;
  fatigued_artists: number;
  fatigued_genres: number;
}

export interface UserProfile {
  user_idx: number;
  user_id: string;
  country: string;
  long_term_gate: number;
  short_term_gate: number;
  long_term_radar: Record<string, number>;
  short_term_radar: Record<string, number>;
  genre_breakdown: { genre: string; share: number }[];
  feedback_summary: FeedbackSummary;
  recent_timeline: TimelineEvent[];
}

export interface RecommendedTrack {
  item_idx: number;
  spotify_track_id: string;
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
  audio: {
    danceability: number;
    energy: number;
    valence: number;
    tempo: number;
    acousticness: number;
    instrumentalness: number;
  };
  explanations: string[];
}

export interface RecommendResponse {
  user_idx: number;
  user_id: string;
  long_term_gate: number;
  short_term_gate: number;
  discovery_ratio_actual: number;
  feedback_summary: FeedbackSummary;
  recommendations: RecommendedTrack[];
}

export interface PlaylistSummary {
  pid: number;
  name: string;
  num_tracks: number;
  unique_artists: number;
  artist_concentration: number;
  sample_genres: string[];
  sample_artists: string[];
}

export interface ShuffleMetrics {
  intra_list_diversity: number;
  local_window_diversity: number;
  min_artist_gap: number;
  mean_artist_gap: number;
  artist_clump_rate: number;
  genre_switch_rate: number;
  acoustic_smoothness: number;
  mean_energy_jump: number;
  mean_bpm_jump: number;
  session_ndcg: number;
  discovery_pacing_std: number;
}

export interface ShuffledTrack {
  position: number;
  item_idx: number;
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  genre: string;
  duration_ms: number;
  is_discovery: boolean;
  audio: {
    danceability: number;
    energy: number;
    valence: number;
    tempo: number;
  };
  step_diagnostics: {
    step: number;
    relevance?: number;
    flow?: number;
    diversity?: number;
    penalty?: number;
    reason: string;
  };
}

export interface ShuffleRunResult {
  mode: string;
  metrics: ShuffleMetrics;
  tracks: ShuffledTrack[];
}

export interface ShuffleCompareResponse {
  source_name: string;
  pool_size: number;
  seed: number;
  adaptive: ShuffleRunResult;
  fisher_yates: ShuffleRunResult;
  artist_spaced: ShuffleRunResult;
}

export interface BenchmarkReport {
  device: string;
  dataset_stats: {
    num_tracks: number;
    num_artists: number;
    num_genres: number;
    num_users: number;
    num_interactions: number;
    num_curated_playlists: number;
    action_distribution: Record<string, number>;
  };
  recommendation_benchmarks: Record<string, Record<string, number>>;
  shuffle_benchmarks: Record<string, Record<string, number>>;
  feedback_adaptation_benchmarks: Record<string, Record<string, number>>;
  total_runtime_seconds: number;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  getUsers: () => apiFetch<{ users: UserPersona[] }>("/users?limit=24"),
  getProfile: (userIdx: number) => apiFetch<UserProfile>(`/users/${userIdx}/profile`),
  recommend: (body: {
    user_idx: number;
    top_k?: number;
    discovery_level: number;
    diversity_level: number;
    novelty_boost: number;
    short_term_weight: number | null;
  }) =>
    apiFetch<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  getPlaylists: () => apiFetch<{ playlists: PlaylistSummary[] }>("/playlists?limit=30"),
  compareShuffle: (body: {
    user_idx: number;
    playlist_pid: number | null;
    pool_size?: number;
    diversity_weight: number;
    flow_weight: number;
    relevance_weight: number;
    artist_cooldown: number;
    genre_cooldown: number;
    entropy_temperature: number;
    discovery_target: number;
    seed?: number | null;
  }) =>
    apiFetch<ShuffleCompareResponse>("/shuffle/compare", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  sendFeedback: (body: {
    user_idx: number;
    item_idx: number;
    action: string;
    played_ratio?: number;
  }) =>
    apiFetch<{ status: string; feedback_summary: FeedbackSummary }>("/feedback", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  resetFeedback: (userIdx: number) =>
    apiFetch<{ status: string }>("/feedback/reset", {
      method: "POST",
      body: JSON.stringify({ user_idx: userIdx }),
    }),
  searchCatalog: (q: string) =>
    apiFetch<{
      results: {
        item_idx: number;
        spotify_track_id: string;
        track_name: string;
        artist_name: string;
        album_name: string;
        genre: string;
        popularity: number;
        duration_ms: number;
      }[];
    }>(`/catalog/search?q=${encodeURIComponent(q)}&limit=12`),
  getBenchmarks: () => apiFetch<BenchmarkReport>("/benchmarks"),
};
