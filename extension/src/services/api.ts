/**
 * Typed FastAPI `/api/v1/*` Companion Client.
 * - Manages isolated extension user token (`X-Extension-Token`) automatically.
 * - Sanitizes error messages so tokens are never leaked to UI or console.
 */

import type {
  AdaptiveShuffleResponse,
  BackendHealthResponse,
  CatalogPlaylistSummary,
  CompanionInsightsResponse,
  DiscoveryFeedResponse,
  ExtensionUser,
  FeedbackAction,
  ShufflePresets,
  SimilarArtistItem,
  SimilarTrackItem,
  SpotifyConnectionState,
  UserPreferences,
} from "../types";
import { extensionStorage } from "./storage";

const STORAGE_KEYS = {
  SESSION_TOKEN: "st_companion_session_token",
  API_BASE_URL: "st_companion_api_base_url",
} as const;

const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function redactTokens(text: string): string {
  return text.replace(/(Bearer\s+[A-Za-z0-9_\-.]+|st_ext_[A-Za-z0-9_-]+)/gi, "[REDACTED]");
}

export class CompanionApiClient {
  private cachedBaseUrl: string | null = null;
  private cachedToken: string | null = null;

  async getBaseUrl(): Promise<string> {
    if (this.cachedBaseUrl) return this.cachedBaseUrl;
    const saved = await extensionStorage.getLocal<string>(STORAGE_KEYS.API_BASE_URL);
    this.cachedBaseUrl = (saved || DEFAULT_API_BASE).replace(/\/+$/, "");
    return this.cachedBaseUrl;
  }

  async setBaseUrl(url: string): Promise<string> {
    const clean = url.trim().replace(/\/+$/, "") || DEFAULT_API_BASE;
    this.cachedBaseUrl = clean;
    await extensionStorage.setLocal(STORAGE_KEYS.API_BASE_URL, clean);
    return clean;
  }

  async getToken(): Promise<string | null> {
    if (this.cachedToken) return this.cachedToken;
    this.cachedToken = await extensionStorage.getLocal<string>(STORAGE_KEYS.SESSION_TOKEN);
    return this.cachedToken;
  }

  async setToken(token: string | null): Promise<void> {
    this.cachedToken = token;
    if (token) {
      await extensionStorage.setLocal(STORAGE_KEYS.SESSION_TOKEN, token);
    } else {
      await extensionStorage.removeLocal(STORAGE_KEYS.SESSION_TOKEN);
    }
  }

  private async request<T>(
    path: string,
    options: RequestInit = {},
    requireAuth = true,
  ): Promise<T> {
    const base = await this.getBaseUrl();
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string> | undefined),
    };

    if (requireAuth) {
      const token = await this.getToken();
      if (token) {
        headers["X-Extension-Token"] = token;
      }
    }

    let response: Response;
    try {
      response = await fetch(`${base}${path}`, {
        ...options,
        headers,
      });
    } catch {
      throw new Error(
        `Cannot reach SPOTIFY-THING backend at ${base}. Ensure the FastAPI server is running.`,
      );
    }

    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const body = (await response.json()) as { detail?: string };
        if (body?.detail) {
          detail = redactTokens(String(body.detail));
        }
      } catch {
        // Ignore non-JSON error payloads
      }
      if (response.status === 401 && requireAuth) {
        await this.setToken(null);
      }
      throw new Error(detail);
    }

    return (await response.json()) as T;
  }

  async getHealth(): Promise<BackendHealthResponse> {
    return this.request<BackendHealthResponse>("/api/v1/health", { method: "GET" }, false);
  }

  async ensureSession(initial?: {
    display_name?: string;
    preferred_genres?: string[];
    discovery_intensity?: number;
    onboarding_completed?: boolean;
  }): Promise<ExtensionUser> {
    const existing = await this.getToken();
    if (existing) {
      try {
        const validated = await this.request<{ authenticated: boolean; user: ExtensionUser }>(
          "/api/v1/auth/session",
          { method: "GET" },
          true,
        );
        return validated.user;
      } catch {
        // Token expired or backend DB reset -> create fresh isolated session
      }
    }

    const created = await this.request<{ session_token: string; user: ExtensionUser }>(
      "/api/v1/auth/session",
      {
        method: "POST",
        body: JSON.stringify({
          display_name: initial?.display_name || "Companion Listener",
          preferred_genres: initial?.preferred_genres || [],
          discovery_intensity: initial?.discovery_intensity ?? 0.4,
          onboarding_completed: initial?.onboarding_completed ?? false,
        }),
      },
      false,
    );
    await this.setToken(created.session_token);
    return created.user;
  }

  async getPreferences(): Promise<{
    user_id: string;
    preferences: UserPreferences;
    available_catalog_genres: string[];
  }> {
    return this.request("/api/v1/preferences", { method: "GET" });
  }

  async updatePreferences(
    updates: Partial<UserPreferences> & { display_name?: string },
  ): Promise<{ status: string; preferences: UserPreferences; display_name: string }> {
    return this.request("/api/v1/preferences", {
      method: "PUT",
      body: JSON.stringify(updates),
    });
  }

  async getDiscoveryFeed(params: {
    top_k?: number;
    discovery_intensity?: number;
    diversity_level?: number;
    novelty_boost?: number;
    exclude_item_indices?: number[];
    genre_filter?: string | null;
  } = {}): Promise<DiscoveryFeedResponse> {
    return this.request<DiscoveryFeedResponse>("/api/v1/discovery", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  async getSimilarTracks(
    itemIdx: number,
    topK = 8,
  ): Promise<{
    seed_track: { item_idx: number; track_name: string; artist_name: string; genre: string };
    similar_tracks: SimilarTrackItem[];
  }> {
    return this.request(`/api/v1/discovery/similar-tracks/${itemIdx}?top_k=${topK}`);
  }

  async getSimilarArtists(
    artistName: string,
    topK = 6,
  ): Promise<{
    seed_artist: { artist_idx: number; artist_name: string; genre: string };
    similar_artists: SimilarArtistItem[];
  }> {
    return this.request(
      `/api/v1/discovery/similar-artists?artist=${encodeURIComponent(artistName)}&top_k=${topK}`,
    );
  }

  async getCatalogPlaylists(limit = 25): Promise<{ playlists: CatalogPlaylistSummary[] }> {
    return this.request(`/api/v1/discovery/catalog-playlists?limit=${limit}`);
  }

  async generateShuffle(payload: {
    source_type: "catalog_playlist" | "discovery_pool" | "authorized_tracks";
    playlist_pid?: number | null;
    source_name?: string;
    pool_size?: number;
    tracks?: {
      spotify_track_id?: string;
      track_name: string;
      artist_name: string;
      album_name?: string;
      duration_ms?: number;
      genre?: string;
    }[];
    presets?: ShufflePresets;
    save_presets?: boolean;
    include_baselines?: boolean;
    seed?: number | null;
  }): Promise<AdaptiveShuffleResponse> {
    return this.request<AdaptiveShuffleResponse>("/api/v1/shuffle", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async saveShufflePresets(
    presets: ShufflePresets,
  ): Promise<{ status: string; shuffle_presets: ShufflePresets }> {
    return this.request("/api/v1/shuffle/preferences", {
      method: "PUT",
      body: JSON.stringify(presets),
    });
  }

  async submitFeedback(payload: {
    item_idx?: number | null;
    spotify_track_id?: string | null;
    artist_name?: string;
    genre?: string;
    action: FeedbackAction;
    played_ratio?: number;
  }): Promise<{
    status: string;
    event: {
      item_idx: number;
      track_name: string;
      artist_name: string;
      genre: string;
      action: string;
      reward: number;
    };
    feedback_summary: DiscoveryFeedResponse["feedback_summary"];
  }> {
    return this.request("/api/v1/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async resetFeedback(): Promise<{ status: string }> {
    return this.request("/api/v1/feedback/reset", { method: "POST" });
  }

  async getInsights(): Promise<CompanionInsightsResponse> {
    return this.request<CompanionInsightsResponse>("/api/v1/insights", { method: "GET" });
  }

  async syncSpotifyStatus(payload: {
    connected: boolean;
    account_label?: string | null;
    product_tier?: string | null;
    granted_scopes: string[];
    capabilities: Record<string, boolean>;
  }): Promise<{ status: string; spotify: SpotifyConnectionState }> {
    return this.request("/api/v1/spotify/status", {
      method: "PUT",
      body: JSON.stringify(payload),
    });
  }

  async disconnectSpotifyBackend(): Promise<{ status: string; spotify: SpotifyConnectionState }> {
    return this.request("/api/v1/spotify/disconnect", { method: "POST" });
  }

  async deleteAllMyData(): Promise<{ status: string }> {
    const res = await this.request<{ status: string }>("/api/v1/users/me", { method: "DELETE" });
    await this.setToken(null);
    return res;
  }
}

export const companionApi = new CompanionApiClient();
