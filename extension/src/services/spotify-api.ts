/**
 * Capability-Aware Spotify Web API Provider Adapter (`SpotifyProvider`).
 *
 * Strictly enforces official Spotify Developer Policy & 2025/2026 Web API specifications:
 * - Uses `POST /v1/me/playlists` (not the removed `/v1/users/{id}/playlists` endpoint).
 * - Uses `POST /v1/playlists/{id}/items` (with `/tracks` fallback if needed) when authorized.
 * - Filters `GET /v1/me/playlists` to owned or collaborative playlists to respect `403` access rules.
 * - NEVER calls restricted `/v1/audio-features`, `/v1/audio-analysis`, `/v1/recommendations`, or `/related-artists`.
 * - NEVER scrapes Spotify DOM or claims native Spotify Shuffle / unauthorized queue override.
 * - Gracefully degrades to Independent Mode on missing credentials, missing scopes, `401`, `403` (Dev Mode allowlist), or `429`.
 */

import type { SpotifyCapabilities } from "../types";
import {
  MINIMAL_READ_SCOPES,
  OPTIONAL_PLAYLIST_WRITE_SCOPE,
  clearStoredTokens,
  getValidSpotifyAccessToken,
  launchSpotifyPkceAuth,
} from "./spotify-auth";

const SPOTIFY_API_BASE = "https://api.spotify.com/v1";

export interface SpotifyUserProfile {
  id: string;
  display_name: string;
  product?: string;
  country?: string;
}

export interface AuthorizedPlaylistSummary {
  id: string;
  name: string;
  trackTotal: number;
  ownerId: string;
  collaborative: boolean;
  externalUrl?: string;
}

export interface AuthorizedTrackItem {
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  album_name: string;
  duration_ms: number;
}

export interface PlaybackSnapshot {
  available: boolean;
  is_playing: boolean;
  device_name: string | null;
  track_name: string | null;
  artist_name: string | null;
  spotify_track_id: string | null;
  reason?: string;
}

export class SpotifyProvider {
  private async authorizedFetch(path: string, init: RequestInit = {}): Promise<Response> {
    const bundle = await getValidSpotifyAccessToken();
    if (!bundle) {
      throw new Error("NOT_AUTHENTICATED");
    }

    const resp = await fetch(`${SPOTIFY_API_BASE}${path}`, {
      ...init,
      headers: {
        Authorization: `Bearer ${bundle.access_token}`,
        "Content-Type": "application/json",
        ...(init.headers as Record<string, string> | undefined),
      },
    });

    if (resp.status === 401) {
      await clearStoredTokens();
      throw new Error("SPOTIFY_TOKEN_REVOKED_OR_EXPIRED");
    }
    return resp;
  }

  async authenticate(options: { clientId?: string; scopes?: string[] } = {}): Promise<{
    connected: boolean;
    user: SpotifyUserProfile | null;
    grantedScopes: string[];
    capabilities: SpotifyCapabilities;
    warning?: string;
  }> {
    const bundle = await launchSpotifyPkceAuth({
      clientId: options.clientId,
      scopes: options.scopes || [...MINIMAL_READ_SCOPES],
    });
    const grantedScopes = bundle.scope ? bundle.scope.split(/\s+/).filter(Boolean) : [];
    const userRes = await this.getCurrentUser();
    const capabilities = await this.getAvailableCapabilities();

    return {
      connected: Boolean(userRes.user),
      user: userRes.user,
      grantedScopes,
      capabilities,
      warning: userRes.warning,
    };
  }

  async disconnect(): Promise<void> {
    await clearStoredTokens();
  }

  async getCurrentUser(): Promise<{ user: SpotifyUserProfile | null; warning?: string }> {
    try {
      const resp = await this.authorizedFetch("/me");
      if (resp.status === 403) {
        return {
          user: null,
          warning:
            "Spotify returned 403 Forbidden. In Development Mode, the app owner must have Spotify Premium and your account must be on the 5-user allowlist.",
        };
      }
      if (!resp.ok) {
        return { user: null, warning: `Spotify profile unavailable (${resp.status}).` };
      }
      const raw = (await resp.json()) as {
        id: string;
        display_name?: string;
        product?: string;
        country?: string;
      };
      return {
        user: {
          id: raw.id,
          display_name: raw.display_name || "Spotify Listener",
          product: raw.product,
          country: raw.country,
        },
      };
    } catch {
      return { user: null };
    }
  }

  async getAvailableCapabilities(): Promise<SpotifyCapabilities> {
    const bundle = await getValidSpotifyAccessToken();
    if (!bundle) {
      return {
        can_read_profile: false,
        can_read_playlists: false,
        can_read_playback_state: false,
        can_create_playlists: false,
        direct_queue_reorder_supported: false,
        native_shuffle_override_supported: false,
      };
    }
    const scopes = new Set((bundle.scope || "").split(/\s+/).filter(Boolean));
    const userCheck = await this.getCurrentUser();
    const profileOk = Boolean(userCheck.user);

    return {
      can_read_profile: profileOk,
      can_read_playlists: profileOk && (scopes.has("playlist-read-private") || scopes.has("playlist-read-collaborative")),
      can_read_playback_state: profileOk && (scopes.has("user-read-playback-state") || scopes.has("user-read-currently-playing")),
      can_create_playlists: profileOk && (scopes.has("playlist-modify-private") || scopes.has("playlist-modify-public")),
      // Official Spotify Web API does not allow reordering a user's active playback queue or replacing native client Shuffle
      direct_queue_reorder_supported: false,
      native_shuffle_override_supported: false,
    };
  }

  async getAuthorizedPlaylistData(playlistId?: string): Promise<{
    playlists?: AuthorizedPlaylistSummary[];
    tracks?: AuthorizedTrackItem[];
    warning?: string;
  }> {
    const userRes = await this.getCurrentUser();
    if (!userRes.user) {
      return {
        playlists: [],
        tracks: [],
        warning: userRes.warning || "Connect Spotify with playlist-read-private scope to load owned playlists.",
      };
    }

    if (!playlistId) {
      const resp = await this.authorizedFetch("/me/playlists?limit=30");
      if (resp.status === 403) {
        return {
          playlists: [],
          warning: "Playlist read permission unavailable (403). Using Independent Catalog playlists.",
        };
      }
      if (!resp.ok) {
        return { playlists: [], warning: `Could not load Spotify playlists (${resp.status}).` };
      }
      const data = (await resp.json()) as {
        items?: Array<{
          id: string;
          name: string;
          collaborative?: boolean;
          owner?: { id?: string };
          tracks?: { total?: number };
          items?: { total?: number };
          external_urls?: { spotify?: string };
        }>;
      };
      // Enforce 2026 Spotify rule: only playlists owned or collaborated on by current user allow item reads
      const eligible = (data.items || [])
        .filter((pl) => pl.owner?.id === userRes.user?.id || Boolean(pl.collaborative))
        .map((pl) => ({
          id: pl.id,
          name: pl.name,
          trackTotal: pl.tracks?.total ?? pl.items?.total ?? 0,
          ownerId: pl.owner?.id || "",
          collaborative: Boolean(pl.collaborative),
          externalUrl: pl.external_urls?.spotify,
        }));
      return { playlists: eligible };
    }

    const resp = await this.authorizedFetch(`/playlists/${encodeURIComponent(playlistId)}/tracks?limit=50`);
    if (resp.status === 403) {
      return {
        tracks: [],
        warning:
          "Spotify restricts track listing to playlists you own or collaborate on (403 Forbidden).",
      };
    }
    if (!resp.ok) {
      return { tracks: [], warning: `Unable to fetch playlist items (${resp.status}).` };
    }

    const payload = (await resp.json()) as {
      items?: Array<{
        track?: {
          id?: string;
          name?: string;
          duration_ms?: number;
          album?: { name?: string };
          artists?: Array<{ name?: string }>;
        };
      }>;
    };

    const tracks: AuthorizedTrackItem[] = [];
    for (const entry of payload.items || []) {
      const t = entry.track;
      if (!t || !t.name) continue;
      tracks.push({
        spotify_track_id: t.id || "",
        track_name: t.name,
        artist_name: t.artists?.[0]?.name || "Unknown Artist",
        album_name: t.album?.name || "Unknown Album",
        duration_ms: t.duration_ms || 210_000,
      });
    }
    return { tracks };
  }

  async getAvailablePlaybackState(): Promise<PlaybackSnapshot> {
    const caps = await this.getAvailableCapabilities();
    if (!caps.can_read_playback_state) {
      return {
        available: false,
        is_playing: false,
        device_name: null,
        track_name: null,
        artist_name: null,
        spotify_track_id: null,
        reason: "Playback state scope not requested (read-only companion mode).",
      };
    }

    try {
      const resp = await this.authorizedFetch("/me/player");
      if (resp.status === 204) {
        return {
          available: true,
          is_playing: false,
          device_name: null,
          track_name: null,
          artist_name: null,
          spotify_track_id: null,
          reason: "No active Spotify playback session detected.",
        };
      }
      if (!resp.ok) {
        return {
          available: false,
          is_playing: false,
          device_name: null,
          track_name: null,
          artist_name: null,
          spotify_track_id: null,
          reason: `Playback state unavailable (${resp.status}).`,
        };
      }
      const data = (await resp.json()) as {
        is_playing?: boolean;
        device?: { name?: string };
        item?: { id?: string; name?: string; artists?: Array<{ name?: string }> };
      };
      return {
        available: true,
        is_playing: Boolean(data.is_playing),
        device_name: data.device?.name || null,
        track_name: data.item?.name || null,
        artist_name: data.item?.artists?.[0]?.name || null,
        spotify_track_id: data.item?.id || null,
      };
    } catch {
      return {
        available: false,
        is_playing: false,
        device_name: null,
        track_name: null,
        artist_name: null,
        spotify_track_id: null,
        reason: "Not connected to Spotify playback endpoint.",
      };
    }
  }

  async createPlaylistIfPermitted(params: {
    name: string;
    description: string;
    trackUris: string[];
  }): Promise<{
    created: boolean;
    playlistUrl?: string;
    playlistId?: string;
    message: string;
  }> {
    const validUris = params.trackUris.filter((u) => u.startsWith("spotify:track:")).slice(0, 100);
    if (validUris.length === 0) {
      return { created: false, message: "No valid Spotify track URIs to export." };
    }

    let caps = await this.getAvailableCapabilities();
    if (!caps.can_create_playlists) {
      // Request playlist-modify-private scope on demand only when user explicitly invokes playlist creation
      try {
        await this.authenticate({
          scopes: [...MINIMAL_READ_SCOPES, OPTIONAL_PLAYLIST_WRITE_SCOPE],
        });
        caps = await this.getAvailableCapabilities();
      } catch (err) {
        return {
          created: false,
          message:
            err instanceof Error
              ? err.message
              : "Playlist creation requires Spotify authorization (playlist-modify-private).",
        };
      }
    }

    if (!caps.can_create_playlists) {
      return {
        created: false,
        message: "Missing playlist-modify-private permission. Use M3U/CSV/URI copy export instead.",
      };
    }

    const createResp = await this.authorizedFetch("/me/playlists", {
      method: "POST",
      body: JSON.stringify({
        name: params.name,
        description: params.description,
        public: false,
      }),
    });

    if (createResp.status === 403 || !createResp.ok) {
      return {
        created: false,
        message: `Spotify denied playlist creation (${createResp.status}). Copy URIs or export M3U instead.`,
      };
    }

    const createdPl = (await createResp.json()) as {
      id: string;
      external_urls?: { spotify?: string };
    };

    // Use updated 2026 `/playlists/{id}/items` route with graceful fallback to `/tracks`
    let addResp = await this.authorizedFetch(`/playlists/${encodeURIComponent(createdPl.id)}/items`, {
      method: "POST",
      body: JSON.stringify({ uris: validUris }),
    });
    if (addResp.status === 404 || addResp.status === 405) {
      addResp = await this.authorizedFetch(`/playlists/${encodeURIComponent(createdPl.id)}/tracks`, {
        method: "POST",
        body: JSON.stringify({ uris: validUris }),
      });
    }

    if (!addResp.ok) {
      return {
        created: false,
        message: `Playlist shell created, but adding items returned ${addResp.status}.`,
      };
    }

    return {
      created: true,
      playlistId: createdPl.id,
      playlistUrl: createdPl.external_urls?.spotify || `https://open.spotify.com/playlist/${createdPl.id}`,
      message: `Saved ${validUris.length} tracks to private Spotify playlist "${params.name}".`,
    };
  }

  async executeSupportedActions(
    action:
      | { type: "open_track_in_spotify"; spotifyTrackId: string }
      | { type: "open_url_in_spotify"; url: string },
  ): Promise<{ executed: boolean; url: string }> {
    const targetUrl =
      action.type === "open_track_in_spotify"
        ? `https://open.spotify.com/track/${encodeURIComponent(action.spotifyTrackId)}`
        : action.url;

    if (typeof chrome !== "undefined" && chrome?.tabs?.create) {
      await chrome.tabs.create({ url: targetUrl });
      return { executed: true, url: targetUrl };
    }
    globalThis.open?.(targetUrl, "_blank", "noopener,noreferrer");
    return { executed: true, url: targetUrl };
  }
}

export const spotifyProvider = new SpotifyProvider();
