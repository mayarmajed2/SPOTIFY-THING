import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { handleServiceWorkerMessage } from "../background/service-worker";
import { App } from "../popup/App";
import { CompanionApiClient } from "../services/api";
import { SpotifyProvider } from "../services/spotify-api";
import {
  deriveCodeChallengeS256,
  generateCodeVerifier,
  isTokenExpired,
  launchSpotifyPkceAuth,
  refreshSpotifyAccessToken,
  saveTokenBundle,
} from "../services/spotify-auth";

const EXTENSION_ROOT = resolve(__dirname, "../..");

describe("1. Manifest V3 & Security Validation", () => {
  const manifest = JSON.parse(
    readFileSync(resolve(EXTENSION_ROOT, "manifest.json"), "utf-8"),
  ) as Record<string, unknown>;

  it("declares valid Manifest V3 with minimal permissions and zero content scripts", () => {
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.action).toBeDefined();
    expect((manifest.background as Record<string, string>).service_worker).toBe("service-worker.js");
    expect((manifest.background as Record<string, string>).type).toBe("module");

    const perms = manifest.permissions as string[];
    expect(perms).toEqual(["storage", "identity", "sidePanel"]);
    expect(perms).not.toContain("tabs");
    expect(perms).not.toContain("scripting");
    expect(perms).not.toContain("webRequest");
    expect(manifest.content_scripts).toBeUndefined();

    const hostPerms = manifest.host_permissions as string[];
    expect(hostPerms).not.toContain("<all_urls>");
    expect(hostPerms).not.toContain("*://*.spotify.com/*");
  });

  it("enforces strict Content Security Policy with no remote scripts and ships required icons", () => {
    const csp = (manifest.content_security_policy as Record<string, string>).extension_pages;
    expect(csp).toContain("script-src 'self'");
    expect(csp).toContain("object-src 'self'");
    expect(csp).not.toContain("unsafe-eval");
    expect(csp).not.toContain("http:");

    for (const size of [16, 32, 48, 128]) {
      expect(existsSync(resolve(EXTENSION_ROOT, `public/icons/icon-${size}.png`))).toBe(true);
    }
  });
});

describe("2. Spotify OAuth 2.0 PKCE & Token Expiration Handling", () => {
  it("generates RFC 7636 S256 code_verifier and URL-safe base64 code_challenge", async () => {
    const verifier = generateCodeVerifier(64);
    expect(verifier).toHaveLength(64);
    // Official RFC 7636 Appendix B test vector check:
    const rfcVerifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const rfcChallenge = await deriveCodeChallengeS256(rfcVerifier);
    expect(rfcChallenge).toBe("E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    expect(rfcChallenge).not.toContain("=");
    expect(rfcChallenge).not.toContain("+");
    expect(rfcChallenge).not.toContain("/");
  });

  it("detects token expiration with safety buffer and refreshes without client_secret", async () => {
    expect(
      isTokenExpired({
        access_token: "tok",
        expires_at: Date.now() + 10_000, // within 60s buffer -> expired
        scope: "user-read-private",
        token_type: "Bearer",
      }),
    ).toBe(true);

    const fetchMock = vi.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        access_token: "fresh_access_tok",
        refresh_token: "fresh_refresh_tok",
        expires_in: 3600,
        scope: "user-read-private playlist-read-private",
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const refreshed = await refreshSpotifyAccessToken("demo_client_id", "old_refresh_tok");
    expect(refreshed.access_token).toBe("fresh_access_tok");
    const callBody = String(fetchMock.mock.calls[0][1]?.body || "");
    expect(callBody).toContain("grant_type=refresh_token");
    expect(callBody).toContain("client_id=demo_client_id");
    expect(callBody).not.toContain("client_secret");
    vi.unstubAllGlobals();
  });

  it("rejects OAuth callback when CSRF state parameter mismatches", async () => {
    vi.stubGlobal("chrome", {
      identity: {
        getRedirectURL: () => "https://abcdefghijklmnopabcdefghijklmnop.chromiumapp.org/spotify-callback",
        launchWebAuthFlow: vi
          .fn()
          .mockResolvedValue(
            "https://abcdefghijklmnopabcdefghijklmnop.chromiumapp.org/spotify-callback?code=abc&state=tampered_state",
          ),
      },
    });

    await expect(
      launchSpotifyPkceAuth({ clientId: "client_123", scopes: ["user-read-private"] }),
    ).rejects.toThrow(/state mismatch/i);
    vi.unstubAllGlobals();
  });
});

describe("3. Capability-Aware SpotifyProvider & Permission Fallbacks", () => {
  it("handles 403 Development Mode restriction gracefully and never claims native shuffle override", async () => {
    await saveTokenBundle({
      access_token: "valid_tok",
      expires_at: Date.now() + 3_600_000,
      scope: "user-read-private playlist-read-private",
      token_type: "Bearer",
    });

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 403,
        json: async () => ({ error: { status: 403, message: "User not registered in Dev Dashboard" } }),
      }),
    );

    const provider = new SpotifyProvider();
    const userRes = await provider.getCurrentUser();
    expect(userRes.user).toBeNull();
    expect(userRes.warning).toMatch(/403 Forbidden/i);

    const caps = await provider.getAvailableCapabilities();
    expect(caps.can_read_profile).toBe(false);
    expect(caps.direct_queue_reorder_supported).toBe(false);
    expect(caps.native_shuffle_override_supported).toBe(false);
    vi.unstubAllGlobals();
  });

  it("filters playlists strictly to owned or collaborative items per 2026 Spotify Web API rules", async () => {
    await saveTokenBundle({
      access_token: "valid_tok",
      expires_at: Date.now() + 3_600_000,
      scope: "user-read-private playlist-read-private",
      token_type: "Bearer",
    });

    const fetchMock = vi.fn().mockImplementation(async (url: string) => {
      if (url.endsWith("/me")) {
        return { ok: true, status: 200, json: async () => ({ id: "owner_me", display_name: "Me" }) };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          items: [
            { id: "pl_owned", name: "My Mix", owner: { id: "owner_me" }, tracks: { total: 15 } },
            { id: "pl_collab", name: "Collab Mix", collaborative: true, owner: { id: "friend" }, tracks: { total: 10 } },
            { id: "pl_editorial", name: "Spotify Editorial", collaborative: false, owner: { id: "spotify" }, tracks: { total: 50 } },
          ],
        }),
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    const provider = new SpotifyProvider();
    const res = await provider.getAuthorizedPlaylistData();
    expect(res.playlists?.map((p) => p.id)).toEqual(["pl_owned", "pl_collab"]);
    vi.unstubAllGlobals();
  });
});

describe("4. Background Service Worker & Sanitized API Client", () => {
  it("handles health check and token validity messages in service worker", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok", engine_loaded: true }),
      }),
    );

    const healthMsg = await handleServiceWorkerMessage({ type: "PING_BACKEND_HEALTH" });
    expect(healthMsg.ok).toBe(true);

    const tokMsg = await handleServiceWorkerMessage({ type: "CHECK_SPOTIFY_TOKEN_VALIDITY" });
    expect(tokMsg.authenticated).toBe(false);
    vi.unstubAllGlobals();
  });

  it("redacts sensitive bearer or session tokens from API error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        json: async () => ({
          detail: "Rejected token st_ext_secretToken1234567890 and Bearer eyJhbGciOiJIUzI1NiJ9",
        }),
      }),
    );

    const client = new CompanionApiClient();
    await expect(client.getHealth()).rejects.toThrow(/\[REDACTED\]/);
    try {
      await client.getHealth();
    } catch (err) {
      const msg = (err as Error).message;
      expect(msg).not.toContain("st_ext_secretToken1234567890");
      expect(msg).not.toContain("eyJhbGciOiJIUzI1NiJ9");
    }
    vi.unstubAllGlobals();
  });
});

describe("5. Popup UI, Onboarding, Navigation, Discovery, Shuffle & Insights Journey", () => {
  it("completes onboarding and navigates across Discovery, Adaptive Shuffle, Insights, and Settings", async () => {
    let onboardingDone = false;

    const mockFetch = vi.fn().mockImplementation(async (input: string, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method || "GET").toUpperCase();

      if (url.endsWith("/api/v1/auth/session") && method === "POST") {
        return {
          ok: true,
          status: 201,
          json: async () => ({
            session_token: "st_ext_test_session_token_1234567890",
            user: {
              user_id: "usr_1",
              display_name: "Companion Listener",
              preferences: {
                onboarding_completed: onboardingDone,
                operating_mode: "independent",
                preferred_genres: ["indie"],
                preferred_artists: [],
                discovery_intensity: 0.4,
                diversity_level: 0.5,
                novelty_boost: 0.3,
                short_term_weight: null,
                shuffle_presets: {
                  artist_repetition: "balanced",
                  genre_diversity: "medium",
                  musical_coherence: "smooth",
                  discovery_mode: "balanced",
                  randomness: "balanced",
                  quota_urgency_weight: 0,
                },
                dismissed_item_indices: [],
              },
              spotify: {
                connected: false,
                mode: "independent",
                account_label: null,
                product_tier: null,
                granted_scopes: [],
                capabilities: {
                  can_read_profile: false,
                  can_read_playlists: false,
                  can_read_playback_state: false,
                  can_create_playlists: false,
                  direct_queue_reorder_supported: false,
                  native_shuffle_override_supported: false,
                },
                last_verified_at: null,
              },
            },
          }),
        };
      }

      if (url.endsWith("/api/v1/preferences") && method === "GET") {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            user_id: "usr_1",
            available_catalog_genres: ["indie", "electronic", "jazz", "rock"],
            preferences: {
              onboarding_completed: onboardingDone,
              operating_mode: "independent",
              preferred_genres: ["indie", "jazz"],
              preferred_artists: [],
              discovery_intensity: 0.45,
              diversity_level: 0.5,
              novelty_boost: 0.3,
              short_term_weight: null,
              shuffle_presets: {
                artist_repetition: "balanced",
                genre_diversity: "medium",
                musical_coherence: "smooth",
                discovery_mode: "balanced",
                randomness: "balanced",
                quota_urgency_weight: 0,
              },
              dismissed_item_indices: [],
            },
          }),
        };
      }

      if (url.endsWith("/api/v1/preferences") && method === "PUT") {
        onboardingDone = true;
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: "updated",
            display_name: "Music Explorer",
            preferences: {
              onboarding_completed: true,
              operating_mode: "independent",
              preferred_genres: ["indie", "electronic", "jazz"],
              preferred_artists: [],
              discovery_intensity: 0.45,
              diversity_level: 0.5,
              novelty_boost: 0.3,
              short_term_weight: null,
              shuffle_presets: {
                artist_repetition: "balanced",
                genre_diversity: "medium",
                musical_coherence: "smooth",
                discovery_mode: "balanced",
                randomness: "balanced",
                quota_urgency_weight: 0,
              },
              dismissed_item_indices: [],
            },
          }),
        };
      }

      if (url.endsWith("/api/v1/discovery")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            recommended_tracks: [
              {
                item_idx: 10,
                spotify_track_id: "sp_10",
                spotify_url: "https://open.spotify.com/track/sp_10",
                spotify_uri: "spotify:track:sp_10",
                track_name: "Midnight Resonance",
                artist_name: "Aurora Synth",
                album_name: "Northern Pulse",
                genre: "electronic",
                popularity: 62,
                duration_ms: 215000,
                is_discovery: true,
                novelty: 0.68,
                score: 0.89,
                short_term_affinity: 0.72,
                long_term_affinity: 0.65,
                audio: { danceability: 0.7, energy: 0.65, valence: 0.5, tempo: 118 },
                explanations: ["Hidden-gem discovery outside mainstream", "Matches your immediate session mood"],
              },
            ],
            new_discoveries: [],
            recommended_artists: [],
            related_genres: [{ genre_idx: 1, genre: "electronic", affinity_score: 0.8, track_count: 200, is_exploratory: false }],
            familiarity_vs_discovery: {
              target_discovery_ratio: 0.45,
              actual_discovery_ratio: 1.0,
              familiar_count: 0,
              discovery_count: 1,
              long_term_gate: 0.52,
              short_term_gate: 0.48,
            },
            feedback_summary: {
              num_feedback_events: 1,
              liked_count: 1,
              disliked_count: 0,
              skipped_count: 0,
              residual_magnitude: 0.12,
              fatigued_artists: 0,
              fatigued_genres: 0,
            },
          }),
        };
      }

      if (url.endsWith("/api/v1/feedback")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            status: "recorded",
            event: {
              item_idx: 10,
              track_name: "Midnight Resonance",
              artist_name: "Aurora Synth",
              genre: "electronic",
              action: "like",
              reward: 1.0,
            },
            feedback_summary: {
              num_feedback_events: 1,
              liked_count: 1,
              disliked_count: 0,
              skipped_count: 0,
              residual_magnitude: 0.18,
              fatigued_artists: 0,
              fatigued_genres: 0,
            },
          }),
        };
      }

      if (url.includes("/api/v1/discovery/catalog-playlists")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            playlists: [
              {
                pid: 101,
                name: "Late Night Indie & Synth",
                num_tracks: 24,
                unique_artists: 16,
                artist_concentration: 0.33,
                sample_genres: ["indie", "electronic"],
                sample_artists: ["Aurora Synth"],
              },
            ],
          }),
        };
      }

      if (url.endsWith("/api/v1/shuffle")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            source_name: "Late Night Indie & Synth",
            source_type: "catalog_playlist",
            pool_size: 2,
            seed: 42,
            applied_config: {
              presets: {
                artist_repetition: "balanced",
                genre_diversity: "medium",
                musical_coherence: "smooth",
                discovery_mode: "balanced",
                randomness: "balanced",
                quota_urgency_weight: 0,
              },
              diversity_weight: 0.65,
              flow_weight: 0.85,
              artist_cooldown: 4,
              genre_cooldown: 2,
              entropy_temperature: 0.18,
              discovery_target: 0.35,
              quota_urgency_weight: 0,
            },
            feature_coverage: {
              catalog_audio_matched: 2,
              fallback_tracks: 0,
              audio_coverage_ratio: 1,
              note: "All tracks matched independent multimodal audio & embedding features.",
            },
            original_tracks: [
              {
                position: 1,
                original_position: 1,
                item_idx: 10,
                spotify_track_id: "sp_10",
                spotify_url: "https://open.spotify.com/track/sp_10",
                spotify_uri: "spotify:track:sp_10",
                track_name: "Midnight Resonance",
                artist_name: "Aurora Synth",
                album_name: "Northern Pulse",
                genre: "electronic",
                duration_ms: 215000,
                is_discovery: true,
                feature_source: "independent_catalog",
                audio: { danceability: 0.7, energy: 0.65, valence: 0.5, tempo: 118 },
                transition: null,
              },
            ],
            generated_tracks: [
              {
                position: 1,
                original_position: 1,
                item_idx: 10,
                spotify_track_id: "sp_10",
                spotify_url: "https://open.spotify.com/track/sp_10",
                spotify_uri: "spotify:track:sp_10",
                track_name: "Midnight Resonance",
                artist_name: "Aurora Synth",
                album_name: "Northern Pulse",
                genre: "electronic",
                duration_ms: 215000,
                is_discovery: true,
                feature_source: "independent_catalog",
                audio: { danceability: 0.7, energy: 0.65, valence: 0.5, tempo: 118 },
                transition: { step: 0, reason: "Strong session opener aligned with taste" },
              },
            ],
            metrics: {
              intra_list_diversity: 0.68,
              local_window_diversity: 0.7,
              min_artist_gap: 5,
              mean_artist_gap: 6.2,
              artist_clump_rate: 0.04,
              genre_switch_rate: 0.82,
              acoustic_smoothness: 0.76,
              mean_energy_jump: 0.12,
              mean_bpm_jump: 14.2,
              discovery_pacing_std: 0.15,
            },
            baselines: {
              original_order: {
                intra_list_diversity: 0.68,
                local_window_diversity: 0.6,
                min_artist_gap: 1,
                mean_artist_gap: 3.1,
                artist_clump_rate: 0.18,
                genre_switch_rate: 0.6,
                acoustic_smoothness: 0.68,
                mean_energy_jump: 0.2,
                mean_bpm_jump: 22.4,
                discovery_pacing_std: 0.25,
              },
            },
            distributions: {
              artists: [{ artist: "Aurora Synth", count: 1, share: 0.5 }],
              genres: [{ genre: "electronic", count: 1, share: 0.5 }],
              discovery: { familiar_count: 1, discovery_count: 1, discovery_ratio: 0.5 },
              repetition: {
                unique_artists: 2,
                repeated_artists_count: 0,
                original_clump_rate: 0.18,
                generated_clump_rate: 0.04,
                min_artist_gap: 5,
                mean_artist_gap: 6.2,
              },
            },
            handoff: {
              direct_spotify_shuffle_override: false,
              limitation_notice: "SPOTIFY-THING generates an independent sequence.",
              spotify_uris: ["spotify:track:sp_10"],
              spotify_urls: ["https://open.spotify.com/track/sp_10"],
              m3u: "#EXTM3U",
              csv: "position,track_name",
            },
          }),
        };
      }

      if (url.endsWith("/api/v1/insights")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            observed: {
              provenance: "observed",
              description: "Directly recorded feedback interactions.",
              total_feedback_actions: 1,
              positive_actions: 1,
              negative_actions: 0,
              liked_tracks_count: 1,
              dismissed_or_disliked_count: 0,
              explicit_preferred_genres: ["indie", "jazz"],
              explicit_preferred_artists: [],
              observed_top_genres: [{ genre: "electronic", positive_signals: 1 }],
              observed_top_artists: [{ artist: "Aurora Synth", positive_signals: 1 }],
            },
            inferred_preferences: {
              provenance: "inferred_preference",
              description: "LinUCB inferred weights.",
              bandit_residual_magnitude: 0.18,
              preference_drift_score: 0.09,
              long_term_horizon_weight: 0.55,
              short_term_session_weight: 0.45,
              configured_discovery_intensity: 0.45,
              boosted_genres: [],
              fatigued_genres: [],
              boosted_artists: [],
              fatigued_artists: [],
            },
            model_predictions: {
              provenance: "model_prediction",
              description: "NeuMF + Transformer predictions.",
              predicted_favorite_genres: [{ genre_idx: 1, genre: "electronic", affinity_score: 0.85, track_count: 100, is_exploratory: false }],
              predicted_favorite_artists: [{ artist: "Aurora Synth", genre: "electronic", affinity_score: 0.88, is_discovery: true }],
              recommendation_diversity_ild: 0.67,
              unique_artists_in_top20: 19,
              repeated_artist_exposure_rate: 0.05,
              familiarity_vs_discovery: {
                target_discovery_ratio: 0.45,
                actual_discovery_ratio: 0.45,
                familiar_count: 11,
                discovery_count: 9,
                long_term_gate: 0.55,
                short_term_gate: 0.45,
              },
              predicted_sonic_profile: { danceability: 0.68, energy: 0.64, valence: 0.52 },
            },
          }),
        };
      }

      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "ok",
          api_version: "1.0.0",
          engine_loaded: true,
          independent_mode_available: true,
          device: "cuda",
          models: {
            multimodal_encoder: true,
            neumf: true,
            dual_horizon_transformer: true,
            item_cf: true,
            adaptive_shuffle: true,
            linucb_bandit: true,
          },
        }),
      };
    });

    vi.stubGlobal("fetch", mockFetch);

    render(<App />);

    // Step 1 -> Step 2 -> Step 3 of Onboarding
    await screen.findByText(/Smarter Discovery & Adaptive Shuffle alongside Spotify/i);
    fireEvent.click(screen.getByRole("button", { name: /Continue/i }));
    await screen.findByText(/Choose Operating Mode & Permissions/i);
    fireEvent.click(screen.getByRole("button", { name: /Continue/i }));
    await screen.findByText(/Tune Your Initial Discovery Profile/i);
    fireEvent.click(screen.getByRole("button", { name: /Launch Discovery/i }));

    // Discovery Page rendered with track + ML explanation
    await screen.findByText("Midnight Resonance");
    expect(screen.getByText(/Hidden-gem discovery outside mainstream/i)).toBeInTheDocument();

    // Submit Like feedback
    fireEvent.click(screen.getByRole("button", { name: /^Like$/i }));
    await screen.findByText(/Liked "Midnight Resonance"/i);

    // Navigate to Adaptive Shuffle Studio
    fireEvent.click(screen.getByRole("button", { name: /Shuffle/i }));
    await screen.findByText(/Sequence Diagnostics: Late Night Indie & Synth/i);
    expect(screen.getByText(/4.0%/i)).toBeInTheDocument();

    // Navigate to Insights & verify Observed / Inferred / Model Prediction provenance badges
    fireEvent.click(screen.getByRole("button", { name: /Insights/i }));
    await screen.findByText(/Observed Companion Activity/i);
    expect(screen.getByText("Observed")).toBeInTheDocument();
    expect(screen.getByText("Inferred Preference")).toBeInTheDocument();
    expect(screen.getByText("Model Prediction")).toBeInTheDocument();

    // Navigate to Settings
    fireEvent.click(screen.getByRole("button", { name: /Settings/i }));
    await waitFor(() => {
      expect(screen.getByText(/Official Spotify OAuth 2.0 \(PKCE\)/i)).toBeInTheDocument();
      expect(screen.getByText(/Delete Stored Preferences & Application Data/i)).toBeInTheDocument();
    });

    vi.unstubAllGlobals();
  });
});
