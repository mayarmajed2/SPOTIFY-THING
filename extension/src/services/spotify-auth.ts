/**
 * Official Spotify OAuth 2.0 Authorization Code with PKCE (S256) Module.
 *
 * Security Compliance:
 * - Zero embedded client secret.
 * - Cryptographically random RFC 7636 code_verifier + SHA-256 base64url code_challenge.
 * - CSRF `state` verification on callback.
 * - Tokens stored strictly in extension session/local storage; NEVER logged or sent to backend.
 */

import { extensionStorage } from "./storage";

export const SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize";
export const SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token";

const TOKEN_STORAGE_KEY = "st_spotify_pkce_token_bundle";
const CLIENT_ID_STORAGE_KEY = "st_spotify_client_id";

export interface SpotifyTokenBundle {
  access_token: string;
  refresh_token?: string;
  expires_at: number; // epoch ms
  scope: string;
  token_type: string;
}

export const MINIMAL_READ_SCOPES = ["user-read-private", "playlist-read-private"] as const;
export const OPTIONAL_PLAYBACK_SCOPE = "user-read-playback-state";
export const OPTIONAL_PLAYLIST_WRITE_SCOPE = "playlist-modify-private";

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.byteLength; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function generateCodeVerifier(length = 64): string {
  const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
  const randomValues = new Uint8Array(length);
  crypto.getRandomValues(randomValues);
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += charset[randomValues[i] % charset.length];
  }
  return out;
}

export async function deriveCodeChallengeS256(verifier: string): Promise<string> {
  const data = new TextEncoder().encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

export function getExtensionRedirectUri(): string {
  if (typeof chrome !== "undefined" && chrome?.identity?.getRedirectURL) {
    return chrome.identity.getRedirectURL("spotify-callback");
  }
  return `${globalThis.location?.origin || "http://localhost:5173"}/spotify-callback`;
}

export async function getConfiguredSpotifyClientId(): Promise<string> {
  return (await extensionStorage.getLocal<string>(CLIENT_ID_STORAGE_KEY)) || "";
}

export async function setConfiguredSpotifyClientId(clientId: string): Promise<void> {
  const trimmed = clientId.trim();
  if (!trimmed) {
    await extensionStorage.removeLocal(CLIENT_ID_STORAGE_KEY);
    return;
  }
  await extensionStorage.setLocal(CLIENT_ID_STORAGE_KEY, trimmed);
}

export async function getStoredTokenBundle(): Promise<SpotifyTokenBundle | null> {
  const sessionBundle = await extensionStorage.getSession<SpotifyTokenBundle>(TOKEN_STORAGE_KEY);
  if (sessionBundle) return sessionBundle;
  return extensionStorage.getLocal<SpotifyTokenBundle>(TOKEN_STORAGE_KEY);
}

export async function saveTokenBundle(bundle: SpotifyTokenBundle): Promise<void> {
  await extensionStorage.setSession(TOKEN_STORAGE_KEY, bundle);
  await extensionStorage.setLocal(TOKEN_STORAGE_KEY, bundle);
}

export async function clearStoredTokens(): Promise<void> {
  await extensionStorage.removeSession(TOKEN_STORAGE_KEY);
  await extensionStorage.removeLocal(TOKEN_STORAGE_KEY);
}

export function isTokenExpired(bundle: SpotifyTokenBundle, safetyBufferMs = 60_000): boolean {
  return Date.now() + safetyBufferMs >= bundle.expires_at;
}

export async function refreshSpotifyAccessToken(
  clientId: string,
  refreshToken: string,
): Promise<SpotifyTokenBundle> {
  const body = new URLSearchParams({
    client_id: clientId,
    grant_type: "refresh_token",
    refresh_token: refreshToken,
  });

  const resp = await fetch(SPOTIFY_TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  if (!resp.ok) {
    await clearStoredTokens();
    throw new Error(`Spotify token refresh failed (${resp.status}). Please reconnect Spotify.`);
  }

  const data = (await resp.json()) as {
    access_token: string;
    expires_in: number;
    refresh_token?: string;
    scope?: string;
    token_type?: string;
  };

  const nextBundle: SpotifyTokenBundle = {
    access_token: data.access_token,
    refresh_token: data.refresh_token || refreshToken,
    expires_at: Date.now() + Number(data.expires_in || 3600) * 1000,
    scope: data.scope || "",
    token_type: data.token_type || "Bearer",
  };
  await saveTokenBundle(nextBundle);
  return nextBundle;
}

export async function getValidSpotifyAccessToken(): Promise<SpotifyTokenBundle | null> {
  const bundle = await getStoredTokenBundle();
  if (!bundle) return null;
  if (!isTokenExpired(bundle)) return bundle;

  const clientId = await getConfiguredSpotifyClientId();
  if (!clientId || !bundle.refresh_token) {
    await clearStoredTokens();
    return null;
  }

  try {
    return await refreshSpotifyAccessToken(clientId, bundle.refresh_token);
  } catch {
    await clearStoredTokens();
    return null;
  }
}

export async function launchSpotifyPkceAuth(options: {
  clientId?: string;
  scopes?: string[];
} = {}): Promise<SpotifyTokenBundle> {
  const clientId = (options.clientId ?? (await getConfiguredSpotifyClientId())).trim();
  if (!clientId) {
    throw new Error(
      "Spotify Client ID is not configured. Enter your Spotify Developer Client ID in Settings, or continue in Independent Mode.",
    );
  }
  if (typeof chrome === "undefined" || !chrome?.identity?.launchWebAuthFlow) {
    throw new Error(
      "Chrome Identity OAuth flow requires loading SPOTIFY-THING as a Chrome Extension. Independent Mode remains active.",
    );
  }

  await setConfiguredSpotifyClientId(clientId);

  const scopes = Array.from(new Set(options.scopes || [...MINIMAL_READ_SCOPES]));
  const verifier = generateCodeVerifier(64);
  const challenge = await deriveCodeChallengeS256(verifier);
  const csrfState = generateCodeVerifier(24);
  const redirectUri = getExtensionRedirectUri();

  const authParams = new URLSearchParams({
    client_id: clientId,
    response_type: "code",
    redirect_uri: redirectUri,
    code_challenge_method: "S256",
    code_challenge: challenge,
    state: csrfState,
    scope: scopes.join(" "),
  });

  const callbackUrl = await chrome.identity.launchWebAuthFlow({
    url: `${SPOTIFY_AUTH_URL}?${authParams.toString()}`,
    interactive: true,
  });

  if (!callbackUrl) {
    throw new Error("Spotify authorization was cancelled.");
  }

  const parsed = new URL(callbackUrl);
  const errorParam = parsed.searchParams.get("error");
  if (errorParam) {
    throw new Error(`Spotify authorization declined: ${errorParam}`);
  }

  const returnedState = parsed.searchParams.get("state");
  const authCode = parsed.searchParams.get("code");
  if (!authCode || returnedState !== csrfState) {
    throw new Error("Spotify OAuth state mismatch or missing authorization code.");
  }

  const tokenParams = new URLSearchParams({
    client_id: clientId,
    grant_type: "authorization_code",
    code: authCode,
    redirect_uri: redirectUri,
    code_verifier: verifier,
  });

  const tokenResp = await fetch(SPOTIFY_TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: tokenParams.toString(),
  });

  if (!tokenResp.ok) {
    throw new Error(
      `Spotify token exchange failed (${tokenResp.status}). Verify Redirect URI in your Spotify Developer Dashboard.`,
    );
  }

  const tokenData = (await tokenResp.json()) as {
    access_token: string;
    refresh_token?: string;
    expires_in: number;
    scope?: string;
    token_type?: string;
  };

  const bundle: SpotifyTokenBundle = {
    access_token: tokenData.access_token,
    refresh_token: tokenData.refresh_token,
    expires_at: Date.now() + Number(tokenData.expires_in || 3600) * 1000,
    scope: tokenData.scope || scopes.join(" "),
    token_type: tokenData.token_type || "Bearer",
  };
  await saveTokenBundle(bundle);
  return bundle;
}
