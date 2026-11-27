import React, { useEffect, useState } from "react";
import {
  CheckCircle2,
  Copy,
  KeyRound,
  Link2,
  RotateCcw,
  Server,
  ShieldAlert,
  Trash2,
  Unlink,
} from "lucide-react";
import { companionApi } from "../services/api";
import { spotifyProvider } from "../services/spotify-api";
import {
  MINIMAL_READ_SCOPES,
  OPTIONAL_PLAYBACK_SCOPE,
  OPTIONAL_PLAYLIST_WRITE_SCOPE,
  getConfiguredSpotifyClientId,
  getExtensionRedirectUri,
  setConfiguredSpotifyClientId,
} from "../services/spotify-auth";
import { extensionStorage } from "../services/storage";
import type { BackendHealthResponse, SpotifyConnectionState } from "../types";

interface SettingsPageProps {
  spotifyState: SpotifyConnectionState;
  onSpotifyStateChanged: (next: SpotifyConnectionState) => void;
  onDataDeleted: () => Promise<void>;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  spotifyState,
  onSpotifyStateChanged,
  onDataDeleted,
}) => {
  const [apiUrl, setApiUrl] = useState("http://127.0.0.1:8000");
  const [health, setHealth] = useState<BackendHealthResponse | null>(null);

  const [clientId, setClientId] = useState("");
  const [optPlaybackScope, setOptPlaybackScope] = useState(false);
  const [optPlaylistWriteScope, setOptPlaylistWriteScope] = useState(false);

  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const redirectUri = getExtensionRedirectUri();

  useEffect(() => {
    void (async () => {
      const savedUrl = await companionApi.getBaseUrl();
      setApiUrl(savedUrl);
      const savedClient = await getConfiguredSpotifyClientId();
      setClientId(savedClient);
      try {
        const h = await companionApi.getHealth();
        setHealth(h);
      } catch {
        setHealth(null);
      }
    })();
  }, []);

  const handleSaveApiUrl = async () => {
    setError(null);
    const cleaned = await companionApi.setBaseUrl(apiUrl);
    setApiUrl(cleaned);
    try {
      const h = await companionApi.getHealth();
      setHealth(h);
      setMessage(`Connected to SPOTIFY-THING backend (${cleaned})`);
    } catch (err) {
      setHealth(null);
      setError(err instanceof Error ? err.message : "Backend unreachable.");
    }
  };

  const handleConnectSpotify = async () => {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      await setConfiguredSpotifyClientId(clientId);
      const scopes: string[] = [...MINIMAL_READ_SCOPES];
      if (optPlaybackScope) scopes.push(OPTIONAL_PLAYBACK_SCOPE);
      if (optPlaylistWriteScope) scopes.push(OPTIONAL_PLAYLIST_WRITE_SCOPE);

      const authRes = await spotifyProvider.authenticate({
        clientId,
        scopes,
      });

      const synced = await companionApi.syncSpotifyStatus({
        connected: authRes.connected,
        account_label: authRes.user?.display_name || null,
        product_tier: authRes.user?.product || null,
        granted_scopes: authRes.grantedScopes,
        capabilities: { ...authRes.capabilities },
      });
      onSpotifyStateChanged(synced.spotify);
      setMessage(
        authRes.warning ||
          `Connected Spotify account (${authRes.user?.display_name || "Authorized"}).`,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Spotify PKCE authentication failed.");
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnectSpotify = async () => {
    setBusy(true);
    setError(null);
    try {
      await spotifyProvider.disconnect();
      const res = await companionApi.disconnectSpotifyBackend();
      onSpotifyStateChanged(res.spotify);
      setMessage("Disconnected Spotify and cleared local OAuth tokens. Independent Mode active.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not disconnect Spotify.");
    } finally {
      setBusy(false);
    }
  };

  const handleResetFeedback = async () => {
    setBusy(true);
    try {
      await companionApi.resetFeedback();
      setMessage("Reset LinUCB contextual bandit weights and cleared temporary artist/genre fatigue.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to reset feedback memory.");
    } finally {
      setBusy(false);
    }
  };

  const handleDeleteAllData = async () => {
    setBusy(true);
    try {
      await spotifyProvider.disconnect();
      await companionApi.deleteAllMyData();
      await extensionStorage.clearAllAppStorage();
      await onDataDeleted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete stored user data.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      {message && (
        <div className="px-3 py-2 rounded-xl bg-teal-500/15 border border-teal-500/35 text-[11px] text-teal-200 flex items-center gap-1.5">
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
          <span>{message}</span>
        </div>
      )}

      {error && (
        <div className="px-3 py-2 rounded-xl bg-rose-500/15 border border-rose-500/35 text-[11px] text-rose-200">
          {error}
        </div>
      )}

      {/* Spotify OAuth PKCE Section */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-white flex items-center gap-1.5">
            <KeyRound className="w-3.5 h-3.5 text-sky-400" />
            Official Spotify OAuth 2.0 (PKCE)
          </h2>
          <span
            className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
              spotifyState.connected
                ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                : "bg-slate-800 text-slate-300"
            }`}
          >
            {spotifyState.connected
              ? `Connected (${spotifyState.account_label || "Authorized"})`
              : "Independent Mode"}
          </span>
        </div>

        <p className="text-[11px] text-slate-400 leading-relaxed">
          Optional connection using Authorization Code with PKCE (no client secret). Tokens remain
          exclusively in Chrome Extension storage and are never sent to the backend or used for ML
          training.
        </p>

        <div className="space-y-1.5">
          <label htmlFor="spotify-client-id" className="block text-[11px] font-medium text-slate-300">
            Spotify Developer Client ID
          </label>
          <input
            id="spotify-client-id"
            type="text"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            placeholder="Paste your Spotify App Client ID"
            className="w-full px-3 py-1.5 rounded-xl bg-[#090e18] border border-slate-700 text-xs text-slate-100 focus:outline-none focus:border-sky-400"
          />
        </div>

        <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-medium text-slate-400">
              Redirect URI (register in Spotify Developer Dashboard)
            </span>
            <button
              type="button"
              onClick={() => void navigator.clipboard?.writeText(redirectUri)}
              className="text-[10px] text-teal-300 hover:underline flex items-center gap-1"
            >
              <Copy className="w-2.5 h-2.5" /> Copy
            </button>
          </div>
          <code className="block text-[10px] text-slate-300 break-all">{redirectUri}</code>
        </div>

        <div className="space-y-1.5 text-[11px] text-slate-300">
          <p className="text-[10px] text-slate-400 font-medium">Requested Permissions:</p>
          <label className="flex items-center gap-2 text-slate-400">
            <input type="checkbox" checked disabled className="rounded" />
            <span>
              Base read (<code>user-read-private</code>, <code>playlist-read-private</code>)
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={optPlaybackScope}
              onChange={(e) => setOptPlaybackScope(e.target.checked)}
              className="rounded"
            />
            <span>
              Read active playback state (<code>user-read-playback-state</code>)
            </span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={optPlaylistWriteScope}
              onChange={(e) => setOptPlaylistWriteScope(e.target.checked)}
              className="rounded"
            />
            <span>
              Export sequences to new private playlists (<code>playlist-modify-private</code>)
            </span>
          </label>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void handleConnectSpotify()}
            className="flex-1 py-2 rounded-xl text-xs font-semibold bg-sky-500/20 hover:bg-sky-500/30 text-sky-200 border border-sky-400/40 transition flex items-center justify-center gap-1.5"
          >
            <Link2 className="w-3.5 h-3.5" />
            {spotifyState.connected ? "Re-Verify Spotify PKCE" : "Connect Spotify (PKCE)"}
          </button>
          {spotifyState.connected && (
            <button
              type="button"
              disabled={busy}
              onClick={() => void handleDisconnectSpotify()}
              className="px-3 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1"
            >
              <Unlink className="w-3.5 h-3.5" /> Disconnect
            </button>
          )}
        </div>
      </section>

      {/* Backend Connection & ML Engine Health */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold text-white flex items-center gap-1.5">
            <Server className="w-3.5 h-3.5 text-teal-400" />
            FastAPI & PyTorch Engine Connection
          </h3>
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full ${
              health?.engine_loaded
                ? "bg-emerald-500/15 text-emerald-300"
                : "bg-amber-500/15 text-amber-300"
            }`}
          >
            {health?.engine_loaded
              ? `Ready (${health.device?.toUpperCase() || "CPU"})`
              : "Offline / Degraded"}
          </span>
        </div>

        <div className="flex gap-2">
          <input
            aria-label="Backend API Base URL"
            type="url"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            className="flex-1 px-3 py-1.5 rounded-xl bg-[#090e18] border border-slate-700 text-xs text-slate-100"
          />
          <button
            type="button"
            onClick={() => void handleSaveApiUrl()}
            className="px-3 py-1.5 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200"
          >
            Verify
          </button>
        </div>

        {health?.catalog && (
          <p className="text-[10px] text-slate-400">
            Independent Catalog: {health.catalog.num_tracks.toLocaleString()} tracks ·{" "}
            {health.catalog.num_artists.toLocaleString()} artists · {health.catalog.num_genres}{" "}
            genres · NeuMF + Dual-Horizon Transformer + LinUCB active.
          </p>
        )}
      </section>

      {/* Personalization & Privacy Data Controls */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
        <h3 className="text-xs font-semibold text-white flex items-center gap-1.5">
          <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
          Personalization & Data Privacy Controls
        </h3>
        <p className="text-[11px] text-slate-400">
          Reset your online bandit adaptation or permanently erase your stored companion profile,
          preferences, and feedback history.
        </p>

        <div className="flex flex-col gap-2 pt-1">
          <button
            type="button"
            disabled={busy}
            onClick={() => void handleResetFeedback()}
            className="w-full py-2 px-3 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center justify-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Bandit Feedback & Fatigue Memory
          </button>

          <button
            type="button"
            disabled={busy}
            onClick={() => void handleDeleteAllData()}
            className="w-full py-2 px-3 rounded-xl text-xs font-medium bg-rose-500/15 hover:bg-rose-500/25 border border-rose-500/35 text-rose-200 flex items-center justify-center gap-1.5"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Delete Stored Preferences & Application Data
          </button>
        </div>
      </section>
    </div>
  );
};
