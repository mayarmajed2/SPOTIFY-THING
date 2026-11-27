import React, { useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  Compass,
  Maximize2,
  Settings as SettingsIcon,
  Shuffle,
  Sparkles,
} from "lucide-react";
import { OnboardingFlow } from "../components/OnboardingFlow";
import { AdaptiveShufflePage } from "../pages/AdaptiveShuffle";
import { DiscoveryPage } from "../pages/Discovery";
import { InsightsPage } from "../pages/Insights";
import { SettingsPage } from "../pages/Settings";
import { companionApi } from "../services/api";
import { spotifyProvider } from "../services/spotify-api";
import type { ExtensionUser, SpotifyConnectionState, UserPreferences } from "../types";

type ActiveTab = "discovery" | "shuffle" | "insights" | "settings";

export const App: React.FC = () => {
  const [user, setUser] = useState<ExtensionUser | null>(null);
  const [availableGenres, setAvailableGenres] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<ActiveTab>("discovery");
  const [booting, setBooting] = useState<boolean>(true);
  const [bootError, setBootError] = useState<string | null>(null);

  const isExpandedView =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("view") === "expanded";

  const initializeCompanion = useCallback(async () => {
    setBooting(true);
    setBootError(null);
    try {
      const sessionUser = await companionApi.ensureSession();
      const prefsPayload = await companionApi.getPreferences();
      setUser({
        ...sessionUser,
        preferences: prefsPayload.preferences,
      });
      setAvailableGenres(prefsPayload.available_catalog_genres || []);
    } catch (err) {
      setBootError(
        err instanceof Error
          ? err.message
          : "Unable to connect to SPOTIFY-THING backend API.",
      );
    } finally {
      setBooting(false);
    }
  }, []);

  useEffect(() => {
    void initializeCompanion();
  }, [initializeCompanion]);

  const handleOnboardingComplete = async (payload: {
    displayName: string;
    preferredGenres: string[];
    discoveryIntensity: number;
    connectSpotifyRequested: boolean;
    spotifyClientId?: string;
  }) => {
    const updated = await companionApi.updatePreferences({
      display_name: payload.displayName,
      preferred_genres: payload.preferredGenres,
      discovery_intensity: payload.discoveryIntensity,
      onboarding_completed: true,
    });

    let nextSpotifyState: SpotifyConnectionState | undefined;
    if (payload.connectSpotifyRequested && payload.spotifyClientId) {
      try {
        const authRes = await spotifyProvider.authenticate({
          clientId: payload.spotifyClientId,
        });
        const synced = await companionApi.syncSpotifyStatus({
          connected: authRes.connected,
          account_label: authRes.user?.display_name || null,
          product_tier: authRes.user?.product || null,
          granted_scopes: authRes.grantedScopes,
          capabilities: { ...authRes.capabilities },
        });
        nextSpotifyState = synced.spotify;
      } catch {
        // Gracefully proceed in Independent Mode if OAuth popup is cancelled or unavailable
      }
    }

    setUser((prev) =>
      prev
        ? {
            ...prev,
            display_name: updated.display_name,
            preferences: updated.preferences,
            spotify: nextSpotifyState || prev.spotify,
          }
        : null,
    );
    setActiveTab("discovery");
  };

  const handlePreferencesUpdated = (prefs: UserPreferences) => {
    setUser((prev) => (prev ? { ...prev, preferences: prefs } : null));
  };

  const handleSpotifyStateChanged = (spotify: SpotifyConnectionState) => {
    setUser((prev) => (prev ? { ...prev, spotify } : null));
  };

  const handleExpandView = () => {
    if (typeof chrome !== "undefined" && chrome?.runtime?.sendMessage) {
      chrome.runtime.sendMessage({ type: "OPEN_COMPANION_TAB" });
      return;
    }
    window.open(`${window.location.pathname}?view=expanded`, "_blank");
  };

  if (booting) {
    return (
      <div className="min-h-[600px] bg-[#070a10] text-slate-100 flex flex-col items-center justify-center p-6 text-center space-y-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-sky-500 flex items-center justify-center animate-pulse">
          <Sparkles className="w-5 h-5 text-slate-950" />
        </div>
        <p className="text-xs font-medium text-slate-300">
          Connecting to SPOTIFY-THING Neural Engine...
        </p>
      </div>
    );
  }

  if (bootError || !user) {
    return (
      <div className="min-h-[600px] bg-[#070a10] text-slate-100 flex flex-col items-center justify-center p-6 text-center space-y-4">
        <div className="p-4 rounded-2xl bg-[#0e1524] border border-rose-500/40 space-y-2 max-w-sm">
          <h1 className="text-sm font-bold text-white">Companion Backend Offline</h1>
          <p className="text-xs text-slate-300 leading-relaxed">{bootError}</p>
          <code className="block p-2 rounded-lg bg-[#070a10] text-[10px] text-teal-300">
            uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
          </code>
          <button
            type="button"
            onClick={() => void initializeCompanion()}
            className="w-full py-2 rounded-xl text-xs font-semibold bg-teal-500 text-slate-950 hover:opacity-95"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!user.preferences.onboarding_completed) {
    return (
      <div className={isExpandedView ? "max-w-2xl mx-auto py-6" : "w-[420px]"}>
        <OnboardingFlow
          availableGenres={availableGenres}
          onComplete={handleOnboardingComplete}
        />
      </div>
    );
  }

  return (
    <div
      className={`min-h-[600px] bg-[#070a10] text-slate-100 flex flex-col ${
        isExpandedView ? "max-w-4xl mx-auto border-x border-slate-800/70" : "w-[420px]"
      }`}
    >
      {/* Top Companion Header */}
      <header className="sticky top-0 z-20 px-4 py-3 bg-[#070a10]/95 backdrop-blur border-b border-slate-800/80 flex items-center justify-between">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-400 to-sky-500 flex items-center justify-center shrink-0">
            <Sparkles className="w-3.5 h-3.5 text-slate-950" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <h1 className="text-xs font-bold tracking-tight text-white">SPOTIFY-THING</h1>
              <span
                className={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                  user.spotify.connected
                    ? "bg-emerald-500/15 text-emerald-300"
                    : "bg-teal-500/15 text-teal-300"
                }`}
              >
                {user.spotify.connected ? "Spotify PKCE" : "Independent AI"}
              </span>
            </div>
            <p className="text-[10px] text-slate-400 truncate">{user.display_name}</p>
          </div>
        </div>

        {!isExpandedView && (
          <button
            type="button"
            onClick={handleExpandView}
            title="Open Expanded Companion Studio"
            className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300"
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
        )}
      </header>

      {/* Main Scrollable Content */}
      <main className="flex-1 p-4 pb-20 overflow-y-auto">
        {activeTab === "discovery" && (
          <DiscoveryPage
            preferences={user.preferences}
            onPreferencesUpdated={handlePreferencesUpdated}
          />
        )}
        {activeTab === "shuffle" && (
          <AdaptiveShufflePage
            preferences={user.preferences}
            spotifyState={user.spotify}
            onPreferencesUpdated={handlePreferencesUpdated}
          />
        )}
        {activeTab === "insights" && <InsightsPage />}
        {activeTab === "settings" && (
          <SettingsPage
            spotifyState={user.spotify}
            onSpotifyStateChanged={handleSpotifyStateChanged}
            onDataDeleted={initializeCompanion}
          />
        )}
      </main>

      {/* Bottom Companion Navigation */}
      <nav
        aria-label="Companion Navigation"
        className="fixed bottom-0 left-0 right-0 z-20 bg-[#0b101b]/95 backdrop-blur border-t border-slate-800/90 px-2 py-1.5"
      >
        <div
          className={`grid grid-cols-4 gap-1 ${
            isExpandedView ? "max-w-4xl mx-auto" : "w-full"
          }`}
        >
          <button
            type="button"
            onClick={() => setActiveTab("discovery")}
            className={`py-1.5 rounded-xl flex flex-col items-center gap-0.5 text-[10px] font-medium transition ${
              activeTab === "discovery"
                ? "bg-teal-500/15 text-teal-300"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Compass className="w-4 h-4" />
            <span>Discovery</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("shuffle")}
            className={`py-1.5 rounded-xl flex flex-col items-center gap-0.5 text-[10px] font-medium transition ${
              activeTab === "shuffle"
                ? "bg-teal-500/15 text-teal-300"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Shuffle className="w-4 h-4" />
            <span>Shuffle</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("insights")}
            className={`py-1.5 rounded-xl flex flex-col items-center gap-0.5 text-[10px] font-medium transition ${
              activeTab === "insights"
                ? "bg-teal-500/15 text-teal-300"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            <span>Insights</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab("settings")}
            className={`py-1.5 rounded-xl flex flex-col items-center gap-0.5 text-[10px] font-medium transition ${
              activeTab === "settings"
                ? "bg-teal-500/15 text-teal-300"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <SettingsIcon className="w-4 h-4" />
            <span>Settings</span>
          </button>
        </div>
      </nav>
    </div>
  );
};
