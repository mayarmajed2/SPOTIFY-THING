import React, { useState } from "react";
import { Check, Compass, Lock, Music2, ShieldCheck, Sparkles, Unplug } from "lucide-react";

const CURATED_ONBOARDING_GENRES = [
  "indie",
  "alternative",
  "electronic",
  "hip-hop",
  "pop",
  "rock",
  "r-n-b",
  "jazz",
  "ambient",
  "house",
  "classical",
  "folk",
  "synth-pop",
  "soul",
  "metal",
  "acoustic",
];

interface OnboardingFlowProps {
  availableGenres: string[];
  onComplete: (payload: {
    displayName: string;
    preferredGenres: string[];
    discoveryIntensity: number;
    connectSpotifyRequested: boolean;
    spotifyClientId?: string;
  }) => Promise<void>;
}

export const OnboardingFlow: React.FC<OnboardingFlowProps> = ({
  availableGenres,
  onComplete,
}) => {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [displayName, setDisplayName] = useState("Music Explorer");
  const [modeChoice, setModeChoice] = useState<"independent" | "spotify">("independent");
  const [spotifyClientId, setSpotifyClientId] = useState("");
  const [selectedGenres, setSelectedGenres] = useState<string[]>(["indie", "electronic", "jazz"]);
  const [discoveryIntensity, setDiscoveryIntensity] = useState(0.45);
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const genrePool =
    availableGenres.length >= 10
      ? CURATED_ONBOARDING_GENRES.filter((g) => availableGenres.includes(g)).concat(
          availableGenres.filter((g) => !CURATED_ONBOARDING_GENRES.includes(g)).slice(0, 8),
        )
      : CURATED_ONBOARDING_GENRES;

  const toggleGenre = (genre: string) => {
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre].slice(0, 8),
    );
  };

  const handleFinish = async () => {
    setSubmitting(true);
    setErrorMsg(null);
    try {
      await onComplete({
        displayName: displayName.trim() || "Music Explorer",
        preferredGenres: selectedGenres,
        discoveryIntensity,
        connectSpotifyRequested: modeChoice === "spotify" && spotifyClientId.trim().length > 0,
        spotifyClientId: spotifyClientId.trim(),
      });
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Could not save onboarding setup.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-[600px] bg-[#070a10] text-slate-100 p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-sky-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
              <Sparkles className="w-4 h-4 text-slate-950" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">SPOTIFY-THING</h1>
              <p className="text-[11px] text-slate-400">Independent AI Music Companion</p>
            </div>
          </div>
          <span className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-slate-800/80 text-teal-300 border border-slate-700/70">
            Step {step} of 3
          </span>
        </div>

        {step === 1 && (
          <div className="space-y-4 animate-fadeIn">
            <div className="p-4 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
              <h2 className="text-base font-semibold text-white">
                Smarter Discovery & Adaptive Shuffle alongside Spotify
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Spotify remains your music streaming and playback platform.{" "}
                <strong className="text-teal-300">SPOTIFY-THING</strong> adds an independent neural
                intelligence layer right in Chrome:
              </p>
              <ul className="space-y-2 text-xs text-slate-300 pt-1">
                <li className="flex items-start gap-2">
                  <Compass className="w-4 h-4 text-teal-400 shrink-0 mt-0.5" />
                  <span>
                    <strong>Dual-Horizon Neural Discovery:</strong> Balances stable long-term taste
                    (NeuMF) with immediate session mood (Causal Transformer) and novelty control.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <Music2 className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
                  <span>
                    <strong>Adaptive Shuffle Studio:</strong> Eliminates artist clumping and jarring
                    BPM/mood jumps with transparent sequence comparison & playlist handoff.
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span>
                    <strong>Privacy-First Bandit Personalization:</strong> Adapts to your likes,
                    skips, and fatigue signals locally without scraping Spotify pages.
                  </span>
                </li>
              </ul>
            </div>

            <div className="space-y-1.5">
              <label htmlFor="onboard-name" className="block text-xs font-medium text-slate-300">
                Profile Label (stored in your local profile)
              </label>
              <input
                id="onboard-name"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                maxLength={60}
                className="w-full px-3 py-2 text-xs rounded-xl bg-[#0e1524] border border-slate-700/80 text-slate-100 focus:outline-none focus:border-teal-400"
                placeholder="Music Explorer"
              />
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-3.5">
            <h2 className="text-base font-semibold text-white">
              Choose Operating Mode & Permissions
            </h2>
            <p className="text-xs text-slate-400 leading-relaxed">
              SPOTIFY-THING works immediately in Independent Mode using its 40,000-track multimodal
              engine, or can optionally connect via official Spotify OAuth 2.0 PKCE.
            </p>

            <button
              type="button"
              onClick={() => setModeChoice("independent")}
              className={`w-full text-left p-3.5 rounded-xl border transition ${
                modeChoice === "independent"
                  ? "bg-teal-500/10 border-teal-400/70"
                  : "bg-[#0e1524] border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Unplug className="w-3.5 h-3.5 text-teal-400" />
                  Independent Companion Mode (Recommended Default)
                </span>
                {modeChoice === "independent" && <Check className="w-4 h-4 text-teal-400" />}
              </div>
              <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                Zero Spotify API credentials required. Full AI Discovery, Adaptive Shuffle Studio,
                Music Insights, M3U/URI export, and one-click Open-in-Spotify playback links.
              </p>
            </button>

            <button
              type="button"
              onClick={() => setModeChoice("spotify")}
              className={`w-full text-left p-3.5 rounded-xl border transition ${
                modeChoice === "spotify"
                  ? "bg-sky-500/10 border-sky-400/70"
                  : "bg-[#0e1524] border-slate-800 hover:border-slate-700"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Lock className="w-3.5 h-3.5 text-sky-400" />
                  Optional Official Spotify OAuth (PKCE)
                </span>
                {modeChoice === "spotify" && <Check className="w-4 h-4 text-sky-400" />}
              </div>
              <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                Uses Authorization Code + PKCE (no client secret). Requests minimal read scopes (
                <code>user-read-private</code>, <code>playlist-read-private</code>) to import owned
                playlists. Never trains ML models on Spotify Content.
              </p>
            </button>

            {modeChoice === "spotify" && (
              <div className="p-3 rounded-xl bg-[#0b111d] border border-slate-800 space-y-2">
                <label
                  htmlFor="onboard-client-id"
                  className="block text-[11px] font-medium text-slate-300"
                >
                  Spotify Developer Client ID (optional — can also be configured later in Settings)
                </label>
                <input
                  id="onboard-client-id"
                  type="text"
                  value={spotifyClientId}
                  onChange={(e) => setSpotifyClientId(e.target.value)}
                  placeholder="e.g. a1b2c3d4e5f6..."
                  className="w-full px-2.5 py-1.5 text-xs rounded-lg bg-[#131b2b] border border-slate-700 text-slate-100 focus:outline-none focus:border-sky-400"
                />
                <p className="text-[10px] text-slate-400">
                  Note: Spotify Development Mode requires an active Premium owner account and up to
                  5 allowlisted test users. If left blank, Independent Mode activates seamlessly.
                </p>
              </div>
            )}
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <div>
              <h2 className="text-base font-semibold text-white">
                Tune Your Initial Discovery Profile
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Pick a few starting genres and set your Familiarity vs. Discovery balance. You can
                adjust or delete these preferences anytime.
              </p>
            </div>

            <div className="flex flex-wrap gap-1.5 max-h-44 overflow-y-auto pr-1">
              {genrePool.slice(0, 20).map((genre) => {
                const active = selectedGenres.includes(genre);
                return (
                  <button
                    key={genre}
                    type="button"
                    onClick={() => toggleGenre(genre)}
                    className={`px-2.5 py-1 rounded-full text-xs font-medium capitalize transition border ${
                      active
                        ? "bg-teal-500/20 text-teal-200 border-teal-400/70"
                        : "bg-[#0e1524] text-slate-300 border-slate-800 hover:border-slate-700"
                    }`}
                  >
                    {genre}
                  </button>
                );
              })}
            </div>

            <div className="p-3.5 rounded-xl bg-[#0e1524] border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-300 font-medium">Discovery Intensity</span>
                <span className="text-teal-300 font-semibold">
                  {Math.round(discoveryIntensity * 100)}% New Music
                </span>
              </div>
              <input
                aria-label="Initial Discovery Intensity"
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={discoveryIntensity}
                onChange={(e) => setDiscoveryIntensity(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-[10px] text-slate-400">
                <span>Familiar Favorites</span>
                <span>Balanced Mix</span>
                <span>Deep Exploration</span>
              </div>
            </div>

            {errorMsg && (
              <div className="p-2.5 rounded-lg bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200">
                {errorMsg}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="pt-4 mt-4 border-t border-slate-800/80 flex items-center justify-between gap-2">
        {step > 1 ? (
          <button
            type="button"
            onClick={() => setStep((s) => (s === 3 ? 2 : 1))}
            className="px-3.5 py-2 rounded-xl text-xs font-medium text-slate-300 bg-slate-800/70 hover:bg-slate-800 transition"
          >
            Back
          </button>
        ) : (
          <span className="text-[11px] text-slate-500">Minimal Chrome permissions</span>
        )}

        {step < 3 ? (
          <button
            type="button"
            onClick={() => setStep((s) => (s === 1 ? 2 : 3))}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-teal-500 to-emerald-500 text-slate-950 hover:opacity-95 transition"
          >
            Continue
          </button>
        ) : (
          <button
            type="button"
            disabled={submitting}
            onClick={handleFinish}
            className="px-4 py-2 rounded-xl text-xs font-semibold bg-gradient-to-r from-teal-500 to-sky-500 text-slate-950 hover:opacity-95 disabled:opacity-50 transition"
          >
            {submitting ? "Initializing AI Profile..." : "Launch Discovery"}
          </button>
        )}
      </div>
    </div>
  );
};
