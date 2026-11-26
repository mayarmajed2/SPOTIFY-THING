"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Compass,
  Disc3,
  Flame,
  Heart,
  Play,
  RefreshCw,
  Repeat,
  RotateCcw,
  Search,
  Shuffle,
  SkipForward,
  Sliders,
  Sparkles,
  ThumbsDown,
  CheckCircle2,
  User,
} from "lucide-react";
import {
  api,
  BenchmarkReport,
  PlaylistSummary,
  RecommendedTrack,
  RecommendResponse,
  ShuffleCompareResponse,
  ShuffledTrack,
  UserPersona,
  UserProfile,
} from "@/lib/api";
import { DualHorizonRadar, EnergyFlowSparkline } from "@/components/VisualCharts";

type ActiveTab = "discover" | "shuffle" | "benchmarks";

interface ActivePlayerTrack {
  item_idx: number;
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  genre: string;
}

export default function ResonanceStudioPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("discover");
  const [users, setUsers] = useState<UserPersona[]>([]);
  const [selectedUserIdx, setSelectedUserIdx] = useState<number>(0);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [recData, setRecData] = useState<RecommendResponse | null>(null);

  // Personalization Sliders
  const [discoveryLevel, setDiscoveryLevel] = useState<number>(0.42);
  const [diversityLevel, setDiversityLevel] = useState<number>(0.55);
  const [noveltyBoost, setNoveltyBoost] = useState<number>(0.35);
  const [autoHorizonGate, setAutoHorizonGate] = useState<boolean>(true);
  const [manualShortTerm, setManualShortTerm] = useState<number>(0.55);

  // Adaptive Shuffle State
  const [playlists, setPlaylists] = useState<PlaylistSummary[]>([]);
  const [selectedPlaylistPid, setSelectedPlaylistPid] = useState<number | null>(null);
  const [shuffleDiversity, setShuffleDiversity] = useState<number>(0.75);
  const [shuffleFlow, setShuffleFlow] = useState<number>(0.55);
  const [artistCooldown, setArtistCooldown] = useState<number>(4);
  const [entropyTemp, setEntropyTemp] = useState<number>(0.18);
  const [shuffleData, setShuffleData] = useState<ShuffleCompareResponse | null>(null);

  // Benchmarks & Search & Active Track
  const [benchmarks, setBenchmarks] = useState<BenchmarkReport | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [searchResults, setSearchResults] = useState<
    {
      item_idx: number;
      spotify_track_id: string;
      track_name: string;
      artist_name: string;
      genre: string;
      popularity: number;
    }[]
  >([]);
  const [activeTrack, setActiveTrack] = useState<ActivePlayerTrack | null>(null);
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3200);
  };

  // Initial bootstrap
  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [uRes, plRes, bmRes] = await Promise.all([
          api.getUsers(),
          api.getPlaylists(),
          api.getBenchmarks().catch(() => null),
        ]);
        setUsers(uRes.users);
        setPlaylists(plRes.playlists);
        if (plRes.playlists.length > 0) {
          setSelectedPlaylistPid(plRes.playlists[0].pid);
        }
        if (bmRes) setBenchmarks(bmRes);
        if (uRes.users.length > 0) {
          setSelectedUserIdx(uRes.users[0].user_idx);
        }
        setError(null);
      } catch (err: unknown) {
        setError(
          err instanceof Error
            ? err.message
            : "Could not connect to FastAPI backend on port 8000."
        );
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const refreshUserAndRecs = useCallback(async () => {
    try {
      const [prof, recs] = await Promise.all([
        api.getProfile(selectedUserIdx),
        api.recommend({
          user_idx: selectedUserIdx,
          top_k: 24,
          discovery_level: discoveryLevel,
          diversity_level: diversityLevel,
          novelty_boost: noveltyBoost,
          short_term_weight: autoHorizonGate ? null : manualShortTerm,
        }),
      ]);
      setProfile(prof);
      setRecData(recs);
      if (!activeTrack && recs.recommendations.length > 0) {
        const first = recs.recommendations[0];
        setActiveTrack({
          item_idx: first.item_idx,
          spotify_track_id: first.spotify_track_id,
          track_name: first.track_name,
          artist_name: first.artist_name,
          genre: first.genre,
        });
      }
    } catch (err: unknown) {
      if (err instanceof Error) setError(err.message);
    }
  }, [
    selectedUserIdx,
    discoveryLevel,
    diversityLevel,
    noveltyBoost,
    autoHorizonGate,
    manualShortTerm,
    activeTrack,
  ]);

  useEffect(() => {
    if (users.length > 0) {
      refreshUserAndRecs();
    }
  }, [users.length, refreshUserAndRecs]);

  const runShuffleComparison = useCallback(
    async (newRandomSeed = false) => {
      try {
        const res = await api.compareShuffle({
          user_idx: selectedUserIdx,
          playlist_pid: selectedPlaylistPid,
          pool_size: 32,
          diversity_weight: shuffleDiversity,
          flow_weight: shuffleFlow,
          relevance_weight: 0.55,
          artist_cooldown: artistCooldown,
          genre_cooldown: 2,
          entropy_temperature: entropyTemp,
          discovery_target: discoveryLevel,
          seed: newRandomSeed ? Math.floor(Math.random() * 90000) + 1 : undefined,
        });
        setShuffleData(res);
      } catch (err: unknown) {
        if (err instanceof Error) setError(err.message);
      }
    },
    [
      selectedUserIdx,
      selectedPlaylistPid,
      shuffleDiversity,
      shuffleFlow,
      artistCooldown,
      entropyTemp,
      discoveryLevel,
    ]
  );

  useEffect(() => {
    if (users.length > 0 && activeTab === "shuffle") {
      runShuffleComparison(false);
    }
  }, [users.length, activeTab, runShuffleComparison]);

  const handleFeedback = async (
    track: {
      item_idx: number;
      spotify_track_id: string;
      track_name: string;
      artist_name: string;
      genre: string;
    },
    action: "like" | "dislike" | "early_skip" | "complete_play" | "repeat"
  ) => {
    const ratio = action === "early_skip" ? 0.1 : 1.0;
    await api.sendFeedback({
      user_idx: selectedUserIdx,
      item_idx: track.item_idx,
      action,
      played_ratio: ratio,
    });
    const labelMap = {
      like: `Liked "${track.track_name}" — boosting ${track.genre} & similar vectors`,
      repeat: `Replayed "${track.track_name}" — reinforcing short-term session mood`,
      complete_play: `Completed "${track.track_name}" — positive implicit signal recorded`,
      early_skip: `Early-skipped "${track.track_name}" (<15%) — applying ${track.artist_name} cooldown`,
      dislike: `Disliked "${track.track_name}" — suppressing similar tracks immediately`,
    };
    showToast(labelMap[action]);
    await refreshUserAndRecs();
    if (activeTab === "shuffle") {
      await runShuffleComparison(false);
    }
  };

  const handleResetFeedback = async () => {
    await api.resetFeedback(selectedUserIdx);
    showToast("Reset live session feedback & bandit memory.");
    await refreshUserAndRecs();
  };

  const handleSearch = async (q: string) => {
    setSearchQuery(q);
    if (q.trim().length < 2) {
      setSearchResults([]);
      return;
    }
    const res = await api.searchCatalog(q);
    setSearchResults(res.results);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-3 bg-[#080b11] text-slate-200">
        <Disc3 className="w-10 h-10 text-emerald-400 animate-spin" />
        <p className="text-sm font-medium tracking-wide text-slate-400">
          Loading Neural Models, FAISS Vector Index & Music Catalog...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center p-8 bg-[#080b11] text-center">
        <div className="max-w-lg rounded-2xl bg-rose-950/30 border border-rose-500/30 p-6">
          <h2 className="text-lg font-semibold text-rose-300 mb-2">
            Backend Engine Unavailable
          </h2>
          <p className="text-sm text-slate-300 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 rounded-lg bg-emerald-500 text-slate-950 font-semibold text-sm hover:bg-emerald-400 transition"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen pb-32 bg-[#080b11] text-slate-100">
      {/* Toast Notification */}
      {toastMsg && (
        <div className="fixed top-5 right-5 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-emerald-500/95 text-slate-950 font-medium text-xs shadow-2xl shadow-emerald-500/20">
          <Sparkles className="w-4 h-4 shrink-0" />
          <span>{toastMsg}</span>
        </div>
      )}

      {/* Top Navigation Header */}
      <header className="sticky top-0 z-30 backdrop-blur-xl bg-[#080b11]/85 border-b border-white/10">
        <div className="max-w-[1480px] mx-auto px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center text-slate-950 font-bold shadow-lg shadow-emerald-500/20">
              <Disc3 className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight flex items-center gap-2">
                Resonance Studio
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  Hybrid DL + Adaptive Shuffle
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Multimodal NeuMF · Causal Session Transformer · LinUCB Feedback · Probabilistic Shuffle
              </p>
            </div>
          </div>

          {/* Primary Navigation Tabs */}
          <nav className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-white/10">
            <button
              onClick={() => setActiveTab("discover")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === "discover"
                  ? "bg-emerald-500 text-slate-950 shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              Personalized Discovery
            </button>
            <button
              onClick={() => setActiveTab("shuffle")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === "shuffle"
                  ? "bg-emerald-500 text-slate-950 shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <Shuffle className="w-3.5 h-3.5" />
              Adaptive Shuffle Arena
            </button>
            <button
              onClick={() => setActiveTab("benchmarks")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition ${
                activeTab === "benchmarks"
                  ? "bg-emerald-500 text-slate-950 shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Model Benchmarks
            </button>
          </nav>

          {/* Listener Persona Selector */}
          <div className="flex items-center gap-2.5">
            <User className="w-4 h-4 text-emerald-400" />
            <select
              value={selectedUserIdx}
              onChange={(e) => setSelectedUserIdx(Number(e.target.value))}
              className="bg-slate-900 border border-white/15 rounded-lg px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
            >
              {users.map((u) => (
                <option key={u.user_idx} value={u.user_idx}>
                  Listener #{u.user_idx} ({u.country}) — {u.top_genres.slice(0, 2).join(", ")}
                </option>
              ))}
            </select>
            <button
              onClick={handleResetFeedback}
              title="Reset live session feedback"
              className="p-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-400 hover:text-white hover:border-white/25 transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1480px] mx-auto px-6 pt-6">
        {/* ==================== TAB 1: PERSONALIZED DISCOVERY ==================== */}
        {activeTab === "discover" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Sidebar: Control Deck & Dual-Horizon User State */}
            <aside className="lg:col-span-4 space-y-5">
              {/* Sliders Card */}
              <div className="rounded-2xl bg-[#101521] border border-white/10 p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-emerald-400" />
                    Personalization Control Deck
                  </h2>
                  <span className="text-[11px] text-purple-300 font-medium">
                    {Math.round((recData?.discovery_ratio_actual ?? discoveryLevel) * 100)}% New Discovery
                  </span>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Familiarity ↔ Discovery</span>
                    <span className="font-mono text-emerald-400">
                      {Math.round(discoveryLevel * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={discoveryLevel}
                    onChange={(e) => setDiscoveryLevel( parseFloat(e.target.value) )}
                    className="w-full"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>Comfort Favorites</span>
                    <span>Balanced</span>
                    <span>Pure Exploration</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Genre & Artist Diversity (MMR)</span>
                    <span className="font-mono text-sky-400">
                      {Math.round(diversityLevel * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={diversityLevel}
                    onChange={(e) => setDiversityLevel(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Hidden-Gem Novelty Boost</span>
                    <span className="font-mono text-purple-400">
                      {Math.round(noveltyBoost * 100)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={noveltyBoost}
                    onChange={(e) => setNoveltyBoost(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div className="pt-2 border-t border-white/10">
                  <div className="flex items-center justify-between text-xs mb-2">
                    <span className="text-slate-300">Preference Horizon Gate</span>
                    <label className="flex items-center gap-1.5 text-[11px] text-emerald-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoHorizonGate}
                        onChange={(e) => setAutoHorizonGate(e.target.checked)}
                        className="rounded accent-emerald-500"
                      />
                      Auto Neural Gate
                    </label>
                  </div>
                  {!autoHorizonGate ? (
                    <div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={manualShortTerm}
                        onChange={(e) => setManualShortTerm(parseFloat(e.target.value))}
                        className="w-full"
                      />
                      <div className="flex justify-between text-[10px] text-slate-400">
                        <span>Long-Term ({Math.round((1 - manualShortTerm) * 100)}%)</span>
                        <span>Session ({Math.round(manualShortTerm * 100)}%)</span>
                      </div>
                    </div>
                  ) : (
                    <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden flex">
                      <div
                        style={{ width: `${(profile?.long_term_gate ?? 0.5) * 100}%` }}
                        className="bg-emerald-500 h-full transition-all"
                      />
                      <div
                        style={{ width: `${(profile?.short_term_gate ?? 0.5) * 100}%` }}
                        className="bg-sky-400 h-full transition-all"
                      />
                    </div>
                  )}
                  <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                    <span>Long-Term NeuMF: {Math.round((recData?.long_term_gate ?? 0.5) * 100)}%</span>
                    <span>Session Transformer: {Math.round((recData?.short_term_gate ?? 0.5) * 100)}%</span>
                  </div>
                </div>
              </div>

              {/* Dual-Horizon Radar & Bandit State */}
              {profile && (
                <div className="rounded-2xl bg-[#101521] border border-white/10 p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-sm font-semibold flex items-center gap-2">
                      <Activity className="w-4 h-4 text-sky-400" />
                      Sonic Taste Fingerprint
                    </h2>
                    <span className="text-[11px] text-slate-400">
                      {profile.feedback_summary.num_feedback_events} live actions
                    </span>
                  </div>

                  <DualHorizonRadar
                    longTerm={profile.long_term_radar}
                    shortTerm={profile.short_term_radar}
                  />

                  <div className="grid grid-cols-3 gap-2 pt-2 text-center">
                    <div className="rounded-xl bg-slate-900/80 p-2 border border-white/5">
                      <div className="text-xs font-bold text-emerald-400">
                        {profile.feedback_summary.liked_count}
                      </div>
                      <div className="text-[10px] text-slate-400">Session Likes</div>
                    </div>
                    <div className="rounded-xl bg-slate-900/80 p-2 border border-white/5">
                      <div className="text-xs font-bold text-rose-400">
                        {profile.feedback_summary.skipped_count +
                          profile.feedback_summary.disliked_count}
                      </div>
                      <div className="text-[10px] text-slate-400">Skips / Suppressed</div>
                    </div>
                    <div className="rounded-xl bg-slate-900/80 p-2 border border-white/5">
                      <div className="text-xs font-bold text-sky-400">
                        {profile.feedback_summary.residual_magnitude.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-slate-400">Bandit Drift ||Δu||</div>
                    </div>
                  </div>

                  {/* Top Historical Genres */}
                  <div>
                    <div className="text-[11px] text-slate-400 mb-1.5 font-medium">
                      Core Genre Affinity
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {profile.genre_breakdown.map((g) => (
                        <span
                          key={g.genre}
                          className="text-[11px] px-2.5 py-0.5 rounded-full bg-slate-900 border border-white/10 text-slate-300"
                        >
                          {g.genre}{" "}
                          <span className="text-emerald-400">
                            {Math.round(g.share * 100)}%
                          </span>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Recent Session & Feedback Stream */}
              {profile && (
                <div className="rounded-2xl bg-[#101521] border border-white/10 p-5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                    Recent Sequence Context (Transformer Input)
                  </h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {[...profile.recent_timeline].reverse().map((ev, i) => (
                      <div
                        key={`${ev.item_idx}-${i}`}
                        className="flex items-center justify-between gap-2 text-xs p-2 rounded-lg bg-slate-900/70 border border-white/5"
                      >
                        <div className="truncate">
                          <div className="font-medium text-slate-200 truncate">
                            {ev.track_name}
                          </div>
                          <div className="text-[11px] text-slate-400 truncate">
                            {ev.artist_name} · {ev.genre}
                          </div>
                        </div>
                        <span
                          className={`shrink-0 text-[10px] px-2 py-0.5 rounded-full font-medium ${
                            ev.action === "early_skip" || ev.action === "dislike"
                              ? "bg-rose-500/15 text-rose-300 border border-rose-500/30"
                              : ev.source === "live_feedback"
                              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                              : "bg-slate-800 text-slate-300"
                          }`}
                        >
                          {ev.action.replace("_", " ")}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </aside>

            {/* Right Main Column: Catalog Search + Personalized Recommendation Grid */}
            <section className="lg:col-span-8 space-y-5">
              {/* Search Bar to Inject Immediate Feedback on Any Track */}
              <div className="rounded-2xl bg-[#101521] border border-white/10 p-4">
                <div className="relative">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-2.5" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => handleSearch(e.target.value)}
                    placeholder="Search 40,000+ Spotify tracks, artists, or genres to simulate instant Like / Early Skip..."
                    className="w-full bg-slate-950 border border-white/10 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500"
                  />
                </div>
                {searchResults.length > 0 && (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 max-h-56 overflow-y-auto">
                    {searchResults.map((tr) => (
                      <div
                        key={tr.item_idx}
                        className="flex items-center justify-between gap-2 p-2.5 rounded-xl bg-slate-900 border border-white/10"
                      >
                        <div className="truncate">
                          <div className="text-xs font-semibold text-white truncate">
                            {tr.track_name}
                          </div>
                          <div className="text-[11px] text-slate-400 truncate">
                            {tr.artist_name} · <span className="text-emerald-400">{tr.genre}</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-1 shrink-0">
                          <button
                            onClick={() => {
                              setActiveTrack(tr);
                              setSearchResults([]);
                              setSearchQuery("");
                            }}
                            className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30"
                            title="Play Preview"
                          >
                            <Play className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleFeedback(tr, "like")}
                            className="p-1.5 rounded-lg bg-pink-500/20 text-pink-300 hover:bg-pink-500/30"
                            title="Like Track"
                          >
                            <Heart className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={() => handleFeedback(tr, "early_skip")}
                            className="p-1.5 rounded-lg bg-rose-500/20 text-rose-300 hover:bg-rose-500/30"
                            title="Simulate Early Skip"
                          >
                            <SkipForward className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Recommendations Header */}
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-lg font-bold flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-emerald-400" />
                    Adaptive Next-Track Recommendations
                  </h2>
                  <p className="text-xs text-slate-400">
                    Click Like, Early Skip (&lt;15%), Repeat, or Dislike on any card to watch the Transformer &amp; LinUCB Bandit re-rank live.
                  </p>
                </div>
                <button
                  onClick={() => {
                    setActiveTab("shuffle");
                    setSelectedPlaylistPid(null);
                  }}
                  className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-xs font-semibold hover:bg-emerald-500/25 transition"
                >
                  <Shuffle className="w-3.5 h-3.5" />
                  Adaptive-Shuffle This Mix
                </button>
              </div>

              {/* Recommendation Cards Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                {recData?.recommendations.map((tr: RecommendedTrack, idx: number) => (
                  <div
                    key={tr.item_idx}
                    className="group rounded-2xl bg-[#111726] hover:bg-[#161e31] border border-white/10 hover:border-emerald-500/40 p-4 transition flex flex-col justify-between gap-3"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-[11px] font-mono text-slate-500">
                              #{idx + 1}
                            </span>
                            <h3 className="text-sm font-semibold text-white truncate">
                              {tr.track_name}
                            </h3>
                          </div>
                          <p className="text-xs text-slate-400 truncate mt-0.5">
                            {tr.artist_name} · <span className="text-slate-500">{tr.album_name}</span>
                          </p>
                        </div>

                        <span
                          className={`shrink-0 text-[10px] font-semibold px-2.5 py-0.5 rounded-full border ${
                            tr.is_discovery
                              ? "bg-purple-500/15 text-purple-300 border-purple-500/30"
                              : "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
                          }`}
                        >
                          {tr.is_discovery ? "Discovery" : "Familiar"}
                        </span>
                      </div>

                      {/* Genre & Audio Pills */}
                      <div className="flex flex-wrap items-center gap-1.5 mt-2.5 text-[11px]">
                        <span className="px-2 py-0.5 rounded-md bg-slate-900 text-sky-300 border border-white/5">
                          {tr.genre}
                        </span>
                        <span className="px-2 py-0.5 rounded-md bg-slate-900 text-slate-300">
                          {Math.round(tr.audio.tempo)} BPM
                        </span>
                        <span className="px-2 py-0.5 rounded-md bg-slate-900 text-slate-300">
                          Energy {Math.round(tr.audio.energy * 100)}%
                        </span>
                        <span className="px-2 py-0.5 rounded-md bg-slate-900 text-purple-300">
                          Novelty {Math.round(tr.novelty * 100)}%
                        </span>
                      </div>

                      {/* AI Explanation Reasons */}
                      <div className="mt-2.5 space-y-1">
                        {tr.explanations.map((reason, rIdx) => (
                          <div
                            key={rIdx}
                            className="text-[11px] text-slate-300 flex items-center gap-1.5"
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                            <span className="truncate">{reason}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Interactive Player & Feedback Bar */}
                    <div className="pt-2.5 border-t border-white/5 flex items-center justify-between gap-2">
                      <button
                        onClick={() =>
                          setActiveTrack({
                            item_idx: tr.item_idx,
                            spotify_track_id: tr.spotify_track_id,
                            track_name: tr.track_name,
                            artist_name: tr.artist_name,
                            genre: tr.genre,
                          })
                        }
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500 text-slate-950 font-semibold text-xs hover:bg-emerald-400 transition"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        Play
                      </button>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleFeedback(tr, "like")}
                          title="Explicit Like (+1.0)"
                          className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-300 transition"
                        >
                          <Heart className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(tr, "repeat")}
                          title="Simulate Repeat Listen (+0.85)"
                          className="p-1.5 rounded-lg bg-slate-900 hover:bg-sky-500/20 text-slate-300 hover:text-sky-300 transition"
                        >
                          <Repeat className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(tr, "complete_play")}
                          title="Simulate 100% Full Play (+0.55)"
                          className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-300 transition"
                        >
                          <CheckCircle2 className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(tr, "early_skip")}
                          title="Simulate Early Skip <15% (-0.75)"
                          className="px-2 py-1 rounded-lg bg-slate-900 hover:bg-rose-500/20 text-slate-300 hover:text-rose-300 text-[11px] font-medium flex items-center gap-1 transition"
                        >
                          <SkipForward className="w-3.5 h-3.5" />
                          Skip
                        </button>
                        <button
                          onClick={() => handleFeedback(tr, "dislike")}
                          title="Explicit Dislike (-1.0)"
                          className="p-1.5 rounded-lg bg-slate-900 hover:bg-rose-500/25 text-slate-400 hover:text-rose-400 transition"
                        >
                          <ThumbsDown className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          </div>
        )}

        {/* ==================== TAB 2: ADAPTIVE SHUFFLE ARENA ==================== */}
        {activeTab === "shuffle" && (
          <div className="space-y-6">
            {/* Shuffle Controls Deck */}
            <div className="rounded-2xl bg-[#101521] border border-white/10 p-5">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
                <div>
                  <h2 className="text-lg font-bold flex items-center gap-2">
                    <Shuffle className="w-5 h-5 text-emerald-400" />
                    Adaptive Shuffle vs. Standard Baselines
                  </h2>
                  <p className="text-xs text-slate-400">
                    Compare identical song pools sequenced by Fisher-Yates Random, Spotify Artist-Spaced Heuristic, and our Constrained Probabilistic Adaptive Shuffle.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <select
                    value={selectedPlaylistPid === null ? "personalized" : String(selectedPlaylistPid)}
                    onChange={(e) =>
                      setSelectedPlaylistPid(
                        e.target.value === "personalized" ? null : Number(e.target.value)
                      )
                    }
                    className="bg-slate-950 border border-white/15 rounded-xl px-3.5 py-2 text-xs text-slate-100"
                  >
                    <option value="personalized">
                      ★ User&apos;s Personalized Hybrid Pool (32 tracks)
                    </option>
                    {playlists.map((pl) => (
                      <option key={pl.pid} value={pl.pid}>
                        Playlist: {pl.name} ({pl.num_tracks} tracks · {pl.unique_artists} artists)
                      </option>
                    ))}
                  </select>

                  <button
                    onClick={() => runShuffleComparison(true)}
                    className="flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 text-slate-950 font-semibold text-xs hover:bg-emerald-400 transition shadow-lg shadow-emerald-500/20"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    Re-Shuffle (New Seed)
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 pt-3 border-t border-white/10">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Diversity & Anti-Clump Weight</span>
                    <span className="font-mono text-emerald-400">{shuffleDiversity.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1.4}
                    step={0.05}
                    value={shuffleDiversity}
                    onChange={(e) => setShuffleDiversity(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Acoustic & BPM Flow Smoothness</span>
                    <span className="font-mono text-sky-400">{shuffleFlow.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={1.4}
                    step={0.05}
                    value={shuffleFlow}
                    onChange={(e) => setShuffleFlow(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Min Artist Cooldown Gap</span>
                    <span className="font-mono text-amber-300">{artistCooldown} songs</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={8}
                    step={1}
                    value={artistCooldown}
                    onChange={(e) => setArtistCooldown(parseInt(e.target.value, 10))}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Unpredictability (Temperature τ)</span>
                    <span className="font-mono text-purple-300">{entropyTemp.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.04}
                    max={0.65}
                    step={0.02}
                    value={entropyTemp}
                    onChange={(e) => setEntropyTemp(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>
            </div>

            {/* 3-Column Head-to-Head Comparison */}
            {shuffleData && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
                {[
                  {
                    title: "Proposed Adaptive Shuffle",
                    subtitle: "Relevance + Acoustic Flow + Cooldown + Boltzmann Sampling",
                    data: shuffleData.adaptive,
                    accent: "#1DB954",
                    badgeClass: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
                  },
                  {
                    title: "Fisher-Yates True Random",
                    subtitle: "Uniform Permutation Baseline (Prone to Clumping)",
                    data: shuffleData.fisher_yates,
                    accent: "#F43F5E",
                    badgeClass: "bg-rose-500/15 text-rose-300 border-rose-500/30",
                  },
                  {
                    title: "Spotify-Style Artist Spaced",
                    subtitle: "Heuristic Interval Spacing (Ignores Genre & Audio Whiplash)",
                    data: shuffleData.artist_spaced,
                    accent: "#38BDF8",
                    badgeClass: "bg-sky-500/15 text-sky-300 border-sky-500/30",
                  },
                ].map((col) => {
                  const m = col.data.metrics;
                  return (
                    <div
                      key={col.title}
                      className="rounded-2xl bg-[#101521] border border-white/10 p-4 flex flex-col gap-4"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <h3 className="text-sm font-bold text-white">{col.title}</h3>
                          <p className="text-[11px] text-slate-400">{col.subtitle}</p>
                        </div>
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${col.badgeClass}`}
                        >
                          Seed #{shuffleData.seed}
                        </span>
                      </div>

                      {/* Metrics Scorecard */}
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                          <div className="text-[10px] text-slate-400">Artist Clump Rate</div>
                          <div
                            className={`text-sm font-bold ${
                              m.artist_clump_rate === 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {(m.artist_clump_rate * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                          <div className="text-[10px] text-slate-400">Min Artist Gap</div>
                          <div className="text-sm font-bold text-slate-100">
                            {m.min_artist_gap} tracks
                          </div>
                        </div>
                        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                          <div className="text-[10px] text-slate-400">Acoustic Flow</div>
                          <div className="text-sm font-bold text-sky-400">
                            {(m.acoustic_smoothness * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                          <div className="text-[10px] text-slate-400">Mean BPM Jump</div>
                          <div className="text-sm font-bold text-amber-300">
                            ±{m.mean_bpm_jump} BPM
                          </div>
                        </div>
                      </div>

                      {/* Energy Sparkline + Clump Dots */}
                      <EnergyFlowSparkline tracks={col.data.tracks} accentColor={col.accent} />

                      {/* Ordered Queue List */}
                      <div className="space-y-1.5 max-h-[440px] overflow-y-auto pr-1">
                        {col.data.tracks.map((tr, idx) => {
                          const prevArtistMatch =
                            (idx >= 1 && col.data.tracks[idx - 1].artist_name === tr.artist_name) ||
                            (idx >= 2 && col.data.tracks[idx - 2].artist_name === tr.artist_name);
                          return (
                            <div
                              key={`${tr.item_idx}-${idx}`}
                              onClick={() =>
                                setActiveTrack({
                                  item_idx: tr.item_idx,
                                  spotify_track_id: tr.spotify_track_id,
                                  track_name: tr.track_name,
                                  artist_name: tr.artist_name,
                                  genre: tr.genre,
                                })
                              }
                              className={`cursor-pointer p-2.5 rounded-xl border transition flex items-center justify-between gap-2 ${
                                prevArtistMatch
                                  ? "bg-rose-950/30 border-rose-500/40"
                                  : "bg-slate-900/70 hover:bg-slate-800/80 border-white/5"
                              }`}
                            >
                              <div className="min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="text-[11px] font-mono text-slate-500">
                                    {tr.position}.
                                  </span>
                                  <span className="text-xs font-semibold text-white truncate">
                                    {tr.track_name}
                                  </span>
                                  {prevArtistMatch && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 font-semibold">
                                      CLUMP
                                    </span>
                                  )}
                                </div>
                                <div className="text-[11px] text-slate-400 truncate">
                                  {tr.artist_name} ·{" "}
                                  <span className="text-sky-300">{tr.genre}</span> ·{" "}
                                  {Math.round(tr.audio.tempo)} BPM
                                </div>
                                <div className="text-[10px] text-emerald-400/90 truncate mt-0.5">
                                  {tr.step_diagnostics.reason}
                                </div>
                              </div>
                              <Play className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ==================== TAB 3: MODEL BENCHMARKS ==================== */}
        {activeTab === "benchmarks" && benchmarks && (
          <div className="space-y-6">
            {/* Dataset & Training Hardware Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                { label: "Active Tracks", val: benchmarks.dataset_stats.num_tracks.toLocaleString() },
                { label: "Unique Artists", val: benchmarks.dataset_stats.num_artists.toLocaleString() },
                { label: "Genres", val: benchmarks.dataset_stats.num_genres.toLocaleString() },
                { label: "Listeners", val: benchmarks.dataset_stats.num_users.toLocaleString() },
                {
                  label: "Timestamped Events",
                  val: benchmarks.dataset_stats.num_interactions.toLocaleString(),
                },
                { label: "Compute Device", val: benchmarks.device },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-2xl bg-[#101521] border border-white/10 p-4"
                >
                  <div className="text-[11px] text-slate-400">{stat.label}</div>
                  <div className="text-base font-bold text-emerald-400 mt-1 truncate">
                    {stat.val}
                  </div>
                </div>
              ))}
            </div>

            {/* Suite A: Recommendation Models Table */}
            <div className="rounded-2xl bg-[#101521] border border-white/10 p-5 overflow-x-auto">
              <h2 className="text-base font-bold mb-1 flex items-center gap-2">
                <Flame className="w-4 h-4 text-emerald-400" />
                Suite A — Next-Track Recommendation Models (Leave-Last-Out Temporal Split)
              </h2>
              <p className="text-xs text-slate-400 mb-4">
                Evaluated across 500-candidate ranking pools per test listener plus full-catalog Top-20 diversity, self-information novelty, and catalog coverage.
              </p>
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400">
                    <th className="py-2.5 pr-4">Model</th>
                    <th className="py-2.5 px-3">HR@10</th>
                    <th className="py-2.5 px-3">NDCG@10</th>
                    <th className="py-2.5 px-3">MRR@10</th>
                    <th className="py-2.5 px-3">Diversity (ILD@20)</th>
                    <th className="py-2.5 px-3">Novelty@20</th>
                    <th className="py-2.5 px-3">Catalog Coverage</th>
                    <th className="py-2.5 pl-3">Unseen Artist %</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(benchmarks.recommendation_benchmarks).map(([model, m]) => {
                    const isOurs = model.includes("Proposed") || model.includes("Transformer");
                    return (
                      <tr
                        key={model}
                        className={`border-b border-white/5 ${
                          isOurs ? "bg-emerald-500/10 font-semibold text-white" : "text-slate-300"
                        }`}
                      >
                        <td className="py-2.5 pr-4">{model}</td>
                        <td className="py-2.5 px-3">{(m["HR@10"] * 100).toFixed(1)}%</td>
                        <td className="py-2.5 px-3 text-emerald-400">{m["NDCG@10"].toFixed(4)}</td>
                        <td className="py-2.5 px-3">{m["MRR@10"].toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-sky-300">{m["ILD@20"].toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-purple-300">{m["Novelty@20"].toFixed(4)}</td>
                        <td className="py-2.5 px-3">
                          {(m["CatalogCoverage@20"] * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 pl-3">
                          {(m["UnseenArtistRatio@20"] * 100).toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Suite B: Adaptive Shuffle Evaluation Table */}
            <div className="rounded-2xl bg-[#101521] border border-white/10 p-5 overflow-x-auto">
              <h2 className="text-base font-bold mb-1 flex items-center gap-2">
                <Shuffle className="w-4 h-4 text-sky-400" />
                Suite B — Shuffle Algorithm Benchmarks (200 Real Curated Spotify Playlists)
              </h2>
              <p className="text-xs text-slate-400 mb-4">
                Lower Artist Clump Rate and Mean BPM Jump indicate smoother, non-repetitive sessions; higher Acoustic Smoothness, Window Diversity, and Session NDCG indicate cohesive personalization.
              </p>
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-white/10 text-slate-400">
                    <th className="py-2.5 pr-4">Algorithm</th>
                    <th className="py-2.5 px-3">Artist Clump Rate ↓</th>
                    <th className="py-2.5 px-3">Min Artist Gap ↑</th>
                    <th className="py-2.5 px-3">Genre Switch % ↑</th>
                    <th className="py-2.5 px-3">Window Diversity ↑</th>
                    <th className="py-2.5 px-3">Acoustic Flow ↑</th>
                    <th className="py-2.5 px-3">Mean BPM Jump ↓</th>
                    <th className="py-2.5 pl-3">Session NDCG ↑</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(benchmarks.shuffle_benchmarks).map(([alg, m]) => {
                    const isOurs = alg.includes("Proposed");
                    return (
                      <tr
                        key={alg}
                        className={`border-b border-white/5 ${
                          isOurs ? "bg-emerald-500/10 font-semibold text-white" : "text-slate-300"
                        }`}
                      >
                        <td className="py-2.5 pr-4">{alg}</td>
                        <td className="py-2.5 px-3 text-rose-300">
                          {(m.artist_clump_rate * 100).toFixed(2)}%
                        </td>
                        <td className="py-2.5 px-3">{m.min_artist_gap.toFixed(2)}</td>
                        <td className="py-2.5 px-3">
                          {(m.genre_switch_rate * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3">{m.local_window_diversity.toFixed(4)}</td>
                        <td className="py-2.5 px-3 text-emerald-400">
                          {(m.acoustic_smoothness * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3">±{m.mean_bpm_jump.toFixed(1)} BPM</td>
                        <td className="py-2.5 pl-3 text-sky-300">{m.session_ndcg.toFixed(4)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Suite C: Online Feedback Adaptation */}
            <div className="rounded-2xl bg-[#101521] border border-white/10 p-5">
              <h2 className="text-base font-bold mb-1">
                Suite C — Real-Time Session Feedback Adaptation (Skip Fatigue & Mood Pivot)
              </h2>
              <p className="text-xs text-slate-400 mb-3">
                Simulated across 100 sessions: listener early-skips 3 tracks of their dominant genre and likes/repeats 3 tracks of a newly discovered genre.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                {Object.entries(benchmarks.feedback_adaptation_benchmarks).map(([stage, vals]) => (
                  <div
                    key={stage}
                    className="p-4 rounded-xl bg-slate-900/80 border border-white/10 flex items-center justify-between"
                  >
                    <div>
                      <div className="font-semibold text-white">
                        {stage.replaceAll("_", " ")}
                      </div>
                      <div className="text-slate-400 mt-1">
                        Skipped Genre Share:{" "}
                        <span className="text-rose-400 font-bold">
                          {(vals["FatiguedGenreShare@20"] * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-slate-400">New Liked Genre Share</div>
                      <div className="text-lg font-bold text-emerald-400">
                        {(vals["NewTargetGenreShare@20"] * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Persistent Spotify Embed & Live Feedback Dock */}
      {activeTrack && (
        <footer className="fixed bottom-0 inset-x-0 z-40 backdrop-blur-2xl bg-[#0b101b]/95 border-t border-white/15 px-4 py-2.5">
          <div className="max-w-[1480px] mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
            <div className="w-full md:w-[440px] h-[80px] rounded-xl overflow-hidden bg-slate-900">
              <iframe
                key={activeTrack.spotify_track_id}
                src={`https://open.spotify.com/embed/track/${activeTrack.spotify_track_id}?utm_source=generator&theme=0`}
                width="100%"
                height="80"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                loading="lazy"
                className="border-0"
              />
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-slate-400 mr-1 hidden xl:inline">
                Simulate Listener Feedback on{" "}
                <strong className="text-slate-200">{activeTrack.track_name}</strong>:
              </span>
              <button
                onClick={() => handleFeedback(activeTrack, "like")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-semibold hover:bg-emerald-500/30 transition"
              >
                <Heart className="w-3.5 h-3.5" />
                Like (+1.0)
              </button>
              <button
                onClick={() => handleFeedback(activeTrack, "repeat")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-500/20 border border-sky-500/40 text-sky-300 text-xs font-semibold hover:bg-sky-500/30 transition"
              >
                <Repeat className="w-3.5 h-3.5" />
                Replay (+0.85)
              </button>
              <button
                onClick={() => handleFeedback(activeTrack, "complete_play")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 border border-white/10 text-slate-200 text-xs font-medium hover:bg-slate-700 transition"
              >
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                Full Listen (+0.55)
              </button>
              <button
                onClick={() => handleFeedback(activeTrack, "early_skip")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold hover:bg-rose-500/30 transition"
              >
                <SkipForward className="w-3.5 h-3.5" />
                Early Skip &lt;15% (-0.75)
              </button>
              <button
                onClick={() => handleFeedback(activeTrack, "dislike")}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 border border-rose-500/30 text-rose-400 text-xs font-medium hover:bg-rose-950/60 transition"
              >
                <ThumbsDown className="w-3.5 h-3.5" />
                Dislike
              </button>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
