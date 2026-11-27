"use client";

import React, { useCallback, useEffect, useState } from "react";
import {
  BarChart3,
  Compass,
  Disc3,
  RotateCcw,
  Shuffle,
  Sparkles,
  User,
} from "lucide-react";
import {
  api,
  BenchmarkReport,
  PlaylistSummary,
  RecommendResponse,
  ShuffleCompareResponse,
  UserPersona,
  UserProfile,
} from "@/lib/api";
import { DiscoveryTab } from "@/components/DiscoveryTab";
import { ShuffleArenaTab } from "@/components/ShuffleArenaTab";
import { BenchmarksTab } from "@/components/BenchmarksTab";
import {
  ActivePlayerTrack,
  FeedbackAction,
  PlayerDock,
} from "@/components/PlayerDock";

type ActiveTab = "discover" | "shuffle" | "benchmarks";

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
  const [selectedPlaylistPid, setSelectedPlaylistPid] = useState<number | null>(
    null
  );
  const [shuffleDiversity, setShuffleDiversity] = useState<number>(0.75);
  const [shuffleFlow, setShuffleFlow] = useState<number>(0.55);
  const [artistCooldown, setArtistCooldown] = useState<number>(4);
  const [entropyTemp, setEntropyTemp] = useState<number>(0.18);
  const [shuffleData, setShuffleData] = useState<ShuffleCompareResponse | null>(
    null
  );

  // Benchmarks, Search & Active Track
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
  const [activeTrack, setActiveTrack] = useState<ActivePlayerTrack | null>(
    null
  );
  const [toastMsg, setToastMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToastMsg(msg);
    setTimeout(() => setToastMsg(null), 3200);
  };

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
          seed: newRandomSeed
            ? Math.floor(Math.random() * 90000) + 1
            : undefined,
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
    track: ActivePlayerTrack,
    action: FeedbackAction
  ) => {
    const ratio = action === "early_skip" ? 0.1 : 1.0;
    await api.sendFeedback({
      user_idx: selectedUserIdx,
      item_idx: track.item_idx,
      action,
      played_ratio: ratio,
    });
    const labelMap: Record<FeedbackAction, string> = {
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
          Loading Neural Models, Vector Index &amp; Music Catalog...
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
      {toastMsg && (
        <div className="fixed top-5 right-5 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-emerald-500/95 text-slate-950 font-medium text-xs shadow-2xl shadow-emerald-500/20">
          <Sparkles className="w-4 h-4 shrink-0" />
          <span>{toastMsg}</span>
        </div>
      )}

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

          <div className="flex items-center gap-2.5">
            <User className="w-4 h-4 text-emerald-400" />
            <select
              value={selectedUserIdx}
              onChange={(e) => setSelectedUserIdx(Number(e.target.value))}
              className="bg-slate-900 border border-white/15 rounded-lg px-3 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
            >
              {users.map((u) => (
                <option key={u.user_idx} value={u.user_idx}>
                  Listener #{u.user_idx} ({u.country}) —{" "}
                  {u.top_genres.slice(0, 2).join(", ")}
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
        {activeTab === "discover" && (
          <DiscoveryTab
            profile={profile}
            recData={recData}
            discoveryLevel={discoveryLevel}
            setDiscoveryLevel={setDiscoveryLevel}
            diversityLevel={diversityLevel}
            setDiversityLevel={setDiversityLevel}
            noveltyBoost={noveltyBoost}
            setNoveltyBoost={setNoveltyBoost}
            autoHorizonGate={autoHorizonGate}
            setAutoHorizonGate={setAutoHorizonGate}
            manualShortTerm={manualShortTerm}
            setManualShortTerm={setManualShortTerm}
            searchQuery={searchQuery}
            searchResults={searchResults}
            onSearch={handleSearch}
            onClearSearch={() => {
              setSearchResults([]);
              setSearchQuery("");
            }}
            onSelectTrack={setActiveTrack}
            onFeedback={handleFeedback}
            onAdaptiveShuffleMix={() => {
              setActiveTab("shuffle");
              setSelectedPlaylistPid(null);
            }}
          />
        )}

        {activeTab === "shuffle" && (
          <ShuffleArenaTab
            playlists={playlists}
            selectedPlaylistPid={selectedPlaylistPid}
            setSelectedPlaylistPid={setSelectedPlaylistPid}
            shuffleDiversity={shuffleDiversity}
            setShuffleDiversity={setShuffleDiversity}
            shuffleFlow={shuffleFlow}
            setShuffleFlow={setShuffleFlow}
            artistCooldown={artistCooldown}
            setArtistCooldown={setArtistCooldown}
            entropyTemp={entropyTemp}
            setEntropyTemp={setEntropyTemp}
            shuffleData={shuffleData}
            onReshuffle={() => runShuffleComparison(true)}
            onSelectTrack={setActiveTrack}
          />
        )}

        {activeTab === "benchmarks" && (
          <BenchmarksTab benchmarks={benchmarks} />
        )}
      </main>

      <PlayerDock activeTrack={activeTrack} onFeedback={handleFeedback} />
    </div>
  );
}
