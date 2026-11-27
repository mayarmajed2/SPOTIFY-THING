"use client";

import React from "react";
import {
  Activity,
  CheckCircle2,
  Heart,
  Play,
  Repeat,
  Search,
  Shuffle,
  SkipForward,
  Sliders,
  Sparkles,
  ThumbsDown,
} from "lucide-react";
import { RecommendedTrack, RecommendResponse, UserProfile } from "@/lib/api";
import { DualHorizonRadar } from "@/components/VisualCharts";
import { ActivePlayerTrack, FeedbackAction } from "@/components/PlayerDock";

interface SearchHit {
  item_idx: number;
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  genre: string;
  popularity: number;
}

interface DiscoveryTabProps {
  profile: UserProfile | null;
  recData: RecommendResponse | null;
  discoveryLevel: number;
  setDiscoveryLevel: (v: number) => void;
  diversityLevel: number;
  setDiversityLevel: (v: number) => void;
  noveltyBoost: number;
  setNoveltyBoost: (v: number) => void;
  autoHorizonGate: boolean;
  setAutoHorizonGate: (v: boolean) => void;
  manualShortTerm: number;
  setManualShortTerm: (v: number) => void;
  searchQuery: string;
  searchResults: SearchHit[];
  onSearch: (q: string) => void;
  onClearSearch: () => void;
  onSelectTrack: (track: ActivePlayerTrack) => void;
  onFeedback: (track: ActivePlayerTrack, action: FeedbackAction) => void;
  onAdaptiveShuffleMix: () => void;
}

export function DiscoveryTab({
  profile,
  recData,
  discoveryLevel,
  setDiscoveryLevel,
  diversityLevel,
  setDiversityLevel,
  noveltyBoost,
  setNoveltyBoost,
  autoHorizonGate,
  setAutoHorizonGate,
  manualShortTerm,
  setManualShortTerm,
  searchQuery,
  searchResults,
  onSearch,
  onClearSearch,
  onSelectTrack,
  onFeedback,
  onAdaptiveShuffleMix,
}: DiscoveryTabProps) {
  return (
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
              {Math.round(
                (recData?.discovery_ratio_actual ?? discoveryLevel) * 100
              )}
              % New Discovery
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
              onChange={(e) => setDiscoveryLevel(parseFloat(e.target.value))}
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
              <span className="text-slate-300">
                Genre &amp; Artist Diversity (MMR)
              </span>
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
                  onChange={(e) =>
                    setManualShortTerm(parseFloat(e.target.value))
                  }
                  className="w-full"
                />
                <div className="flex justify-between text-[10px] text-slate-400">
                  <span>
                    Long-Term ({Math.round((1 - manualShortTerm) * 100)}%)
                  </span>
                  <span>Session ({Math.round(manualShortTerm * 100)}%)</span>
                </div>
              </div>
            ) : (
              <div className="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden flex">
                <div
                  style={{
                    width: `${(profile?.long_term_gate ?? 0.5) * 100}%`,
                  }}
                  className="bg-emerald-500 h-full transition-all"
                />
                <div
                  style={{
                    width: `${(profile?.short_term_gate ?? 0.5) * 100}%`,
                  }}
                  className="bg-sky-400 h-full transition-all"
                />
              </div>
            )}
            <div className="flex justify-between text-[11px] text-slate-400 mt-1">
              <span>
                Long-Term NeuMF:{" "}
                {Math.round((recData?.long_term_gate ?? 0.5) * 100)}%
              </span>
              <span>
                Session Transformer:{" "}
                {Math.round((recData?.short_term_gate ?? 0.5) * 100)}%
              </span>
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
                <div className="text-[10px] text-slate-400">
                  Skips / Suppressed
                </div>
              </div>
              <div className="rounded-xl bg-slate-900/80 p-2 border border-white/5">
                <div className="text-xs font-bold text-sky-400">
                  {profile.feedback_summary.residual_magnitude.toFixed(2)}
                </div>
                <div className="text-[10px] text-slate-400">
                  Bandit Drift ||Δu||
                </div>
              </div>
            </div>

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
        <div className="rounded-2xl bg-[#101521] border border-white/10 p-4">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-2.5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => onSearch(e.target.value)}
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
                      {tr.artist_name} ·{" "}
                      <span className="text-emerald-400">{tr.genre}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 shrink-0">
                    <button
                      onClick={() => {
                        onSelectTrack(tr);
                        onClearSearch();
                      }}
                      className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30"
                      title="Play Preview"
                    >
                      <Play className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => onFeedback(tr, "like")}
                      className="p-1.5 rounded-lg bg-pink-500/20 text-pink-300 hover:bg-pink-500/30"
                      title="Like Track"
                    >
                      <Heart className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => onFeedback(tr, "early_skip")}
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
            onClick={onAdaptiveShuffleMix}
            className="flex items-center gap-2 px-3.5 py-2 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-xs font-semibold hover:bg-emerald-500/25 transition"
          >
            <Shuffle className="w-3.5 h-3.5" />
            Adaptive-Shuffle This Mix
          </button>
        </div>

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
                      {tr.artist_name} ·{" "}
                      <span className="text-slate-500">{tr.album_name}</span>
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

              <div className="pt-2.5 border-t border-white/5 flex items-center justify-between gap-2">
                <button
                  onClick={() =>
                    onSelectTrack({
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
                    onClick={() => onFeedback(tr, "like")}
                    title="Explicit Like (+1.0)"
                    className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-300 transition"
                  >
                    <Heart className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onFeedback(tr, "repeat")}
                    title="Simulate Repeat Listen (+0.85)"
                    className="p-1.5 rounded-lg bg-slate-900 hover:bg-sky-500/20 text-slate-300 hover:text-sky-300 transition"
                  >
                    <Repeat className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onFeedback(tr, "complete_play")}
                    title="Simulate 100% Full Play (+0.55)"
                    className="p-1.5 rounded-lg bg-slate-900 hover:bg-emerald-500/20 text-slate-300 hover:text-emerald-300 transition"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={() => onFeedback(tr, "early_skip")}
                    title="Simulate Early Skip <15% (-0.75)"
                    className="px-2 py-1 rounded-lg bg-slate-900 hover:bg-rose-500/20 text-slate-300 hover:text-rose-300 text-[11px] font-medium flex items-center gap-1 transition"
                  >
                    <SkipForward className="w-3.5 h-3.5" />
                    Skip
                  </button>
                  <button
                    onClick={() => onFeedback(tr, "dislike")}
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
  );
}
