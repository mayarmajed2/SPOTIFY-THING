"use client";

import React from "react";
import { Play, RefreshCw, Shuffle } from "lucide-react";
import { PlaylistSummary, ShuffleCompareResponse } from "@/lib/api";
import { EnergyFlowSparkline } from "@/components/VisualCharts";
import { ActivePlayerTrack } from "@/components/PlayerDock";

interface ShuffleArenaTabProps {
  playlists: PlaylistSummary[];
  selectedPlaylistPid: number | null;
  setSelectedPlaylistPid: (pid: number | null) => void;
  shuffleDiversity: number;
  setShuffleDiversity: (v: number) => void;
  shuffleFlow: number;
  setShuffleFlow: (v: number) => void;
  artistCooldown: number;
  setArtistCooldown: (v: number) => void;
  entropyTemp: number;
  setEntropyTemp: (v: number) => void;
  shuffleData: ShuffleCompareResponse | null;
  onReshuffle: () => void;
  onSelectTrack: (track: ActivePlayerTrack) => void;
}

export function ShuffleArenaTab({
  playlists,
  selectedPlaylistPid,
  setSelectedPlaylistPid,
  shuffleDiversity,
  setShuffleDiversity,
  shuffleFlow,
  setShuffleFlow,
  artistCooldown,
  setArtistCooldown,
  entropyTemp,
  setEntropyTemp,
  shuffleData,
  onReshuffle,
  onSelectTrack,
}: ShuffleArenaTabProps) {
  return (
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
              value={
                selectedPlaylistPid === null
                  ? "personalized"
                  : String(selectedPlaylistPid)
              }
              onChange={(e) =>
                setSelectedPlaylistPid(
                  e.target.value === "personalized"
                    ? null
                    : Number(e.target.value)
                )
              }
              className="bg-slate-950 border border-white/15 rounded-xl px-3.5 py-2 text-xs text-slate-100"
            >
              <option value="personalized">
                ★ User&apos;s Personalized Hybrid Pool (32 tracks)
              </option>
              {playlists.map((pl) => (
                <option key={pl.pid} value={pl.pid}>
                  Playlist: {pl.name} ({pl.num_tracks} tracks ·{" "}
                  {pl.unique_artists} artists)
                </option>
              ))}
            </select>

            <button
              onClick={onReshuffle}
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
              <span className="text-slate-300">Diversity &amp; Anti-Clump Weight</span>
              <span className="font-mono text-emerald-400">
                {shuffleDiversity.toFixed(2)}
              </span>
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
              <span className="text-slate-300">Acoustic &amp; BPM Flow Smoothness</span>
              <span className="font-mono text-sky-400">
                {shuffleFlow.toFixed(2)}
              </span>
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
              <span className="font-mono text-amber-300">
                {artistCooldown} songs
              </span>
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
              <span className="font-mono text-purple-300">
                {entropyTemp.toFixed(2)}
              </span>
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
              subtitle:
                "Relevance + Acoustic Flow + Cooldown + Boltzmann Sampling",
              data: shuffleData.adaptive,
              accent: "#1DB954",
              badgeClass:
                "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
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
              subtitle:
                "Heuristic Interval Spacing (Ignores Genre & Audio Whiplash)",
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
                    <div className="text-[10px] text-slate-400">
                      Artist Clump Rate
                    </div>
                    <div
                      className={`text-sm font-bold ${
                        m.artist_clump_rate === 0
                          ? "text-emerald-400"
                          : "text-rose-400"
                      }`}
                    >
                      {(m.artist_clump_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                    <div className="text-[10px] text-slate-400">
                      Min Artist Gap
                    </div>
                    <div className="text-sm font-bold text-slate-100">
                      {m.min_artist_gap} tracks
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                    <div className="text-[10px] text-slate-400">
                      Acoustic Flow
                    </div>
                    <div className="text-sm font-bold text-sky-400">
                      {(m.acoustic_smoothness * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-slate-900/80 border border-white/5">
                    <div className="text-[10px] text-slate-400">
                      Mean BPM Jump
                    </div>
                    <div className="text-sm font-bold text-amber-300">
                      ±{m.mean_bpm_jump} BPM
                    </div>
                  </div>
                </div>

                {/* Energy Sparkline + Clump Dots */}
                <EnergyFlowSparkline
                  tracks={col.data.tracks}
                  accentColor={col.accent}
                />

                {/* Ordered Queue List */}
                <div className="space-y-1.5 max-h-[440px] overflow-y-auto pr-1">
                  {col.data.tracks.map((tr, idx) => {
                    const prevArtistMatch =
                      (idx >= 1 &&
                        col.data.tracks[idx - 1].artist_name ===
                          tr.artist_name) ||
                      (idx >= 2 &&
                        col.data.tracks[idx - 2].artist_name ===
                          tr.artist_name);
                    return (
                      <div
                        key={`${tr.item_idx}-${idx}`}
                        onClick={() =>
                          onSelectTrack({
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
  );
}
