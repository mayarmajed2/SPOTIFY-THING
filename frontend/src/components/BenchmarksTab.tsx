"use client";

import React from "react";
import { Flame, Shuffle } from "lucide-react";
import { BenchmarkReport } from "@/lib/api";

export function BenchmarksTab({
  benchmarks,
}: {
  benchmarks: BenchmarkReport | null;
}) {
  if (!benchmarks) return null;

  return (
    <div className="space-y-6">
      {/* Dataset & Training Hardware Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        {[
          {
            label: "Active Tracks",
            val: benchmarks.dataset_stats.num_tracks.toLocaleString(),
          },
          {
            label: "Unique Artists",
            val: benchmarks.dataset_stats.num_artists.toLocaleString(),
          },
          {
            label: "Genres",
            val: benchmarks.dataset_stats.num_genres.toLocaleString(),
          },
          {
            label: "Listeners",
            val: benchmarks.dataset_stats.num_users.toLocaleString(),
          },
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
            {Object.entries(benchmarks.recommendation_benchmarks).map(
              ([model, m]) => {
                const isOurs =
                  model.includes("Proposed") || model.includes("Transformer");
                return (
                  <tr
                    key={model}
                    className={`border-b border-white/5 ${
                      isOurs
                        ? "bg-emerald-500/10 font-semibold text-white"
                        : "text-slate-300"
                    }`}
                  >
                    <td className="py-2.5 pr-4">{model}</td>
                    <td className="py-2.5 px-3">
                      {(m["HR@10"] * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-3 text-emerald-400">
                      {m["NDCG@10"].toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">{m["MRR@10"].toFixed(4)}</td>
                    <td className="py-2.5 px-3 text-sky-300">
                      {m["ILD@20"].toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3 text-purple-300">
                      {m["Novelty@20"].toFixed(4)}
                    </td>
                    <td className="py-2.5 px-3">
                      {(m["CatalogCoverage@20"] * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 pl-3">
                      {(m["UnseenArtistRatio@20"] * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              }
            )}
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
                    isOurs
                      ? "bg-emerald-500/10 font-semibold text-white"
                      : "text-slate-300"
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
                  <td className="py-2.5 px-3">
                    {m.local_window_diversity.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 text-emerald-400">
                    {(m.acoustic_smoothness * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-3">
                    ±{m.mean_bpm_jump.toFixed(1)} BPM
                  </td>
                  <td className="py-2.5 pl-3 text-sky-300">
                    {m.session_ndcg.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Suite C: Online Feedback Adaptation */}
      <div className="rounded-2xl bg-[#101521] border border-white/10 p-5">
        <h2 className="text-base font-bold mb-1">
          Suite C — Real-Time Session Feedback Adaptation (Skip Fatigue &amp; Mood Pivot)
        </h2>
        <p className="text-xs text-slate-400 mb-3">
          Simulated across 100 sessions: listener early-skips 3 tracks of their dominant genre and likes/repeats 3 tracks of a newly discovered genre.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {Object.entries(benchmarks.feedback_adaptation_benchmarks).map(
            ([stage, vals]) => (
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
            )
          )}
        </div>
      </div>
    </div>
  );
}
