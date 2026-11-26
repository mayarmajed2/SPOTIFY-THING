"use client";

import React from "react";
import { ShuffledTrack } from "@/lib/api";

const RADAR_AXES = [
  { key: "danceability", label: "Dance" },
  { key: "energy", label: "Energy" },
  { key: "valence", label: "Valence" },
  { key: "acousticness", label: "Acoustic" },
  { key: "instrumentalness", label: "Instrum." },
  { key: "speechiness", label: "Vocal/Speech" },
];

export function DualHorizonRadar({
  longTerm,
  shortTerm,
}: {
  longTerm: Record<string, number>;
  shortTerm: Record<string, number>;
}) {
  const size = 230;
  const cx = size / 2;
  const cy = size / 2;
  const radius = 78;

  const toPoints = (vals: Record<string, number>) =>
    RADAR_AXES.map((ax, idx) => {
      const angle = (Math.PI * 2 * idx) / RADAR_AXES.length - Math.PI / 2;
      const v = Math.max(0.06, Math.min(1, vals[ax.key] ?? 0.2));
      const x = cx + radius * v * Math.cos(angle);
      const y = cy + radius * v * Math.sin(angle);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size} className="overflow-visible">
        {[0.25, 0.5, 0.75, 1.0].map((ring) => (
          <polygon
            key={ring}
            points={RADAR_AXES.map((_, idx) => {
              const angle = (Math.PI * 2 * idx) / RADAR_AXES.length - Math.PI / 2;
              return `${(cx + radius * ring * Math.cos(angle)).toFixed(1)},${(
                cy +
                radius * ring * Math.sin(angle)
              ).toFixed(1)}`;
            }).join(" ")}
            fill="none"
            stroke="rgba(148,163,184,0.16)"
            strokeWidth="1"
          />
        ))}
        {RADAR_AXES.map((ax, idx) => {
          const angle = (Math.PI * 2 * idx) / RADAR_AXES.length - Math.PI / 2;
          const x2 = cx + radius * Math.cos(angle);
          const y2 = cy + radius * Math.sin(angle);
          const lx = cx + (radius + 22) * Math.cos(angle);
          const ly = cy + (radius + 18) * Math.sin(angle);
          return (
            <g key={ax.key}>
              <line
                x1={cx}
                y1={cy}
                x2={x2}
                y2={y2}
                stroke="rgba(148,163,184,0.16)"
                strokeWidth="1"
              />
              <text
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-slate-400 text-[10px] font-medium"
              >
                {ax.label}
              </text>
            </g>
          );
        })}

        {/* Long-Term Taste Polygon */}
        <polygon
          points={toPoints(longTerm)}
          fill="rgba(29, 185, 84, 0.22)"
          stroke="#1DB954"
          strokeWidth="2"
        />

        {/* Short-Term Session Context Polygon */}
        <polygon
          points={toPoints(shortTerm)}
          fill="rgba(56, 189, 248, 0.20)"
          stroke="#38BDF8"
          strokeWidth="2"
          strokeDasharray="4 2"
        />
      </svg>

      <div className="flex items-center gap-4 text-xs text-slate-300 mt-1">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
          Long-Term Profile
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" />
          Active Session Mood
        </span>
      </div>
    </div>
  );
}

export function EnergyFlowSparkline({
  tracks,
  accentColor = "#1DB954",
}: {
  tracks: ShuffledTrack[];
  accentColor?: string;
}) {
  if (!tracks.length) return null;
  const w = 320;
  const h = 56;
  const pad = 6;

  // Detect artist clump positions (same artist within <= 2 steps)
  const clumpIndices = new Set<number>();
  const lastArtistPos = new Map<string, number>();
  tracks.forEach((t, idx) => {
    const prev = lastArtistPos.get(t.artist_name);
    if (prev !== undefined && idx - prev <= 2) {
      clumpIndices.add(idx);
      clumpIndices.add(prev);
    }
    lastArtistPos.set(t.artist_name, idx);
  });

  const points = tracks
    .map((t, idx) => {
      const x =
        pad + (idx / Math.max(1, tracks.length - 1)) * (w - pad * 2);
      const y = h - pad - t.audio.energy * (h - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="w-full">
      <div className="flex justify-between text-[11px] text-slate-400 mb-1">
        <span>Session Energy Curve (Track 1 → {tracks.length})</span>
        {clumpIndices.size > 0 ? (
          <span className="text-rose-400 font-medium">
            {clumpIndices.size} clumped positions
          </span>
        ) : (
          <span className="text-emerald-400 font-medium">Zero artist clumps</span>
        )}
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="w-full h-14 bg-slate-950/60 rounded-lg border border-white/5"
      >
        <polyline
          fill="none"
          stroke={accentColor}
          strokeWidth="2"
          points={points}
        />
        {tracks.map((t, idx) => {
          const x =
            pad + (idx / Math.max(1, tracks.length - 1)) * (w - pad * 2);
          const y = h - pad - t.audio.energy * (h - pad * 2);
          const isClump = clumpIndices.has(idx);
          return (
            <circle
              key={`${t.item_idx}-${idx}`}
              cx={x}
              cy={y}
              r={isClump ? 4 : 2.2}
              fill={isClump ? "#F43F5E" : accentColor}
            />
          );
        })}
      </svg>
    </div>
  );
}
