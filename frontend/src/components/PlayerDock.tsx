"use client";

import React from "react";
import {
  CheckCircle2,
  Heart,
  Repeat,
  SkipForward,
  ThumbsDown,
} from "lucide-react";

export interface ActivePlayerTrack {
  item_idx: number;
  spotify_track_id: string;
  track_name: string;
  artist_name: string;
  genre: string;
}

export type FeedbackAction =
  | "like"
  | "dislike"
  | "early_skip"
  | "complete_play"
  | "repeat";

export function PlayerDock({
  activeTrack,
  onFeedback,
}: {
  activeTrack: ActivePlayerTrack | null;
  onFeedback: (track: ActivePlayerTrack, action: FeedbackAction) => void;
}) {
  if (!activeTrack) return null;

  return (
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
            onClick={() => onFeedback(activeTrack, "like")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 text-xs font-semibold hover:bg-emerald-500/30 transition"
          >
            <Heart className="w-3.5 h-3.5" />
            Like (+1.0)
          </button>
          <button
            onClick={() => onFeedback(activeTrack, "repeat")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-sky-500/20 border border-sky-500/40 text-sky-300 text-xs font-semibold hover:bg-sky-500/30 transition"
          >
            <Repeat className="w-3.5 h-3.5" />
            Replay (+0.85)
          </button>
          <button
            onClick={() => onFeedback(activeTrack, "complete_play")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-800 border border-white/10 text-slate-200 text-xs font-medium hover:bg-slate-700 transition"
          >
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            Full Listen (+0.55)
          </button>
          <button
            onClick={() => onFeedback(activeTrack, "early_skip")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-300 text-xs font-semibold hover:bg-rose-500/30 transition"
          >
            <SkipForward className="w-3.5 h-3.5" />
            Early Skip &lt;15% (-0.75)
          </button>
          <button
            onClick={() => onFeedback(activeTrack, "dislike")}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 border border-rose-500/30 text-rose-400 text-xs font-medium hover:bg-rose-950/60 transition"
          >
            <ThumbsDown className="w-3.5 h-3.5" />
            Dislike
          </button>
        </div>
      </div>
    </footer>
  );
}
