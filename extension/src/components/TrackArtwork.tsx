import React from "react";

/**
 * Generates a deterministic, rich acoustic-inspired SVG gradient artwork tile
 * from legitimate track/artist/genre strings without scraping external pages.
 */
function hashString(input: string): number {
  let h = 2166136261;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}

const PALETTES: Array<[string, string, string]> = [
  ["#0f766e", "#1d4ed8", "#0f172a"],
  ["#047857", "#0369a1", "#111827"],
  ["#4338ca", "#0e7490", "#090d16"],
  ["#be185d", "#4f46e5", "#0f172a"],
  ["#b45309", "#0f766e", "#0b1120"],
  ["#6d28d9", "#0284c7", "#090d16"],
];

export const TrackArtwork: React.FC<{
  title: string;
  artist: string;
  genre?: string;
  size?: "sm" | "md" | "lg";
}> = ({ title, artist, genre, size = "md" }) => {
  const seed = hashString(`${title}::${artist}::${genre || ""}`);
  const [c1, c2, c3] = PALETTES[seed % PALETTES.length];
  const initials = `${title.trim()[0] || "S"}${artist.trim()[0] || "T"}`.toUpperCase();

  const dims =
    size === "sm"
      ? "w-9 h-9 text-[11px] rounded-lg"
      : size === "lg"
        ? "w-14 h-14 text-sm rounded-xl"
        : "w-11 h-11 text-xs rounded-xl";

  return (
    <div
      className={`${dims} shrink-0 relative overflow-hidden flex items-center justify-center font-semibold tracking-wider text-white/90 shadow-inner border border-white/10 select-none`}
      style={{
        background: `radial-gradient(circle at 25% 20%, ${c1}, ${c2} 62%, ${c3})`,
      }}
      aria-hidden="true"
    >
      <div className="absolute inset-0 bg-gradient-to-t from-black/35 via-transparent to-white/10" />
      <span className="relative z-10">{initials}</span>
    </div>
  );
};
