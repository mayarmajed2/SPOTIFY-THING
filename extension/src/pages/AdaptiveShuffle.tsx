import React, { useEffect, useState } from "react";
import {
  ArrowRightLeft,
  Check,
  Copy,
  Download,
  ExternalLink,
  Heart,
  Info,
  ListPlus,
  RefreshCw,
  Save,
  Shuffle,
  ThumbsDown,
} from "lucide-react";
import { companionApi } from "../services/api";
import {
  type AuthorizedPlaylistSummary,
  spotifyProvider,
} from "../services/spotify-api";
import type {
  AdaptiveShuffleResponse,
  ArtistRepetitionPreset,
  CatalogPlaylistSummary,
  DiscoveryModePreset,
  GenreDiversityPreset,
  MusicalCoherencePreset,
  RandomnessPreset,
  ShufflePresets,
  SpotifyConnectionState,
  UserPreferences,
} from "../types";

interface AdaptiveShufflePageProps {
  preferences: UserPreferences;
  spotifyState: SpotifyConnectionState;
  onPreferencesUpdated: (prefs: UserPreferences) => void;
}

function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (next: T) => void;
}) {
  return (
    <div className="space-y-1">
      <span className="block text-[11px] font-medium text-slate-300">{label}</span>
      <div className="grid grid-cols-3 gap-1 p-1 rounded-xl bg-[#090e18] border border-slate-800/90">
        {options.map((opt) => {
          const active = value === opt.value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              className={`py-1 px-2 rounded-lg text-[10px] font-medium transition ${
                active
                  ? "bg-teal-500/20 text-teal-200 border border-teal-400/50 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export const AdaptiveShufflePage: React.FC<AdaptiveShufflePageProps> = ({
  preferences,
  spotifyState,
  onPreferencesUpdated,
}) => {
  const [sourceType, setSourceType] = useState<
    "catalog_playlist" | "discovery_pool" | "authorized_tracks"
  >("catalog_playlist");
  const [catalogPlaylists, setCatalogPlaylists] = useState<CatalogPlaylistSummary[]>([]);
  const [selectedPid, setSelectedPid] = useState<number | null>(null);

  const [spotifyPlaylists, setSpotifyPlaylists] = useState<AuthorizedPlaylistSummary[]>([]);
  const [selectedSpotifyPlaylistId, setSelectedSpotifyPlaylistId] = useState<string>("");

  const [presets, setPresets] = useState<ShufflePresets>(preferences.shuffle_presets);
  const [shuffleResult, setShuffleResult] = useState<AdaptiveShuffleResponse | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<"generated" | "original" | "distributions">(
    "generated",
  );

  const [loading, setLoading] = useState<boolean>(false);
  const [statusNotice, setStatusNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const res = await companionApi.getCatalogPlaylists(24);
        setCatalogPlaylists(res.playlists);
        if (res.playlists.length > 0) {
          const initialPid = res.playlists[0].pid;
          setSelectedPid(initialPid);
          setLoading(true);
          const initialSeq = await companionApi.generateShuffle({
            source_type: "catalog_playlist",
            playlist_pid: initialPid,
            pool_size: 28,
            presets: preferences.shuffle_presets,
            include_baselines: true,
          });
          setShuffleResult(initialSeq);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not initialize Shuffle Studio.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (spotifyState.connected && spotifyState.capabilities.can_read_playlists) {
      void spotifyProvider.getAuthorizedPlaylistData().then((res) => {
        if (res.playlists && res.playlists.length > 0) {
          setSpotifyPlaylists(res.playlists);
          setSelectedSpotifyPlaylistId(res.playlists[0].id);
        }
      });
    }
  }, [spotifyState.connected, spotifyState.capabilities.can_read_playlists]);

  const runShuffle = async (customPresets?: ShufflePresets) => {
    setLoading(true);
    setError(null);
    const activeCfg = customPresets || presets;

    try {
      if (sourceType === "authorized_tracks") {
        if (!selectedSpotifyPlaylistId) {
          throw new Error("Select an authorized Spotify playlist first, or use Catalog Playlists.");
        }
        const plData = await spotifyProvider.getAuthorizedPlaylistData(selectedSpotifyPlaylistId);
        if (!plData.tracks || plData.tracks.length === 0) {
          throw new Error(plData.warning || "Selected Spotify playlist returned no accessible tracks.");
        }
        const chosenPl = spotifyPlaylists.find((p) => p.id === selectedSpotifyPlaylistId);
        const res = await companionApi.generateShuffle({
          source_type: "authorized_tracks",
          source_name: chosenPl?.name || "Authorized Spotify Playlist",
          pool_size: Math.min(50, plData.tracks.length),
          tracks: plData.tracks,
          presets: activeCfg,
          include_baselines: true,
        });
        setShuffleResult(res);
      } else {
        const res = await companionApi.generateShuffle({
          source_type: sourceType,
          playlist_pid: sourceType === "catalog_playlist" ? selectedPid : null,
          pool_size: 28,
          presets: activeCfg,
          include_baselines: true,
        });
        setShuffleResult(res);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Shuffle sequence generation failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleSavePresets = async () => {
    try {
      const saved = await companionApi.saveShufflePresets(presets);
      onPreferencesUpdated({
        ...preferences,
        shuffle_presets: saved.shuffle_presets,
      });
      setStatusNotice("Saved preferred Adaptive Shuffle configuration.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save shuffle configuration.");
    }
  };

  const handleCopyText = async (content: string, label: string) => {
    await navigator.clipboard?.writeText(content);
    setStatusNotice(`Copied ${label} to clipboard.`);
  };

  const handleDownloadFile = (content: string, filename: string, mime: string) => {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    setStatusNotice(`Exported ${filename}`);
  };

  const handleExportToSpotifyPlaylist = async () => {
    if (!shuffleResult) return;
    setStatusNotice("Checking Spotify playlist permission...");
    const res = await spotifyProvider.createPlaylistIfPermitted({
      name: `${shuffleResult.source_name} · Adaptive Sequence`,
      description: "Sequenced by SPOTIFY-THING Adaptive Shuffle Studio",
      trackUris: shuffleResult.handoff.spotify_uris,
    });
    setStatusNotice(res.message);
    if (res.created && res.playlistUrl) {
      await spotifyProvider.executeSupportedActions({
        type: "open_url_in_spotify",
        url: res.playlistUrl,
      });
    }
  };

  const handleTrackFeedback = async (
    itemIdx: number | null,
    spotifyTrackId: string,
    action: "like" | "dislike",
    trackName: string,
  ) => {
    try {
      await companionApi.submitFeedback({
        item_idx: itemIdx,
        spotify_track_id: spotifyTrackId,
        action,
      });
      setStatusNotice(
        action === "like"
          ? `Liked "${trackName}" — future shuffles will favor its vibe.`
          : `Disliked "${trackName}" — increased artist cooldown & fatigue.`,
      );
    } catch {
      setStatusNotice("Feedback available for catalog-matched tracks.");
    }
  };

  return (
    <div className="space-y-4">
      {/* Source Selection + Configuration Card */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold text-white flex items-center gap-1.5">
            <Shuffle className="w-3.5 h-3.5 text-teal-400" />
            Adaptive Shuffle Studio
          </h2>
          <button
            type="button"
            onClick={() => void handleSavePresets()}
            className="px-2.5 py-1 rounded-lg text-[11px] font-medium bg-teal-500/15 text-teal-200 border border-teal-500/40 hover:bg-teal-500/25 flex items-center gap-1"
          >
            <Save className="w-3 h-3" /> Save Config
          </button>
        </div>

        {/* Source Selector */}
        <div className="grid grid-cols-3 gap-1 p-1 rounded-xl bg-[#090e18] border border-slate-800">
          <button
            type="button"
            onClick={() => setSourceType("catalog_playlist")}
            className={`py-1.5 px-2 rounded-lg text-[10px] font-medium transition ${
              sourceType === "catalog_playlist"
                ? "bg-slate-800 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Curated Playlists
          </button>
          <button
            type="button"
            onClick={() => setSourceType("discovery_pool")}
            className={`py-1.5 px-2 rounded-lg text-[10px] font-medium transition ${
              sourceType === "discovery_pool"
                ? "bg-slate-800 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            AI Discovery Mix
          </button>
          <button
            type="button"
            onClick={() => setSourceType("authorized_tracks")}
            className={`py-1.5 px-2 rounded-lg text-[10px] font-medium transition ${
              sourceType === "authorized_tracks"
                ? "bg-slate-800 text-white"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Spotify Playlists
          </button>
        </div>

        {sourceType === "catalog_playlist" && (
          <select
            aria-label="Select Curated Playlist"
            value={selectedPid ?? ""}
            onChange={(e) => setSelectedPid(Number(e.target.value))}
            className="w-full px-3 py-2 rounded-xl bg-[#090e18] border border-slate-700/80 text-xs text-slate-100 focus:outline-none focus:border-teal-400"
          >
            {catalogPlaylists.map((pl) => (
              <option key={pl.pid} value={pl.pid}>
                {pl.name} ({pl.num_tracks} tracks · {Math.round(pl.artist_concentration * 100)}% artist repeat)
              </option>
            ))}
          </select>
        )}

        {sourceType === "authorized_tracks" && (
          <div>
            {spotifyPlaylists.length > 0 ? (
              <select
                aria-label="Select Authorized Spotify Playlist"
                value={selectedSpotifyPlaylistId}
                onChange={(e) => setSelectedSpotifyPlaylistId(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-[#090e18] border border-slate-700/80 text-xs text-slate-100"
              >
                {spotifyPlaylists.map((pl) => (
                  <option key={pl.id} value={pl.id}>
                    {pl.name} ({pl.trackTotal} tracks)
                  </option>
                ))}
              </select>
            ) : (
              <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800 text-[11px] text-slate-400">
                Spotify OAuth playlist access is not connected. Connect in <strong>Settings</strong>{" "}
                with <code>playlist-read-private</code>, or sequence <strong>Curated Playlists</strong>{" "}
                / <strong>AI Discovery Mix</strong> in Independent Mode.
              </div>
            )}
          </div>
        )}

        {/* 5 Core Shuffle Configuration Axes */}
        <div className="space-y-2.5 pt-1">
          <SegmentedControl<ArtistRepetitionPreset>
            label="Artist Repetition"
            value={presets.artist_repetition}
            options={[
              { value: "strict", label: "Strict spacing" },
              { value: "balanced", label: "Balanced" },
              { value: "flexible", label: "Flexible" },
            ]}
            onChange={(v) => setPresets((p) => ({ ...p, artist_repetition: v }))}
          />
          <SegmentedControl<GenreDiversityPreset>
            label="Genre Diversity"
            value={presets.genre_diversity}
            options={[
              { value: "low", label: "Low" },
              { value: "medium", label: "Medium" },
              { value: "high", label: "High" },
            ]}
            onChange={(v) => setPresets((p) => ({ ...p, genre_diversity: v }))}
          />
          <SegmentedControl<MusicalCoherencePreset>
            label="Musical Coherence"
            value={presets.musical_coherence}
            options={[
              { value: "smooth", label: "Smooth flow" },
              { value: "balanced", label: "Balanced" },
              { value: "variety", label: "More variety" },
            ]}
            onChange={(v) => setPresets((p) => ({ ...p, musical_coherence: v }))}
          />
          <SegmentedControl<DiscoveryModePreset>
            label="Discovery Balance"
            value={presets.discovery_mode}
            options={[
              { value: "familiar", label: "Familiar" },
              { value: "balanced", label: "Balanced" },
              { value: "exploratory", label: "Exploratory" },
            ]}
            onChange={(v) => setPresets((p) => ({ ...p, discovery_mode: v }))}
          />
          <SegmentedControl<RandomnessPreset>
            label="Randomness"
            value={presets.randomness}
            options={[
              { value: "predictable", label: "Predictable" },
              { value: "balanced", label: "Balanced" },
              { value: "varied", label: "Highly varied" },
            ]}
            onChange={(v) => setPresets((p) => ({ ...p, randomness: v }))}
          />
        </div>

        <button
          type="button"
          disabled={loading}
          onClick={() => void runShuffle()}
          className="w-full py-2.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-teal-500 to-emerald-500 text-slate-950 hover:opacity-95 disabled:opacity-50 transition flex items-center justify-center gap-1.5"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          {loading ? "Optimizing Listening Sequence..." : "Generate / Regenerate Adaptive Sequence"}
        </button>
      </section>

      {statusNotice && (
        <div
          role="status"
          className="px-3 py-2 rounded-xl bg-teal-500/12 border border-teal-500/35 text-[11px] text-teal-200 flex items-center justify-between"
        >
          <span className=" flex items-center gap-1.5">
            <Check className="w-3.5 h-3.5 shrink-0" /> {statusNotice}
          </span>
          <button type="button" onClick={() => setStatusNotice(null)} className="ml-2">
            ✕
          </button>
        </div>
      )}

      {error && (
        <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200">
          {error}
        </div>
      )}

      {shuffleResult && (
        <>
          {/* Sequence Comparison Metrics Card */}
          <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-white">
                Sequence Diagnostics: {shuffleResult.source_name}
              </h3>
              <span className="text-[10px] text-slate-400">
                {shuffleResult.pool_size} tracks · Seed #{shuffleResult.seed}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800/80">
                <p className="text-[10px] text-slate-400">Artist Clump Rate</p>
                <p className="text-sm font-bold text-teal-300">
                  {(shuffleResult.metrics.artist_clump_rate * 100).toFixed(1)}%
                </p>
                {shuffleResult.baselines.original_order && (
                  <p className="text-[10px] text-slate-500">
                    Orig {(shuffleResult.baselines.original_order.artist_clump_rate * 100).toFixed(1)}% ·
                    Rand {((shuffleResult.baselines.fisher_yates?.artist_clump_rate ?? 0) * 100).toFixed(1)}%
                  </p>
                )}
              </div>

              <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800/80">
                <p className="text-[10px] text-slate-400">Mean Artist Gap</p>
                <p className="text-sm font-bold text-sky-300">
                  {shuffleResult.metrics.mean_artist_gap.toFixed(1)} steps
                </p>
                <p className="text-[10px] text-slate-500">
                  Min gap {shuffleResult.metrics.min_artist_gap.toFixed(0)}
                </p>
              </div>

              <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800/80">
                <p className="text-[10px] text-slate-400">Acoustic Flow</p>
                <p className="text-sm font-bold text-emerald-300">
                  {(shuffleResult.metrics.acoustic_smoothness * 100).toFixed(1)}%
                </p>
                <p className="text-[10px] text-slate-500">
                  ±{shuffleResult.metrics.mean_bpm_jump.toFixed(1)} BPM
                </p>
              </div>
            </div>

            <p className="text-[10px] text-slate-400 flex items-start gap-1.5">
              <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
              <span>{shuffleResult.handoff.limitation_notice}</span>
            </p>

            {/* Export & Handoff Toolbar */}
            <div className="pt-1 flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() =>
                  void handleCopyText(
                    shuffleResult.handoff.spotify_uris.join("\n"),
                    `${shuffleResult.handoff.spotify_uris.length} Spotify URIs`,
                  )
                }
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] font-medium text-slate-200 flex items-center gap-1"
              >
                <Copy className="w-3 h-3" /> Copy Spotify URIs
              </button>
              <button
                type="button"
                onClick={() =>
                  handleDownloadFile(
                    shuffleResult.handoff.m3u,
                    "adaptive-shuffle.m3u",
                    "audio/x-mpegurl",
                  )
                }
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] font-medium text-slate-200 flex items-center gap-1"
              >
                <Download className="w-3 h-3" /> Export M3U
              </button>
              <button
                type="button"
                onClick={() =>
                  handleDownloadFile(shuffleResult.handoff.csv, "adaptive-shuffle.csv", "text/csv")
                }
                className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] font-medium text-slate-200 flex items-center gap-1"
              >
                <Download className="w-3 h-3" /> Export CSV
              </button>
              {spotifyState.connected && (
                <button
                  type="button"
                  onClick={() => void handleExportToSpotifyPlaylist()}
                  className="px-2.5 py-1.5 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/35 text-[10px] font-medium text-emerald-200 flex items-center gap-1"
                >
                  <ListPlus className="w-3 h-3" /> Save Private Spotify Playlist
                </button>
              )}
            </div>
          </section>

          {/* View Switcher: Generated Order vs. Original Order vs. Distributions */}
          <div className="flex gap-1 p-1 rounded-xl bg-[#0e1524] border border-slate-800">
            <button
              type="button"
              onClick={() => setActiveSubTab("generated")}
              className={`flex-1 py-1.5 rounded-lg text-[11px] font-medium transition ${
                activeSubTab === "generated"
                  ? "bg-teal-500/20 text-teal-200"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Generated Sequence
            </button>
            <button
              type="button"
              onClick={() => setActiveSubTab("original")}
              className={`flex-1 py-1.5 rounded-lg text-[11px] font-medium transition flex items-center justify-center gap-1 ${
                activeSubTab === "original"
                  ? "bg-slate-800 text-white"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <ArrowRightLeft className="w-3 h-3" /> Original Order
            </button>
            <button
              type="button"
              onClick={() => setActiveSubTab("distributions")}
              className={`flex-1 py-1.5 rounded-lg text-[11px] font-medium transition ${
                activeSubTab === "distributions"
                  ? "bg-sky-500/20 text-sky-200"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Distributions
            </button>
          </div>

          {(activeSubTab === "generated" || activeSubTab === "original") && (
            <div className="space-y-2">
              {(activeSubTab === "generated"
                ? shuffleResult.generated_tracks
                : shuffleResult.original_tracks
              ).map((t) => (
                <div
                  key={`${activeSubTab}-${t.position}`}
                  className="p-2.5 rounded-xl bg-[#0e1524] border border-slate-800/80 flex items-center justify-between gap-2.5"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-6 text-center text-[11px] font-mono text-teal-300 shrink-0">
                      #{t.position}
                    </span>
                    <div className="min-w-0">
                      <p className="text-xs font-medium text-white truncate">{t.track_name}</p>
                      <p className="text-[11px] text-slate-400 truncate">
                        {t.artist_name} · <span className="capitalize">{t.genre}</span>
                        {activeSubTab === "generated" && (
                          <span className="text-slate-500"> (was #{t.original_position})</span>
                        )}
                      </p>
                      {activeSubTab === "generated" && t.transition?.reason && (
                        <p className="text-[10px] text-teal-300/85 truncate">
                          {t.transition.reason}
                          {t.audio ? ` · ${Math.round(t.audio.tempo)} BPM` : ""}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    {t.item_idx !== null && (
                      <>
                        <button
                          type="button"
                          onClick={() =>
                            void handleTrackFeedback(
                              t.item_idx,
                              t.spotify_track_id,
                              "like",
                              t.track_name,
                            )
                          }
                          className="p-1 rounded hover:bg-teal-500/20 text-slate-400 hover:text-teal-300"
                          title="Like track"
                        >
                          <Heart className="w-3 h-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            void handleTrackFeedback(
                              t.item_idx,
                              t.spotify_track_id,
                              "dislike",
                              t.track_name,
                            )
                          }
                          className="p-1 rounded hover:bg-rose-500/20 text-slate-400 hover:text-rose-300"
                          title="Dislike track"
                        >
                          <ThumbsDown className="w-3 h-3" />
                        </button>
                      </>
                    )}
                    {t.spotify_url && (
                      <button
                        type="button"
                        onClick={() =>
                          void spotifyProvider.executeSupportedActions({
                            type: "open_url_in_spotify",
                            url: t.spotify_url!,
                          })
                        }
                        className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-200 flex items-center gap-0.5"
                      >
                        Play <ExternalLink className="w-2.5 h-2.5" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeSubTab === "distributions" && (
            <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800 space-y-3">
              <div className="text-[11px] text-slate-300">
                <strong>Feature Coverage:</strong> {shuffleResult.feature_coverage.note}
              </div>

              <div>
                <h4 className="text-[11px] font-semibold text-slate-300 mb-1.5">
                  Artist Distribution ({shuffleResult.distributions.repetition.unique_artists} unique)
                </h4>
                <div className="space-y-1.5">
                  {shuffleResult.distributions.artists.slice(0, 6).map((a) => (
                    <div key={a.artist} className="space-y-0.5">
                      <div className="flex justify-between text-[10px] text-slate-300">
                        <span className="truncate">{a.artist}</span>
                        <span>
                          {a.count} tracks ({Math.round(a.share * 100)}%)
                        </span>
                      </div>
                      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                        <div
                          className="h-full bg-teal-400 rounded-full"
                          style={{ width: `${Math.min(100, Math.round(a.share * 220))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-[11px] font-semibold text-slate-300 mb-1.5">
                  Genre Distribution & Discovery Balance
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {shuffleResult.distributions.genres.slice(0, 8).map((g) => (
                    <span
                      key={g.genre}
                      className="px-2 py-1 rounded-lg bg-[#090e18] border border-slate-800 text-[10px] text-slate-300 capitalize"
                    >
                      {g.genre}: {g.count}
                    </span>
                  ))}
                </div>
                <p className="text-[11px] text-slate-400 mt-2">
                  Familiar tracks: {shuffleResult.distributions.discovery.familiar_count} ·
                  Discovery tracks: {shuffleResult.distributions.discovery.discovery_count} (
                  {Math.round(shuffleResult.distributions.discovery.discovery_ratio * 100)}%)
                </p>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
};
