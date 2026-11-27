import React, { useCallback, useEffect, useState } from "react";
import {
  ExternalLink,
  Flame,
  GitBranch,
  Heart,
  RefreshCw,
  Save,
  Sparkles,
  ThumbsDown,
  TrendingDown,
  Users,
  XCircle,
} from "lucide-react";
import { TrackArtwork } from "../components/TrackArtwork";
import { companionApi } from "../services/api";
import { spotifyProvider } from "../services/spotify-api";
import type {
  DiscoveryFeedResponse,
  FeedbackAction,
  RecommendedTrack,
  SimilarArtistItem,
  SimilarTrackItem,
  UserPreferences,
} from "../types";

interface DiscoveryPageProps {
  preferences: UserPreferences;
  onPreferencesUpdated: (prefs: UserPreferences) => void;
}

export const DiscoveryPage: React.FC<DiscoveryPageProps> = ({
  preferences,
  onPreferencesUpdated,
}) => {
  const [discoveryIntensity, setDiscoveryIntensity] = useState<number>(
    preferences.discovery_intensity ?? 0.4,
  );
  const [selectedGenre, setSelectedGenre] = useState<string | null>(null);
  const [dismissedLocal, setDismissedLocal] = useState<number[]>([]);
  const [feed, setFeed] = useState<DiscoveryFeedResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [savingPrefs, setSavingPrefs] = useState<boolean>(false);
  const [feedbackToast, setFeedbackToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [similarDrawer, setSimilarDrawer] = useState<{
    mode: "tracks" | "artists";
    title: string;
    tracks?: SimilarTrackItem[];
    artists?: SimilarArtistItem[];
  } | null>(null);

  const loadDiscovery = useCallback(
    async (opts?: { intensity?: number; excludeExtra?: number[]; genre?: string | null }) => {
      setLoading(true);
      setError(null);
      try {
        const res = await companionApi.getDiscoveryFeed({
          top_k: 16,
          discovery_intensity: opts?.intensity ?? discoveryIntensity,
          diversity_level: preferences.diversity_level ?? 0.5,
          novelty_boost: preferences.novelty_boost ?? 0.3,
          exclude_item_indices: opts?.excludeExtra ?? dismissedLocal,
          genre_filter: opts?.genre !== undefined ? opts.genre : selectedGenre,
        });
        setFeed(res);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load recommendations.");
      } finally {
        setLoading(false);
      }
    },
    [discoveryIntensity, dismissedLocal, preferences.diversity_level, preferences.novelty_boost, selectedGenre],
  );

  useEffect(() => {
    void loadDiscovery();
  }, []);

  const handleSaveIntensity = async () => {
    setSavingPrefs(true);
    try {
      const updated = await companionApi.updatePreferences({
        discovery_intensity: discoveryIntensity,
      });
      onPreferencesUpdated(updated.preferences);
      setFeedbackToast(`Saved discovery balance (${Math.round(discoveryIntensity * 100)}% new music)`);
      await loadDiscovery({ intensity: discoveryIntensity });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not save discovery settings.");
    } finally {
      setSavingPrefs(false);
    }
  };

  const handleFeedback = async (track: RecommendedTrack, action: FeedbackAction) => {
    try {
      const res = await companionApi.submitFeedback({
        item_idx: track.item_idx,
        action,
      });
      const labelMap: Record<string, string> = {
        like: `Liked "${track.track_name}" — boosted similar taste`,
        dislike: `Disliked "${track.track_name}" — suppressed artist & style`,
        not_interested: `Replaced "${track.track_name}" with fresh alternatives`,
        more_like_this: `Tuned session toward "${track.track_name}" (${track.genre})`,
        less_like_this: `Dialed back "${track.artist_name}" for this session`,
      };
      setFeedbackToast(labelMap[action] || `Recorded ${res.event.action}`);

      let nextExcluded = dismissedLocal;
      if (action === "not_interested" || action === "dislike" || action === "less_like_this") {
        nextExcluded = [...dismissedLocal, track.item_idx];
        setDismissedLocal(nextExcluded);
      }
      await loadDiscovery({ excludeExtra: nextExcluded });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feedback update failed.");
    }
  };

  const handleExploreSimilarTracks = async (track: RecommendedTrack) => {
    try {
      const res = await companionApi.getSimilarTracks(track.item_idx, 6);
      setSimilarDrawer({
        mode: "tracks",
        title: `Similar to "${res.seed_track.track_name}"`,
        tracks: res.similar_tracks,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load similar tracks.");
    }
  };

  const handleExploreSimilarArtists = async (artistName: string) => {
    try {
      const res = await companionApi.getSimilarArtists(artistName, 6);
      setSimilarDrawer({
        mode: "artists",
        title: `Artists close to ${res.seed_artist.artist_name}`,
        artists: res.similar_artists,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load similar artists.");
    }
  };

  return (
    <div className="space-y-4">
      {/* Familiarity vs. Discovery Controller */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-xs font-semibold text-white flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-teal-400" />
              Familiarity vs. Discovery Controller
            </h2>
            {feed && (
              <p className="text-[11px] text-slate-400">
                {feed.familiarity_vs_discovery.familiar_count} familiar ·{" "}
                {feed.familiarity_vs_discovery.discovery_count} discoveries · Gate{" "}
                {Math.round(feed.familiarity_vs_discovery.long_term_gate * 100)}% NeuMF /{" "}
                {Math.round(feed.familiarity_vs_discovery.short_term_gate * 100)}% Session
              </p>
            )}
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={handleSaveIntensity}
              disabled={savingPrefs}
              className="px-2.5 py-1.5 rounded-lg text-[11px] font-medium bg-teal-500/15 text-teal-200 border border-teal-500/40 hover:bg-teal-500/25 transition flex items-center gap-1"
            >
              <Save className="w-3 h-3" />
              Save
            </button>
            <button
              type="button"
              onClick={() => void loadDiscovery()}
              disabled={loading}
              title="Request fresh alternatives"
              className="px-2.5 py-1.5 rounded-lg text-[11px] font-medium bg-slate-800 text-slate-200 hover:bg-slate-700 transition flex items-center gap-1"
            >
              <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>
        </div>

        <div className="space-y-1">
          <input
            aria-label="Discovery Intensity"
            type="range"
            min={0.05}
            max={0.95}
            step={0.05}
            value={discoveryIntensity}
            onChange={(e) => {
              const v = Number(e.target.value);
              setDiscoveryIntensity(v);
            }}
            onMouseUp={() => void loadDiscovery({ intensity: discoveryIntensity })}
            onTouchEnd={() => void loadDiscovery({ intensity: discoveryIntensity })}
            className="w-full"
          />
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Comfort & Favorites</span>
            <span className="text-teal-300 font-medium">
              Target: {Math.round(discoveryIntensity * 100)}% Discovery
            </span>
            <span>Deep Novelty</span>
          </div>
        </div>

        {/* Related Genres Filter Pills */}
        {feed && feed.related_genres.length > 0 && (
          <div className="flex items-center gap-1.5 overflow-x-auto pb-1 pt-0.5">
            <button
              type="button"
              onClick={() => {
                setSelectedGenre(null);
                void loadDiscovery({ genre: null });
              }}
              className={`px-2.5 py-1 rounded-full text-[11px] shrink-0 border transition ${
                selectedGenre === null
                  ? "bg-teal-500/20 text-teal-200 border-teal-400/60"
                  : "bg-[#131b2b] text-slate-300 border-slate-800"
              }`}
            >
              All Genres
            </button>
            {feed.related_genres.slice(0, 8).map((g) => (
              <button
                key={g.genre_idx}
                type="button"
                onClick={() => {
                  const next = selectedGenre === g.genre ? null : g.genre;
                  setSelectedGenre(next);
                  void loadDiscovery({ genre: next });
                }}
                className={`px-2.5 py-1 rounded-full text-[11px] capitalize shrink-0 border transition ${
                  selectedGenre === g.genre
                    ? "bg-sky-500/20 text-sky-200 border-sky-400/60"
                    : "bg-[#131b2b] text-slate-300 border-slate-800 hover:border-slate-700"
                }`}
              >
                {g.genre} {g.is_exploratory ? "✦" : ""}
              </button>
            ))}
          </div>
        )}
      </section>

      {feedbackToast && (
        <div
          role="status"
          className="px-3 py-2 rounded-xl bg-teal-500/12 border border-teal-500/35 text-[11px] text-teal-200 flex items-center justify-between"
        >
          <span>{feedbackToast}</span>
          <button
            type="button"
            onClick={() => setFeedbackToast(null)}
            className="text-teal-300/70 hover:text-teal-100 ml-2"
          >
            ✕
          </button>
        </div>
      )}

      {error && (
        <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200">
          {error}
        </div>
      )}

      {/* Similar Tracks / Similar Artists Interactive Drawer */}
      {similarDrawer && (
        <section className="p-3.5 rounded-2xl bg-[#111a2e] border border-sky-500/40 space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-sky-200">{similarDrawer.title}</h3>
            <button
              type="button"
              onClick={() => setSimilarDrawer(null)}
              className="text-[11px] text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>

          {similarDrawer.mode === "tracks" && similarDrawer.tracks && (
            <div className="space-y-2">
              {similarDrawer.tracks.map((st) => (
                <div
                  key={st.item_idx}
                  className="flex items-center justify-between gap-2 p-2 rounded-xl bg-[#0b111e] border border-slate-800/80"
                >
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-white truncate">{st.track_name}</p>
                    <p className="text-[11px] text-slate-400 truncate">
                      {st.artist_name} · <span className="capitalize">{st.genre}</span>
                    </p>
                    <p className="text-[10px] text-sky-300/90">{st.reason}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      void spotifyProvider.executeSupportedActions({
                        type: "open_url_in_spotify",
                        url: st.spotify_url,
                      })
                    }
                    className="px-2 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-[10px] text-slate-200 shrink-0 flex items-center gap-1"
                  >
                    Spotify <ExternalLink className="w-2.5 h-2.5" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {similarDrawer.mode === "artists" && similarDrawer.artists && (
            <div className="grid grid-cols-2 gap-2">
              {similarDrawer.artists.map((sa) => (
                <div
                  key={sa.artist_idx}
                  className="p-2.5 rounded-xl bg-[#0b111e] border border-slate-800/80 space-y-1"
                >
                  <p className="text-xs font-semibold text-white truncate">{sa.artist_name}</p>
                  <p className="text-[10px] text-slate-400 capitalize">
                    {sa.genre} · {Math.round(sa.similarity * 100)}% sonic match
                  </p>
                  <button
                    type="button"
                    onClick={() =>
                      void spotifyProvider.executeSupportedActions({
                        type: "open_url_in_spotify",
                        url: sa.sample_track.spotify_url,
                      })
                    }
                    className="text-[10px] text-teal-300 hover:underline flex items-center gap-1 truncate"
                  >
                    Play "{sa.sample_track.track_name}" <ExternalLink className="w-2.5 h-2.5 shrink-0" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Recommended Tracks */}
      <section className="space-y-2.5">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Personalized Recommendations
          </h3>
          {feed && (
            <span className="text-[11px] text-slate-400">
              {feed.recommended_tracks.length} tracks ranked
            </span>
          )}
        </div>

        {loading && !feed ? (
          <div className="p-8 text-center text-xs text-slate-400">
            Running Multimodal NeuMF + Dual-Horizon Transformer...
          </div>
        ) : (
          <div className="space-y-2.5">
            {feed?.recommended_tracks.map((track) => (
              <article
                key={track.item_idx}
                className="p-3 rounded-2xl bg-[#0e1524] border border-slate-800/90 hover:border-slate-700/90 transition space-y-2"
              >
                <div className="flex items-start gap-3">
                  <TrackArtwork
                    title={track.track_name}
                    artist={track.artist_name}
                    genre={track.genre}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <h4 className="text-xs font-semibold text-white truncate">
                          {track.track_name}
                        </h4>
                        <p className="text-[11px] text-slate-300 truncate">
                          {track.artist_name} ·{" "}
                          <span className="text-slate-400">{track.album_name}</span>
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() =>
                          void spotifyProvider.executeSupportedActions({
                            type: "open_url_in_spotify",
                            url: track.spotify_url,
                          })
                        }
                        className="px-2 py-1 rounded-lg bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/30 text-[10px] font-medium shrink-0 flex items-center gap-1"
                        title="Open track in Spotify"
                      >
                        Listen <ExternalLink className="w-2.5 h-2.5" />
                      </button>
                    </div>

                    <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                      <span
                        className={`px-2 py-0.5 rounded-full text-[10px] font-medium ${
                          track.is_discovery
                            ? "bg-sky-500/15 text-sky-300 border border-sky-500/30"
                            : "bg-slate-800 text-slate-300"
                        }`}
                      >
                        {track.is_discovery ? "Discovery" : "Familiar"}
                      </span>
                      <span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-800/90 text-slate-300 capitalize">
                        {track.genre}
                      </span>
                      <span className="text-[10px] text-teal-300/90 font-mono">
                        Score {track.score.toFixed(2)} · Nov {Math.round(track.novelty * 100)}%
                      </span>
                    </div>

                    {track.explanations.length > 0 && (
                      <p className="text-[11px] text-slate-400 mt-1.5 leading-snug">
                        {track.explanations.join(" · ")}
                      </p>
                    )}
                  </div>
                </div>

                {/* Rich Personalization & Exploration Controls */}
                <div className="pt-2 border-t border-slate-800/70 flex flex-wrap items-center justify-between gap-1">
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => void handleFeedback(track, "like")}
                      className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 hover:bg-teal-500/20 hover:text-teal-200 text-slate-300 transition flex items-center gap-1"
                      title="Like track"
                    >
                      <Heart className="w-3 h-3" /> Like
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleFeedback(track, "dislike")}
                      className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 hover:bg-rose-500/20 hover:text-rose-200 text-slate-300 transition flex items-center gap-1"
                      title="Dislike track"
                    >
                      <ThumbsDown className="w-3 h-3" /> Dislike
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleFeedback(track, "more_like_this")}
                      className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 hover:bg-sky-500/20 hover:text-sky-200 text-slate-300 transition flex items-center gap-1"
                      title="More like this"
                    >
                      <Flame className="w-3 h-3" /> More
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleFeedback(track, "less_like_this")}
                      className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 hover:bg-amber-500/20 hover:text-amber-200 text-slate-300 transition flex items-center gap-1"
                      title="Less like this"
                    >
                      <TrendingDown className="w-3 h-3" /> Less
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleFeedback(track, "not_interested")}
                      className="px-2 py-1 rounded-lg text-[10px] font-medium bg-slate-800/80 hover:bg-slate-700 text-slate-400 transition flex items-center gap-1"
                      title="Dismiss and request alternatives"
                    >
                      <XCircle className="w-3 h-3" /> Alt
                    </button>
                  </div>

                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => void handleExploreSimilarTracks(track)}
                      className="px-2 py-1 rounded-lg text-[10px] text-sky-300 bg-sky-500/10 hover:bg-sky-500/20 transition flex items-center gap-1"
                    >
                      <GitBranch className="w-3 h-3" /> Similar Tracks
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleExploreSimilarArtists(track.artist_name)}
                      className="px-2 py-1 rounded-lg text-[10px] text-purple-300 bg-purple-500/10 hover:bg-purple-500/20 transition flex items-center gap-1"
                    >
                      <Users className="w-3 h-3" /> Artists
                    </button>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      {/* Recommended Artists Section */}
      {feed && feed.recommended_artists.length > 0 && (
        <section className="space-y-2 pt-1">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Recommended Artists For You
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {feed.recommended_artists.slice(0, 6).map((art) => (
              <div
                key={art.artist_idx}
                className="p-3 rounded-xl bg-[#0e1524] border border-slate-800/90 space-y-1"
              >
                <div className="flex items-center justify-between gap-1">
                  <p className="text-xs font-semibold text-white truncate">{art.artist_name}</p>
                  {art.is_discovery && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300">
                      New
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-slate-400 capitalize">{art.genre}</p>
                <p className="text-[10px] text-slate-400 line-clamp-2">{art.reason}</p>
                <div className="pt-1 flex items-center justify-between">
                  <button
                    type="button"
                    onClick={() => void handleExploreSimilarArtists(art.artist_name)}
                    className="text-[10px] text-teal-300 hover:underline"
                  >
                    Explore similar
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      void spotifyProvider.executeSupportedActions({
                        type: "open_url_in_spotify",
                        url: art.representative_track.spotify_url,
                      })
                    }
                    className="text-[10px] text-slate-300 hover:text-white flex items-center gap-0.5"
                  >
                    Sample <ExternalLink className="w-2.5 h-2.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
