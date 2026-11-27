import React, { useCallback, useEffect, useState } from "react";
import { Activity, BarChart3, Brain, Eye, PauseCircle, RefreshCw } from "lucide-react";
import { companionApi } from "../services/api";
import type { CompanionInsightsResponse } from "../types";

export const InsightsPage: React.FC = () => {
  const [insights, setInsights] = useState<CompanionInsightsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await companionApi.getInsights();
      setInsights(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to compute Music Insights.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadInsights();
  }, [loadInsights]);

  const triggerTemporaryFatigue = async (type: "genre" | "artist", name: string) => {
    try {
      await companionApi.submitFeedback({
        action: type === "genre" ? "genre_fatigue" : "artist_fatigue",
        genre: type === "genre" ? name : undefined,
        artist_name: type === "artist" ? name : undefined,
      });
      setNotice(`Applied temporary ${type} fatigue on "${name}" (decays automatically).`);
      await loadInsights();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not apply temporary fatigue.");
    }
  };

  if (loading && !insights) {
    return (
      <div className="p-8 text-center text-xs text-slate-400">
        Calculating provenance-verified Music Insights...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-3 rounded-xl bg-rose-500/15 border border-rose-500/40 text-xs text-rose-200">
        {error}
      </div>
    );
  }

  if (!insights) return null;

  const { observed, inferred_preferences: inferred, model_predictions: pred } = insights;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xs font-semibold text-white flex items-center gap-1.5">
            <BarChart3 className="w-3.5 h-3.5 text-teal-400" />
            Transparent Music Intelligence
          </h2>
          <p className="text-[11px] text-slate-400">
            Strictly separates Observed signals, Inferred preferences, and Model predictions.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void loadInsights()}
          className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300"
          title="Refresh insights"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
        </button>
      </div>

      {notice && (
        <div className="px-3 py-2 rounded-xl bg-amber-500/12 border border-amber-500/30 text-[11px] text-amber-200">
          {notice}
        </div>
      )}

      {/* 1. OBSERVED SECTION */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-white flex items-center gap-1.5">
            <Eye className="w-3.5 h-3.5 text-emerald-400" />
            Observed Companion Activity
          </span>
          <span className="px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
            Observed
          </span>
        </div>
        <p className="text-[10px] text-slate-400">{observed.description}</p>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Feedback Events</p>
            <p className="text-sm font-bold text-white">{observed.total_feedback_actions}</p>
          </div>
          <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Liked Tracks</p>
            <p className="text-sm font-bold text-emerald-300">{observed.liked_tracks_count}</p>
          </div>
          <div className="p-2 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Dismissed</p>
            <p className="text-sm font-bold text-rose-300">
              {observed.dismissed_or_disliked_count}
            </p>
          </div>
        </div>

        {observed.explicit_preferred_genres.length > 0 && (
          <div>
            <p className="text-[10px] text-slate-400 mb-1">Explicitly Selected Genres:</p>
            <div className="flex flex-wrap gap-1">
              {observed.explicit_preferred_genres.map((g) => (
                <span
                  key={g}
                  className="px-2 py-0.5 rounded-full bg-slate-800 text-[10px] text-slate-200 capitalize"
                >
                  {g}
                </span>
              ))}
            </div>
          </div>
        )}

        {observed.observed_top_artists.length > 0 && (
          <div>
            <p className="text-[10px] text-slate-400 mb-1">Most Positively Rated Artists:</p>
            <div className="flex flex-wrap gap-1.5">
              {observed.observed_top_artists.map((a) => (
                <span
                  key={a.artist}
                  className="px-2 py-0.5 rounded-lg bg-emerald-500/10 border border-emerald-500/25 text-[10px] text-emerald-200"
                >
                  {a.artist} ({a.positive_signals})
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* 2. INFERRED PREFERENCES SECTION */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-white flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            Bandit Adaptation & Fatigue Memory
          </span>
          <span className="px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-amber-500/15 text-amber-300 border border-amber-500/30">
            Inferred Preference
          </span>
        </div>
        <p className="text-[10px] text-slate-400">{inferred.description}</p>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Preference Drift Score</p>
            <p className="text-sm font-bold text-amber-300">
              {(inferred.preference_drift_score * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] text-slate-500">
              Bandit ||Δu|| = {inferred.bandit_residual_magnitude.toFixed(3)}
            </p>
          </div>
          <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Dual-Horizon Balance</p>
            <p className="text-sm font-bold text-teal-300">
              {Math.round(inferred.long_term_horizon_weight * 100)}% Long-Term
            </p>
            <p className="text-[10px] text-slate-500">
              {Math.round(inferred.short_term_session_weight * 100)}% Active Session
            </p>
          </div>
        </div>

        {(inferred.fatigued_artists.length > 0 || inferred.fatigued_genres.length > 0) && (
          <div className="space-y-1">
            <p className="text-[10px] text-slate-400">Active Temporary Fatigue (Auto-Decaying):</p>
            <div className="flex flex-wrap gap-1.5">
              {inferred.fatigued_artists.map((fa) => (
                <span
                  key={fa.artist}
                  className="px-2 py-0.5 rounded-lg bg-rose-500/12 border border-rose-500/30 text-[10px] text-rose-200"
                >
                  Artist: {fa.artist} (-{fa.fatigue_penalty.toFixed(2)})
                </span>
              ))}
              {inferred.fatigued_genres.map((fg) => (
                <span
                  key={fg.genre}
                  className="px-2 py-0.5 rounded-lg bg-amber-500/12 border border-amber-500/30 text-[10px] text-amber-200 capitalize"
                >
                  Genre: {fg.genre} (-{fg.fatigue_penalty.toFixed(2)})
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* 3. MODEL PREDICTIONS SECTION */}
      <section className="p-3.5 rounded-2xl bg-[#0e1524] border border-slate-800/90 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-white flex items-center gap-1.5">
            <Brain className="w-3.5 h-3.5 text-sky-400" />
            Neural Model Predictions
          </span>
          <span className="px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-sky-500/15 text-sky-300 border border-sky-500/30">
            Model Prediction
          </span>
        </div>
        <p className="text-[10px] text-slate-400">{pred.description}</p>

        <div className="grid grid-cols-2 gap-2">
          <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Recommendation Diversity (ILD)</p>
            <p className="text-sm font-bold text-sky-300">
              {(pred.recommendation_diversity_ild * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] text-slate-500">
              {pred.unique_artists_in_top20}/20 unique artists
            </p>
          </div>
          <div className="p-2.5 rounded-xl bg-[#090e18] border border-slate-800">
            <p className="text-[10px] text-slate-400">Repeated Artist Exposure</p>
            <p className="text-sm font-bold text-teal-300">
              {(pred.repeated_artist_exposure_rate * 100).toFixed(1)}%
            </p>
            <p className="text-[10px] text-slate-500">
              Discovery share: {Math.round(pred.familiarity_vs_discovery.actual_discovery_ratio * 100)}%
            </p>
          </div>
        </div>

        {/* Predicted Sonic Profile Bars */}
        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold text-slate-300">Predicted Sonic Profile</p>
          {Object.entries(pred.predicted_sonic_profile).map(([trait, val]) => (
            <div key={trait} className="space-y-0.5">
              <div className="flex justify-between text-[10px] text-slate-400 capitalize">
                <span>{trait}</span>
                <span>{Math.round(val * 100)}%</span>
              </div>
              <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-teal-400 to-sky-400"
                  style={{ width: `${Math.min(100, Math.round(val * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Predicted Top Genres with Temporary Fatigue Controls */}
        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold text-slate-300">
            Predicted Favorite Genres (click pause icon to trigger temporary genre fatigue)
          </p>
          <div className="flex flex-wrap gap-1.5">
            {pred.predicted_favorite_genres.map((g) => (
              <div
                key={g.genre_idx}
                className="px-2.5 py-1 rounded-lg bg-[#090e18] border border-slate-800 text-[10px] text-slate-200 flex items-center gap-1.5"
              >
                <span className="capitalize">{g.genre}</span>
                <button
                  type="button"
                  onClick={() => void triggerTemporaryFatigue("genre", g.genre)}
                  title={`Take a temporary break from ${g.genre}`}
                  className="text-slate-400 hover:text-amber-300"
                >
                  <PauseCircle className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Predicted Top Artists */}
        <div className="space-y-1.5">
          <p className="text-[10px] font-semibold text-slate-300">
            Predicted High-Affinity Artists
          </p>
          <div className="flex flex-wrap gap-1.5">
            {pred.predicted_favorite_artists.map((a) => (
              <div
                key={a.artist}
                className="px-2.5 py-1 rounded-lg bg-[#090e18] border border-slate-800 text-[10px] text-slate-200 flex items-center gap-1.5"
              >
                <span>{a.artist}</span>
                <button
                  type="button"
                  onClick={() => void triggerTemporaryFatigue("artist", a.artist)}
                  title={`Temporarily mute ${a.artist} in recommendations`}
                  className="text-slate-400 hover:text-rose-300"
                >
                  <PauseCircle className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};
