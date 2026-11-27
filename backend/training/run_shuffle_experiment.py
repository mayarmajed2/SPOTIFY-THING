"""
Controlled Multi-Seed Adaptive Shuffle Optimization Experiment.

Compares 5 algorithms across the exact same 200 real Spotify playlists,
candidate tracks, NeuMF relevance scores, and discovery masks over 5 random seeds:
1. Fisher-Yates Random
2. Artist-Spacing Heuristic
3. Greedy Relevance
4. Original Adaptive Shuffle (quota_urgency_weight = 0.0)
5. Adaptive Shuffle + Remaining-Quota Urgency (experimental: w = 0.40, 0.75, 1.20)

Reports mean +/- std across seeds, overall and stratified by playlist artist concentration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from backend.app.engine.adaptive_shuffle import AdaptiveShuffleEngine, ShuffleConfig
from backend.app.models.collaborative import NeuralCollaborativeFiltering
from backend.app.models.embeddings import MultimodalTrackEncoder

METRIC_KEYS = [
    "artist_clump_rate",
    "min_artist_gap",
    "p25_artist_gap",
    "median_artist_gap",
    "mean_artist_gap",
    "max_artist_gap",
    "session_ndcg",
    "genre_switch_rate",
    "window_genre_diversity",
    "intra_list_diversity",
    "local_window_diversity",
    "acoustic_smoothness",
    "mean_bpm_jump",
    "discovery_pacing_std",
    "sequence_validity",
]


def aggregate_seed_runs(seed_dicts: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in METRIC_KEYS:
        vals = [d[k] for d in seed_dicts]
        out[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=0)), 4),
        }
    return out


def run_experiment(
    processed_dir: Path = Path("data/processed"),
    artifacts_dir: Path = Path("data/artifacts"),
    reports_dir: Path = Path("data/reports"),
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    seeds = seeds or [1000, 2000, 3000, 4000, 5000]

    tracks_df = pd.read_parquet(processed_dir / "tracks.parquet").sort_values("item_idx").reset_index(drop=True)
    playlists_df = pd.read_parquet(processed_dir / "playlists.parquet")
    stats = json.loads((processed_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    item_embeddings = np.load(artifacts_dir / "track_embeddings.npy").astype(np.float32)

    num_items = len(tracks_df)
    num_artists = int(tracks_df["artist_idx"].max()) + 1
    num_genres = int(tracks_df["genre_idx"].max()) + 1
    num_users = int(stats["num_users"])

    enc = MultimodalTrackEncoder(num_items, num_artists, num_genres, 12, 128)
    neumf = NeuralCollaborativeFiltering(num_users, enc, embed_dim=128)
    neumf.load_state_dict(torch.load(artifacts_dir / "neumf_checkpoint.pt", map_location="cpu", weights_only=True))
    neumf.eval()
    with torch.no_grad():
        all_u_long = neumf.get_long_term_user_emb(torch.arange(num_users)).numpy()

    artist_ids = tracks_df["artist_idx"].to_numpy(dtype=np.int32)
    genre_ids = tracks_df["genre_idx"].to_numpy(dtype=np.int32)
    raw_audio = tracks_df[["danceability", "energy", "valence", "tempo", "key"]].to_numpy(dtype=np.float32)

    # Exact same 200 playlists & discovery masks as Suite B
    sample_playlists = playlists_df[playlists_df["num_tracks"] >= 20].head(200)
    rng_mask = np.random.default_rng(99)

    prepared_pools: list[dict[str, Any]] = []
    for idx, row in enumerate(sample_playlists.itertuples(index=False)):
        pool = list(row.item_indices[:50])
        u_vec = all_u_long[idx % num_users]
        embs = item_embeddings[pool]
        arts = artist_ids[pool]
        gens = genre_ids[pool]
        aud = raw_audio[pool]
        rel = (embs @ u_vec).astype(np.float32)
        is_disc = rng_mask.random(len(pool)) < 0.35
        conc = 1.0 - (len(np.unique(arts)) / len(arts))
        prepared_pools.append(
            {
                "idx": idx,
                "embs": embs,
                "arts": arts,
                "gens": gens,
                "aud": aud,
                "rel": rel,
                "is_disc": is_disc,
                "artist_concentration": float(conc),
            }
        )

    variants: list[tuple[str, dict[str, Any]]] = [
        ("Fisher-Yates Random", {"mode": "fisher_yates", "quota_urgency_weight": 0.0}),
        ("Artist-Spacing Heuristic", {"mode": "artist_spaced", "quota_urgency_weight": 0.0}),
        ("Greedy Relevance", {"mode": "greedy_relevance", "quota_urgency_weight": 0.0}),
        ("Original Adaptive Shuffle", {"mode": "adaptive", "quota_urgency_weight": 0.0}),
        ("Adaptive + Quota Urgency (w=0.40)", {"mode": "adaptive", "quota_urgency_weight": 0.40}),
        ("Adaptive + Quota Urgency (w=0.75)", {"mode": "adaptive", "quota_urgency_weight": 0.75}),
        ("Adaptive + Quota Urgency (w=1.20)", {"mode": "adaptive", "quota_urgency_weight": 1.20}),
    ]

    engine = AdaptiveShuffleEngine()
    overall_seed_means: dict[str, list[dict[str, float]]] = {name: [] for name, _ in variants}
    heavy_seed_means: dict[str, list[dict[str, float]]] = {name: [] for name, _ in variants}
    balanced_seed_means: dict[str, list[dict[str, float]]] = {name: [] for name, _ in variants}

    for seed in seeds:
        for name, cfg_kwargs in variants:
            all_rows, heavy_rows, bal_rows = [], [], []
            for p in prepared_pools:
                cfg = ShuffleConfig(seed=seed + p["idx"], **cfg_kwargs)
                res = engine.sequence(
                    p["embs"], p["arts"], p["gens"], p["aud"], p["rel"], p["is_disc"], cfg
                )
                m = res["metrics"]
                all_rows.append(m)
                if p["artist_concentration"] >= 0.40:
                    heavy_rows.append(m)
                else:
                    bal_rows.append(m)

            overall_seed_means[name].append({k: float(np.mean([r[k] for r in all_rows])) for k in METRIC_KEYS})
            if heavy_rows:
                heavy_seed_means[name].append({k: float(np.mean([r[k] for r in heavy_rows])) for k in METRIC_KEYS})
            if bal_rows:
                balanced_seed_means[name].append({k: float(np.mean([r[k] for r in bal_rows])) for k in METRIC_KEYS})

    n_heavy = sum(1 for p in prepared_pools if p["artist_concentration"] >= 0.40)
    n_bal = len(prepared_pools) - n_heavy

    report = {
        "experiment": "Adaptive Shuffle Remaining-Quota Urgency Evaluation",
        "num_playlists": len(prepared_pools),
        "num_artist_heavy_playlists": n_heavy,
        "num_balanced_playlists": n_bal,
        "seeds": seeds,
        "overall_200_playlists": {name: aggregate_seed_runs(rows) for name, rows in overall_seed_means.items()},
        "artist_heavy_subset": {name: aggregate_seed_runs(rows) for name, rows in heavy_seed_means.items()},
        "balanced_subset": {name: aggregate_seed_runs(rows) for name, rows in balanced_seed_means.items()},
        "runtime_seconds": round(time.time() - t0, 2),
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "shuffle_experiment_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Multi-Seed Shuffle Experiment Complete ({report['runtime_seconds']}s) ===")
    print(f"Playlists: {len(prepared_pools)} total ({n_heavy} artist-heavy >=40%, {n_bal} balanced <40%) | Seeds: {seeds}\n")
    for name in [v[0] for v in variants]:
        r = report["overall_200_playlists"][name]
        print(
            f"{name:36s} | Clump: {r['artist_clump_rate']['mean']*100:5.2f}±{r['artist_clump_rate']['std']*100:.2f}% "
            f"| MinGap: {r['min_artist_gap']['mean']:4.2f} | MedGap: {r['median_artist_gap']['mean']:5.2f} "
            f"| NDCG: {r['session_ndcg']['mean']:.4f} | Smooth: {r['acoustic_smoothness']['mean']*100:5.2f}% "
            f"| BPM: {r['mean_bpm_jump']['mean']:5.2f} | GSwitch: {r['genre_switch_rate']['mean']*100:5.2f}% "
            f"| Valid: {r['sequence_validity']['mean']:.1f}"
        )
    return report


if __name__ == "__main__":
    run_experiment()
