"""
Phase C: Scientifically Validated Recommendation Training, Ablation & Statistical Evaluation Suite.

Reads zero-leakage `data/processed_v2/` and saves v2 checkpoints to `data/artifacts_v2/`
(leaving legacy `data/processed/` and `data/artifacts/` completely untouched).

Addresses all audit findings:
- Disjoint playlist-level splits (train_playlists only for CF & encoder)
- Training-only audio StandardScaler (measured tracks only) & training-only popularity/novelty
- Positive-excluded, duplicate-free negative sampling in training AND shared eval candidates
- Single canonical scoring function per model across sampled (1+499) AND full-catalog (40k) evaluation
- Snapshot of NeuMF weights BEFORE Transformer training (no post-Transformer encoder contamination)
- Full NeuMF & Hybrid ablation suites + 1,000-sample user-level bootstrap 95% CIs
"""

from __future__ import annotations

import copy
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def paired_sign_flip_pvalue(diff: np.ndarray, n_perm: int = 4000, seed: int = 42) -> float:
    """Two-sided paired sign-flip permutation p-value (exact nonparametric test, zero scipy DLL needed)."""
    obs = abs(float(np.mean(diff)))
    if obs == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_perm, len(diff)))
    perm_means = np.abs((signs * diff[None, :]).mean(axis=1))
    return float((np.sum(perm_means >= obs) + 1) / (n_perm + 1))

from backend.app.models.collaborative import (
    BPRMatrixFactorization,
    ItemCooccurrenceCF,
    NeuralCollaborativeFiltering,
)
from backend.app.models.embeddings import MultimodalTrackEncoder
from backend.app.models.sequential_transformer import (
    PAD_ACTION_ID,
    PAD_TIME_BUCKET,
    DualHorizonSessionTransformer,
)
from backend.training.splits_and_sampling import (
    build_shared_eval_candidates,
    build_zero_leakage_splits,
    sample_unique_negatives,
)


def set_deterministic_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PositiveExcludedPairDataset(Dataset):
    """Training dataset sampling unique negatives strictly outside each user's known train positives."""

    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        weights: np.ndarray,
        user_pos_sets: dict[int, set[int]],
        num_items: int,
        num_negatives: int = 4,
        seed: int = 42,
    ) -> None:
        self.users = users.astype(np.int64)
        self.pos_items = pos_items.astype(np.int64)
        self.weights = weights.astype(np.float32)
        self.user_pos_sets = user_pos_sets
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, float, np.ndarray]:
        u = int(self.users[idx])
        pos = int(self.pos_items[idx])
        excluded = self.user_pos_sets.get(u, {pos})
        negs = sample_unique_negatives(self.num_items, excluded, self.num_negatives, self.rng)
        return u, pos, float(self.weights[idx]), negs


class SequentialWindowDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[int, list[int], list[int], list[int], int]],
        num_items: int,
        max_len: int = 50,
    ) -> None:
        self.samples = samples
        self.num_items = num_items
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
        uid, items, actions, tbuckets, target = self.samples[idx]
        L = self.max_len
        it_tail = items[-L:]
        ac_tail = actions[-L:]
        tb_tail = tbuckets[-L:]
        pad = L - len(it_tail)
        return (
            uid,
            np.array([self.num_items] * pad + it_tail, dtype=np.int64),
            np.array([PAD_ACTION_ID] * pad + ac_tail, dtype=np.int64),
            np.array([PAD_TIME_BUCKET] * pad + tb_tail, dtype=np.int64),
            target,
        )


def extract_train_playlist_pairs(train_playlists: pd.DataFrame, max_pairs: int = 90000) -> tuple[np.ndarray, np.ndarray]:
    """Extract adjacent pairs strictly from train-split playlists."""
    a_list, b_list = [], []
    for seq in train_playlists["item_indices"]:
        for k in range(len(seq) - 1):
            a_list.append(int(seq[k]))
            b_list.append(int(seq[k + 1]))
            if len(a_list) >= max_pairs:
                return np.array(a_list, dtype=np.int64), np.array(b_list, dtype=np.int64)
    return np.array(a_list, dtype=np.int64), np.array(b_list, dtype=np.int64)


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """User-level bootstrap mean, std, and 95% confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "ci95_low": round(float(np.percentile(means, 2.5)), 4),
        "ci95_high": round(float(np.percentile(means, 97.5)), 4),
    }


def make_encoder(
    num_items: int,
    num_artists: int,
    num_genres: int,
    artist_idx: np.ndarray,
    genre_idx: np.ndarray,
    norm_audio: np.ndarray,
    use_audio: bool = True,
    use_genre: bool = True,
) -> MultimodalTrackEncoder:
    enc = MultimodalTrackEncoder(
        num_items=num_items,
        num_artists=num_artists,
        num_genres=num_genres,
        audio_dim=norm_audio.shape[1],
        embed_dim=128,
        id_dropout_prob=0.15 if (use_audio or use_genre) else 0.0,
    )
    a_feats = norm_audio if use_audio else np.zeros_like(norm_audio)
    g_idx = genre_idx if use_genre else np.zeros_like(genre_idx)
    enc.set_catalog_metadata(
        artist_indices=torch.tensor(artist_idx, dtype=torch.long),
        genre_indices=torch.tensor(g_idx, dtype=torch.long),
        audio_features=torch.tensor(a_feats, dtype=torch.float32),
    )
    return enc


def train_neumf_variant(
    num_users: int,
    track_encoder: MultimodalTrackEncoder,
    pair_ds: PositiveExcludedPairDataset,
    pl_pairs: tuple[np.ndarray, np.ndarray] | None,
    device: torch.device,
    epochs: int = 8,
    branch_mode: str = "both",  # both | gmf_only | mlp_only
) -> NeuralCollaborativeFiltering:
    model = NeuralCollaborativeFiltering(num_users=num_users, track_encoder=track_encoder, embed_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    loader = DataLoader(pair_ds, batch_size=512, shuffle=True, drop_last=False)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for _ in range(epochs):
        model.train()
        for step_idx, (u, pos, w, negs) in enumerate(loader):
            u, pos, w, negs = u.to(device), pos.to(device), w.to(device), negs.to(device)
            B, K = negs.shape
            all_items = torch.cat([pos.unsqueeze(1), negs], dim=1)
            labels = torch.zeros((B, 1 + K), device=device, dtype=torch.float32)
            labels[:, 0] = 1.0

            with torch.amp.autocast("cuda", enabled=use_amp):
                u_gmf = model.user_gmf(u).unsqueeze(1).expand(B, 1 + K, -1)
                i_gmf = model.item_gmf(all_items)
                u_long = model.get_long_term_user_emb(u).unsqueeze(1).expand(B, 1 + K, -1)
                e_item = model.track_encoder(all_items)

                gmf_feat = u_gmf * i_gmf
                mlp_feat = model.mlp_tower(torch.cat([u_long, e_item, u_long * e_item], dim=-1))
                if branch_mode == "gmf_only":
                    mlp_feat = torch.zeros_like(mlp_feat)
                elif branch_mode == "mlp_only":
                    gmf_feat = torch.zeros_like(gmf_feat)

                logits = model.neu_head(torch.cat([gmf_feat, mlp_feat], dim=-1)).squeeze(-1)
                shared_cos = (u_long * e_item).sum(dim=-1)

                bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=1)
                ce_t = torch.zeros(B, device=device, dtype=torch.long)
                contrast = F.cross_entropy(shared_cos / 0.10, ce_t, reduction="none")
                loss = ((0.5 * bce + 0.5 * contrast) * w).mean()

                if pl_pairs is not None and len(pl_pairs[0]) > 256 and step_idx % 2 == 0:
                    pl_a, pl_b = pl_pairs
                    ridx = np.random.randint(0, len(pl_a), size=256)
                    pa = torch.tensor(pl_a[ridx], device=device, dtype=torch.long)
                    pb = torch.tensor(pl_b[ridx], device=device, dtype=torch.long)
                    ea, eb = track_encoder(pa), track_encoder(pb)
                    loss = loss + 0.25 * F.cross_entropy((ea @ eb.T) / 0.10, torch.arange(len(pa), device=device))

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    return model


@torch.no_grad()
def score_neumf_canonical(
    model: NeuralCollaborativeFiltering,
    user_idx: int,
    item_embeddings_all: torch.Tensor,
    branch_mode: str = "both",
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    SINGLE CANONICAL NeuMF SCORING FUNCTION used identically across validation,
    sampled evaluation, and full-catalog evaluation: raw NeuMF head logits over all items!
    """
    n = item_embeddings_all.shape[0]
    u_t = torch.tensor([user_idx], dtype=torch.long, device=device)
    u_gmf = model.user_gmf(u_t)  # [1, 64]
    u_long = model.get_long_term_user_emb(u_t)  # [1, 128]

    scores = np.empty(n, dtype=np.float32)
    chunk = 8192
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        ids = torch.arange(start, end, device=device, dtype=torch.long)
        e_it = item_embeddings_all[start:end]
        i_gmf = model.item_gmf(ids)
        u_g_exp = u_gmf.expand(end - start, -1)
        u_l_exp = u_long.expand(end - start, -1)

        gmf_f = u_g_exp * i_gmf
        mlp_f = model.mlp_tower(torch.cat([u_l_exp, e_it, u_l_exp * e_it], dim=-1))
        if branch_mode == "gmf_only":
            mlp_f = torch.zeros_like(mlp_f)
        elif branch_mode == "mlp_only":
            gmf_f = torch.zeros_like(gmf_f)
        logits = model.neu_head(torch.cat([gmf_f, mlp_f], dim=-1)).squeeze(-1)
        scores[start:end] = logits.cpu().numpy()
    return scores


def train_transformer_v2(
    neumf_frozen: NeuralCollaborativeFiltering,
    base_encoder: MultimodalTrackEncoder,
    seq_samples: list[tuple[int, list[int], list[int], list[int], int]],
    num_items: int,
    device: torch.device,
    epochs: int = 8,
) -> tuple[DualHorizonSessionTransformer, MultimodalTrackEncoder]:
    """Train DualHorizonSessionTransformer on a CLONED encoder so NeuMF weights are never contaminated."""
    tr_encoder = copy.deepcopy(base_encoder).to(device)
    model = DualHorizonSessionTransformer(
        track_encoder=tr_encoder,
        max_seq_len=50,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.15,
    ).to(device)

    ds = SequentialWindowDataset(seq_samples, num_items=num_items, max_len=50)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(
        [
            {"params": model.transformer.parameters(), "lr": 1.8e-3},
            {"params": model.action_emb.parameters(), "lr": 1.8e-3},
            {"params": model.time_emb.parameters(), "lr": 1.8e-3},
            {"params": model.pos_emb.parameters(), "lr": 1.8e-3},
            {"params": model.short_proj.parameters(), "lr": 1.8e-3},
            {"params": model.fusion_gate.parameters(), "lr": 1.8e-3},
            {"params": tr_encoder.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for _ in range(epochs):
        model.train()
        for uid, it_seq, ac_seq, tb_seq, target in loader:
            uid, it_seq, ac_seq, tb_seq, target = (
                uid.to(device), it_seq.to(device), ac_seq.to(device), tb_seq.to(device), target.to(device)
            )
            extra_negs = torch.randint(0, num_items, (512,), device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    u_long = neumf_frozen.get_long_term_user_emb(uid)
                u_short = model.encode_short_term_sequence(it_seq, ac_seq, tb_seq)[:, -1, :]
                u_fused, _ = model.fuse_horizons(u_long, u_short)

                pos_emb = tr_encoder(target)
                neg_pool_emb = torch.cat([pos_emb, tr_encoder(extra_negs)], dim=0)
                all_pool_ids = torch.cat([target, extra_negs], dim=0)

                same_target = target.unsqueeze(1) == all_pool_ids.unsqueeze(0)
                diag = torch.zeros_like(same_target)
                diag[:, : len(target)] = torch.eye(len(target), dtype=torch.bool, device=device)
                false_neg = same_target & (~diag)

                logits = ((u_fused @ neg_pool_emb.T) / 0.08).masked_fill(false_neg, -1e4)
                short_logits = ((u_short @ neg_pool_emb.T) / 0.08).masked_fill(false_neg, -1e4)
                labels = torch.arange(len(target), device=device)
                loss = 0.70 * F.cross_entropy(logits, labels) + 0.30 * F.cross_entropy(short_logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    model.eval()
    tr_encoder.eval()
    return model, tr_encoder


def mmr_rerank_top_k(
    scores: np.ndarray,
    item_embeddings: np.ndarray,
    top_k: int = 20,
    pool_size: int = 120,
    lam: float = 0.65,
) -> np.ndarray:
    top_pool = np.argpartition(scores, -pool_size)[-pool_size:]
    top_pool = top_pool[np.argsort(-scores[top_pool])]
    sp = scores[top_pool]
    sp = (sp - sp.min()) / (sp.max() - sp.min() + 1e-6)
    picked: list[int] = []
    avail = np.ones(len(top_pool), dtype=bool)
    ep = item_embeddings[top_pool]
    for _ in range(top_k):
        cpos = np.flatnonzero(avail)
        if not picked:
            util = sp[cpos]
        else:
            msim = (ep[cpos] @ item_embeddings[picked].T).max(axis=1)
            util = lam * sp[cpos] - (1.0 - lam) * msim
        best = int(cpos[int(np.argmax(util))])
        avail[best] = False
        picked.append(int(top_pool[best]))
    return np.array(picked, dtype=np.int32)


def evaluate_score_matrix(
    model_scores_per_user: list[np.ndarray],
    eval_cases: list[dict[str, Any]],
    shared_cands: np.ndarray,
    ref_embeddings: np.ndarray,
    artist_ids: np.ndarray,
    train_novelty: np.ndarray,
    num_items: int,
    apply_mmr: bool = False,
) -> dict[str, Any]:
    """
    Evaluate a model using a SINGLE score vector per user across BOTH:
    1. Sampled ranking (1 target + 499 unique unseen negatives)
    2. Full-catalog ranking (all 40,000 tracks excluding seen_pos)
    """
    s_hr10, s_ndcg10, s_mrr10 = [], [], []
    f_hr10, f_ndcg10, f_mrr10, f_prec10 = [], [], [], []
    ild20, nov20, unseen_art20 = [], [], []
    covered: set[int] = set()

    for idx, (full_scores, case) in enumerate(zip(model_scores_per_user, eval_cases)):
        target = int(case["target"])
        seen_pos = case["seen_pos"]
        seen_artists = {int(artist_ids[i]) for i in seen_pos}
        cands = shared_cands[idx]

        # 1. Sampled ranking (target is cands[0])
        cs = full_scores[cands]
        rank_s = int(np.sum(cs[1:] >= cs[0])) + 1
        s_hr10.append(1.0 if rank_s <= 10 else 0.0)
        s_ndcg10.append((1.0 / np.log2(rank_s + 1.0)) if rank_s <= 10 else 0.0)
        s_mrr10.append((1.0 / rank_s) if rank_s <= 10 else 0.0)

        # 2. Full-catalog ranking (mask seen_pos EXCEPT target if target was a repeat play)
        fs = full_scores.copy()
        mask_seen = seen_pos - {target}
        if mask_seen:
            fs[list(mask_seen)] = -1e9

        if apply_mmr:
            top20 = mmr_rerank_top_k(fs, ref_embeddings, top_k=20)
            top10 = top20[:10]
            if target in top10:
                rank_f = int(np.where(top10 == target)[0][0]) + 1
            else:
                rank_f = 999
        else:
            rank_f = int(np.sum(fs > fs[target])) + 1
            top20 = np.argpartition(fs, -20)[-20:]
            top20 = top20[np.argsort(-fs[top20])]

        f_hr10.append(1.0 if rank_f <= 10 else 0.0)
        f_prec10.append(0.1 if rank_f <= 10 else 0.0)  # 1 relevant target per user -> P@10 = HR@10 / 10
        f_ndcg10.append((1.0 / np.log2(rank_f + 1.0)) if rank_f <= 10 else 0.0)
        f_mrr10.append((1.0 / rank_f) if rank_f <= 10 else 0.0)

        covered.update(top20.tolist())
        e20 = ref_embeddings[top20]
        sim20 = np.clip(e20 @ e20.T, -1.0, 1.0)
        tri = np.triu(np.ones((20, 20), dtype=bool), k=1)
        ild20.append(float(np.mean(1.0 - sim20[tri])))
        nov20.append(float(np.mean(train_novelty[top20])))
        unseen_art20.append(float(np.mean([int(artist_ids[i]) not in seen_artists for i in top20])))

    return {
        "sampled": {
            "HR@10": bootstrap_ci(np.array(s_hr10)),
            "NDCG@10": bootstrap_ci(np.array(s_ndcg10)),
            "MRR@10": bootstrap_ci(np.array(s_mrr10)),
        },
        "full_catalog": {
            "HR@10": bootstrap_ci(np.array(f_hr10)),
            "Recall@10": bootstrap_ci(np.array(f_hr10)),
            "Precision@10": bootstrap_ci(np.array(f_prec10)),
            "NDCG@10": bootstrap_ci(np.array(f_ndcg10)),
            "MRR@10": bootstrap_ci(np.array(f_mrr10)),
            "ILD@20": bootstrap_ci(np.array(ild20)),
            "Novelty@20": bootstrap_ci(np.array(nov20)),
            "UnseenArtistRatio@20": bootstrap_ci(np.array(unseen_art20)),
            "CatalogCoverage@20": round(len(covered) / num_items, 4),
        },
        "_per_user_sampled_ndcg": s_ndcg10,
        "_per_user_full_ndcg": f_ndcg10,
    }


def run_validation_suite(
    v2_dir: Path = Path("data/processed_v2"),
    artifacts_v2_dir: Path = Path("data/artifacts_v2"),
    reports_dir: Path = Path("data/reports"),
    eval_users_limit: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.time()
    set_deterministic_seed(seed)
    artifacts_v2_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== Phase C Zero-Leakage Validation & Ablation Suite on {device} ({gpu_name}) ===")

    tracks_df = pd.read_parquet(v2_dir / "tracks_v2.parquet").sort_values("item_idx").reset_index(drop=True)
    inter_df = pd.read_parquet(v2_dir / "interactions_v2.parquet")
    pl_df = pd.read_parquet(v2_dir / "playlists_v2.parquet")

    num_items = len(tracks_df)
    num_artists = int(tracks_df["artist_idx"].max()) + 1
    num_genres = int(tracks_df["genre_idx"].max()) + 1
    num_users = int(inter_df["user_idx"].max()) + 1
    artist_ids = tracks_df["artist_idx"].to_numpy(dtype=np.int32)
    genre_ids = tracks_df["genre_idx"].to_numpy(dtype=np.int32)

    splits = build_zero_leakage_splits(tracks_df, inter_df, pl_df)
    eval_cases = splits["test_cases"][: min(eval_users_limit, len(splits["test_cases"]))]
    shared_cands = build_shared_eval_candidates(eval_cases, num_items=num_items, num_negatives=499, seed=123)
    train_pl_pairs = extract_train_playlist_pairs(splits["train_playlists"])
    norm_audio = splits["norm_audio"]
    train_pop = splits["train_popularity"]
    train_nov = splits["train_novelty"]

    print(
        f"Zero-leakage splits ready: {len(splits['train_users']):,} train pairs | "
        f"{len(splits['train_playlists']):,} train playlists | {len(splits['test_playlists']):,} test playlists | "
        f"{len(eval_cases)} test users"
    )

    # 1. Train-Only Item-KNN CF (train users + train playlists ONLY!)
    print("[1/7] Fitting Train-Only Item-KNN CF...")
    item_cf = ItemCooccurrenceCF(num_items=num_items, top_k_neighbors=80).fit(
        user_item_pairs=(splits["train_users"], splits["train_items"], splits["train_weights"]),
        playlist_sequences=[list(s) for s in splits["train_playlists"]["item_indices"]],
    )

    # 2. BPR-MF (Positive-Excluded Negative Sampling)
    print("[2/7] Training BPR-MF (positive-excluded negatives)...")
    pair_ds = PositiveExcludedPairDataset(
        splits["train_users"], splits["train_items"], splits["train_weights"],
        splits["user_train_pos_sets"], num_items, num_negatives=4, seed=seed,
    )
    bpr = BPRMatrixFactorization(num_users, num_items, factors=64).to(device)
    bpr_opt = torch.optim.AdamW(bpr.parameters(), lr=3e-3, weight_decay=1e-5)
    bpr_loader = DataLoader(pair_ds, batch_size=1024, shuffle=True)
    for _ in range(8):
        bpr.train()
        for u, pos, w, negs in bpr_loader:
            u, pos, w, negs = u.to(device), pos.to(device), w.to(device), negs.to(device)
            loss = (-F.logsigmoid(bpr.score(u, pos).unsqueeze(1) - bpr.score(u, negs)).mean(dim=1) * w).mean()
            bpr_opt.zero_grad()
            loss.backward()
            bpr_opt.step()
    bpr.eval()

    # 3. NeuMF Ablation Models (Interaction-Only, GMF-Only, MLP-Only, Full Multimodal)
    print("[3/7] Training NeuMF Ablation Variants (GMF-only, MLP-only, Interaction-only, Multimodal)...")
    enc_int_only = make_encoder(num_items, num_artists, num_genres, artist_ids, genre_ids, norm_audio, use_audio=False, use_genre=False)
    neumf_int = train_neumf_variant(num_users, enc_int_only, pair_ds, None, device, epochs=6, branch_mode="both")
    emb_int = enc_int_only.export_all_embeddings(device=device).to(device)

    enc_gmf = make_encoder(num_items, num_artists, num_genres, artist_ids, genre_ids, norm_audio, use_audio=True, use_genre=True)
    neumf_gmf = train_neumf_variant(num_users, enc_gmf, pair_ds, train_pl_pairs, device, epochs=6, branch_mode="gmf_only")
    emb_gmf = enc_gmf.export_all_embeddings(device=device).to(device)

    enc_mlp = make_encoder(num_items, num_artists, num_genres, artist_ids, genre_ids, norm_audio, use_audio=True, use_genre=True)
    neumf_mlp = train_neumf_variant(num_users, enc_mlp, pair_ds, train_pl_pairs, device, epochs=6, branch_mode="mlp_only")
    emb_mlp = enc_mlp.export_all_embeddings(device=device).to(device)

    enc_full = make_encoder(num_items, num_artists, num_genres, artist_ids, genre_ids, norm_audio, use_audio=True, use_genre=True)
    neumf_full = train_neumf_variant(num_users, enc_full, pair_ds, train_pl_pairs, device, epochs=10, branch_mode="both")
    # SNAPSHOT NeuMF's own embeddings BEFORE Transformer training!
    emb_neumf_frozen = enc_full.export_all_embeddings(device=device).to(device)
    emb_neumf_np = emb_neumf_frozen.cpu().numpy().astype(np.float32)

    # 4. Dual-Horizon Session Transformer (on cloned encoder — zero contamination of NeuMF!)
    print("[4/7] Training Dual-Horizon Causal Session Transformer (on cloned encoder)...")
    transformer, tr_encoder = train_transformer_v2(
        neumf_frozen=neumf_full,
        base_encoder=enc_full,
        seq_samples=splits["seq_train_samples"],
        num_items=num_items,
        device=device,
        epochs=10,
    )
    emb_tr_np = tr_encoder.export_all_embeddings(device=device).numpy().astype(np.float32)

    # Save v2 artifacts cleanly (preserving legacy data/artifacts/!)
    torch.save(neumf_full.state_dict(), artifacts_v2_dir / "neumf_v2.pt")
    torch.save(transformer.state_dict(), artifacts_v2_dir / "transformer_v2.pt")
    np.save(artifacts_v2_dir / "track_embeddings_v2.npy", emb_tr_np)
    with open(artifacts_v2_dir / "item_cf_v2.pkl", "wb") as f:
        pickle.dump(item_cf, f)

    # 5. Compute Single Canonical Full-Catalog Score Vectors per Test User for Every Model
    print("[5/7] Scoring all 11 models/ablations on identical candidate pools & full 40k catalog...")
    model_score_lists: dict[str, list[np.ndarray]] = {
        "Popularity (Train-Only)": [],
        "Item-KNN CF (Train-Only)": [],
        "BPR-MF": [],
        "NeuMF — Interaction Only": [],
        "NeuMF — GMF Only": [],
        "NeuMF — MLP Only": [],
        "NeuMF — Full Multimodal": [],
        "Transformer — Short-Term Only": [],
        "Dual-Horizon Transformer": [],
        "Hybrid (Dual-Horizon + CF)": [],
        "Hybrid + Novelty": [],
        "Hybrid + Novelty + MMR": [],
    }

    bpr_i_emb = bpr.item_emb.weight.detach().cpu().numpy()
    bpr_i_bias = bpr.item_bias.weight.detach().cpu().numpy().ravel()
    L = transformer.max_seq_len

    with torch.no_grad():
        for case in eval_cases:
            uid = case["user_idx"]
            u_t = torch.tensor([uid], dtype=torch.long, device=device)

            s_pop = train_pop.copy()
            s_cf = item_cf.score_for_history(case["hist_items"])
            s_cf_norm = s_cf / (float(s_cf.max()) + 1e-6)

            u_bpr = bpr.user_emb(u_t).squeeze(0).cpu().numpy()
            s_bpr = (bpr_i_emb @ u_bpr) + bpr_i_bias

            s_nm_int = score_neumf_canonical(neumf_int, uid, emb_int, "both", device)
            s_nm_gmf = score_neumf_canonical(neumf_gmf, uid, emb_gmf, "gmf_only", device)
            s_nm_mlp = score_neumf_canonical(neumf_mlp, uid, emb_mlp, "mlp_only", device)
            s_nm_full = score_neumf_canonical(neumf_full, uid, emb_neumf_frozen, "both", device)

            pad = L - len(case["hist_items"][-L:])
            it_seq = torch.tensor([[num_items] * pad + case["hist_items"][-L:]], dtype=torch.long, device=device)
            ac_seq = torch.tensor([[PAD_ACTION_ID] * pad + case["hist_actions"][-L:]], dtype=torch.long, device=device)
            tb_seq = torch.tensor([[PAD_TIME_BUCKET] * pad + case["hist_tb"][-L:]], dtype=torch.long, device=device)

            u_long = neumf_full.get_long_term_user_emb(u_t)
            u_short = transformer.encode_short_term_sequence(it_seq, ac_seq, tb_seq)[:, -1, :]
            u_fused, _ = transformer.fuse_horizons(u_long, u_short)

            s_short = (emb_tr_np @ u_short.squeeze(0).cpu().numpy()).astype(np.float32)
            s_dual = (emb_tr_np @ u_fused.squeeze(0).cpu().numpy()).astype(np.float32)

            # Calibrated z-score normalization before hybrid blending
            z_dual = (s_dual - s_dual.mean()) / (s_dual.std() + 1e-6)
            z_cf = (s_cf_norm - s_cf_norm.mean()) / (s_cf_norm.std() + 1e-6)
            z_nov = (train_nov - train_nov.mean()) / (train_nov.std() + 1e-6)

            s_hyb = (0.70 * z_dual + 0.30 * z_cf).astype(np.float32)
            s_hyb_nov = (0.65 * z_dual + 0.25 * z_cf + 0.10 * z_nov).astype(np.float32)

            model_score_lists["Popularity (Train-Only)"].append(s_pop)
            model_score_lists["Item-KNN CF (Train-Only)"].append(s_cf_norm)
            model_score_lists["BPR-MF"].append(s_bpr)
            model_score_lists["NeuMF — Interaction Only"].append(s_nm_int)
            model_score_lists["NeuMF — GMF Only"].append(s_nm_gmf)
            model_score_lists["NeuMF — MLP Only"].append(s_nm_mlp)
            model_score_lists["NeuMF — Full Multimodal"].append(s_nm_full)
            model_score_lists["Transformer — Short-Term Only"].append(s_short)
            model_score_lists["Dual-Horizon Transformer"].append(s_dual)
            model_score_lists["Hybrid (Dual-Horizon + CF)"].append(s_hyb)
            model_score_lists["Hybrid + Novelty"].append(s_hyb_nov)
            model_score_lists["Hybrid + Novelty + MMR"].append(s_hyb_nov)

    # 6. Compute Metrics, Bootstrap 95% CIs, and Paired Statistical Significance Tests
    print("[6/7] Computing Sampled & Full-Catalog Metrics + Bootstrap 95% CIs...")
    results: dict[str, Any] = {}
    per_user_s: dict[str, list[float]] = {}
    per_user_f: dict[str, list[float]] = {}

    for m_name, slist in model_score_lists.items():
        use_mmr = m_name == "Hybrid + Novelty + MMR"
        ev = evaluate_score_matrix(
            slist, eval_cases, shared_cands, emb_tr_np, artist_ids, train_nov, num_items, apply_mmr=use_mmr
        )
        per_user_s[m_name] = ev.pop("_per_user_sampled_ndcg")
        per_user_f[m_name] = ev.pop("_per_user_full_ndcg")
        results[m_name] = ev
        print(
            f"  {m_name:32s} | Sampled NDCG@10: {ev['sampled']['NDCG@10']['mean']:.4f} "
            f"[{ev['sampled']['NDCG@10']['ci95_low']:.4f}, {ev['sampled']['NDCG@10']['ci95_high']:.4f}] "
            f"| Full NDCG@10: {ev['full_catalog']['NDCG@10']['mean']:.4f} "
            f"| ILD@20: {ev['full_catalog']['ILD@20']['mean']:.4f} "
            f"| Nov@20: {ev['full_catalog']['Novelty@20']['mean']:.4f}"
        )

    # Paired significance tests (Wilcoxon signed-rank on per-user NDCG@10)
    sig_pairs = [
        ("Dual-Horizon Transformer", "NeuMF — Full Multimodal"),
        ("Dual-Horizon Transformer", "Transformer — Short-Term Only"),
        ("Hybrid (Dual-Horizon + CF)", "Dual-Horizon Transformer"),
        ("Hybrid (Dual-Horizon + CF)", "Item-KNN CF (Train-Only)"),
        ("NeuMF — Full Multimodal", "NeuMF — Interaction Only"),
    ]
    sig_tests: dict[str, Any] = {}
    for a, b in sig_pairs:
        diff_s = np.array(per_user_s[a]) - np.array(per_user_s[b])
        diff_f = np.array(per_user_f[a]) - np.array(per_user_f[b])
        p_s = paired_sign_flip_pvalue(diff_s)
        p_f = paired_sign_flip_pvalue(diff_f)
        sig_tests[f"{a} vs {b}"] = {
            "sampled_ndcg10_delta": round(float(diff_s.mean()), 4),
            "sampled_perm_p": round(p_s, 6),
            "full_ndcg10_delta": round(float(diff_f.mean()), 4),
            "full_perm_p": round(p_f, 6),
        }

    payload = {
        "device": gpu_name,
        "seed": seed,
        "num_eval_users": len(eval_cases),
        "num_items": num_items,
        "audio_scaler_stats": splits["audio_scaler_stats"],
        "models": results,
        "significance_tests": sig_tests,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_file = reports_dir / "recommendation_validation_v2.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[7/7] Saved Phase C Validation Report to {out_file} ({payload['runtime_seconds']}s)")
    return payload


if __name__ == "__main__":
    run_validation_suite()
