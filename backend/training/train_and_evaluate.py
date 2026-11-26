"""
End-to-End GPU Training & Quantitative Benchmark Evaluation Pipeline.

Trains:
1. Popularity & Item-KNN Co-occurrence CF (User Histories + Spotify MPD Playlists)
2. BPR Matrix Factorization (Latent Collaborative Filtering Baseline)
3. Neural Collaborative Filtering (NeuMF) + Shared MultimodalTrackEncoder (R^128)
4. Dual-Horizon Causal Session Transformer (Short-Term + Long-Term Gated Fusion)

Evaluates:
- Suite A: Recommendation Accuracy, Diversity, Novelty & Catalog Coverage
- Suite B: Adaptive Shuffle vs. Fisher-Yates, Artist-Spaced & Greedy Order (200 Playlists)
- Suite C: Online Feedback Adaptation under Simulated Mood Shift & Early Skips
- Generates publication-ready charts in `data/reports/` and JSON metrics in `data/reports/evaluation_results.json`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.app.engine.adaptive_shuffle import AdaptiveShuffleEngine, ShuffleConfig
from backend.app.engine.online_feedback import UserFeedbackState
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
    bucketize_time_deltas,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InteractionPairDataset(Dataset):
    """Positive user-item pairs + playlist co-occurrence pairs for NeuMF & Multimodal Encoder."""

    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        weights: np.ndarray,
        num_items: int,
        num_negatives: int = 4,
    ) -> None:
        self.users = users.astype(np.int64)
        self.pos_items = pos_items.astype(np.int64)
        self.weights = weights.astype(np.float32)
        self.num_items = num_items
        self.num_negatives = num_negatives

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[int, int, float, np.ndarray]:
        negs = np.random.randint(0, self.num_items, size=self.num_negatives, dtype=np.int64)
        return int(self.users[idx]), int(self.pos_items[idx]), float(self.weights[idx]), negs


class SequentialWindowDataset(Dataset):
    """Sliding causal windows (including skips!) predicting next positive track."""

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
        items_tail = items[-L:]
        acts_tail = actions[-L:]
        tb_tail = tbuckets[-L:]
        pad = L - len(items_tail)

        it_arr = np.array([self.num_items] * pad + items_tail, dtype=np.int64)
        ac_arr = np.array([PAD_ACTION_ID] * pad + acts_tail, dtype=np.int64)
        tb_arr = np.array([PAD_TIME_BUCKET] * pad + tb_tail, dtype=np.int64)
        return uid, it_arr, ac_arr, tb_arr, target


def build_chronological_splits(interactions_df: pd.DataFrame) -> dict[str, Any]:
    """Leave-last-two positive events out per user for validation and testing."""
    train_u, train_i, train_w = [], [], []
    seq_train_samples: list[tuple[int, list[int], list[int], list[int], int]] = []
    val_cases: list[dict[str, Any]] = []
    test_cases: list[dict[str, Any]] = []
    user_train_pos_sets: dict[int, set[int]] = {}

    for uid, grp in interactions_df.groupby("user_idx", sort=False):
        uid_int = int(uid)
        items = grp["item_idx"].to_numpy(dtype=np.int32)
        actions = grp["action_id"].to_numpy(dtype=np.int32)
        rewards = grp["reward"].to_numpy(dtype=np.float32)
        ts = grp["unix_ts"].to_numpy(dtype=np.int64)

        pos_positions = np.flatnonzero(actions >= 2)
        if len(pos_positions) < 8:
            continue

        val_pos_idx = int(pos_positions[-2])
        test_pos_idx = int(pos_positions[-1])
        if val_pos_idx < 5:
            continue

        deltas = np.zeros(len(ts), dtype=np.int64)
        deltas[0] = 3600
        deltas[1:] = np.maximum(0, ts[1:] - ts[:-1])
        tbuckets = bucketize_time_deltas(torch.tensor(deltas, dtype=torch.long)).numpy()

        # Training slice: [0 : val_pos_idx]
        tr_items = items[:val_pos_idx]
        tr_actions = actions[:val_pos_idx]
        tr_rewards = rewards[:val_pos_idx]
        tr_tb = tbuckets[:val_pos_idx]

        pos_mask = tr_actions >= 2
        pos_tr_items = tr_items[pos_mask]
        pos_tr_weights = np.clip(tr_rewards[pos_mask], 0.3, 1.0)
        user_train_pos_sets[uid_int] = set(pos_tr_items.tolist())

        # Deduplicate user-item pairs with summed log weight for CF/BPR/NeuMF
        item_w_acc: dict[int, float] = {}
        for it, w in zip(pos_tr_items.tolist(), pos_tr_weights.tolist()):
            item_w_acc[it] = item_w_acc.get(it, 0.0) + w
        for it, w_sum in item_w_acc.items():
            train_u.append(uid_int)
            train_i.append(it)
            train_w.append(float(1.0 + np.log1p(w_sum)))

        # Build sliding windows ending at positive targets inside tr_items (stride 2 for diversity & speed)
        tr_pos_locs = np.flatnonzero(tr_actions >= 2)
        for p_loc in tr_pos_locs[3::2]:
            p_int = int(p_loc)
            start = max(0, p_int - 50)
            seq_train_samples.append(
                (
                    uid_int,
                    tr_items[start:p_int].tolist(),
                    tr_actions[start:p_int].tolist(),
                    tr_tb[start:p_int].tolist(),
                    int(tr_items[p_int]),
                )
            )

        # Validation case
        v_start = max(0, val_pos_idx - 50)
        val_cases.append(
            {
                "user_idx": uid_int,
                "hist_items": items[v_start:val_pos_idx].tolist(),
                "hist_actions": actions[v_start:val_pos_idx].tolist(),
                "hist_tb": tbuckets[v_start:val_pos_idx].tolist(),
                "target": int(items[val_pos_idx]),
                "seen_pos": set(pos_tr_items.tolist()),
            }
        )

        # Test case
        te_start = max(0, test_pos_idx - 50)
        test_seen = set(items[:test_pos_idx][actions[:test_pos_idx] >= 2].tolist())
        test_cases.append(
            {
                "user_idx": uid_int,
                "hist_items": items[te_start:test_pos_idx].tolist(),
                "hist_actions": actions[te_start:test_pos_idx].tolist(),
                "hist_tb": tbuckets[te_start:test_pos_idx].tolist(),
                "target": int(items[test_pos_idx]),
                "seen_pos": test_seen,
            }
        )

    return {
        "train_users": np.array(train_u, dtype=np.int32),
        "train_items": np.array(train_i, dtype=np.int32),
        "train_weights": np.array(train_w, dtype=np.float32),
        "seq_train_samples": seq_train_samples,
        "val_cases": val_cases,
        "test_cases": test_cases,
        "user_train_pos_sets": user_train_pos_sets,
    }


def extract_playlist_adjacent_pairs(playlists_df: pd.DataFrame, max_pairs: int = 120000) -> tuple[np.ndarray, np.ndarray]:
    """Extract adjacent track pairs from Spotify MPD playlists for track-space co-occurrence alignment."""
    a_list, b_list = [], []
    for seq in playlists_df["item_indices"]:
        for k in range(len(seq) - 1):
            a_list.append(int(seq[k]))
            b_list.append(int(seq[k + 1]))
            if len(a_list) >= max_pairs:
                break
        if len(a_list) >= max_pairs:
            break
    return np.array(a_list, dtype=np.int64), np.array(b_list, dtype=np.int64)


def train_bpr_mf(
    num_users: int,
    num_items: int,
    pair_ds: InteractionPairDataset,
    device: torch.device,
    epochs: int = 10,
) -> BPRMatrixFactorization:
    print("\n--- Training Model 1: BPR Matrix Factorization ---")
    model = BPRMatrixFactorization(num_users=num_users, num_items=num_items, factors=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    loader = DataLoader(pair_ds, batch_size=1024, shuffle=True, drop_last=False)

    model.train()
    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for u, pos, w, negs in loader:
            u = u.to(device)
            pos = pos.to(device)
            w = w.to(device)
            negs = negs.to(device)  # [B, K]

            pos_s = model.score(u, pos).unsqueeze(1)  # [B, 1]
            neg_s = model.score(u, negs)  # [B, K]
            loss = (-F.logsigmoid(pos_s - neg_s).mean(dim=1) * w).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        if ep in (1, epochs // 2, epochs):
            print(f"  [BPR-MF] Epoch {ep}/{epochs} | Loss: {total_loss / max(1, len(loader)):.4f}")
    model.eval()
    return model


def train_neumf_and_encoder(
    num_users: int,
    track_encoder: MultimodalTrackEncoder,
    pair_ds: InteractionPairDataset,
    pl_pairs: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    epochs: int = 12,
) -> NeuralCollaborativeFiltering:
    print("\n--- Training Model 2: NeuMF + Shared Multimodal Track Encoder (R^128) ---")
    model = NeuralCollaborativeFiltering(num_users=num_users, track_encoder=track_encoder, embed_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    loader = DataLoader(pair_ds, batch_size=512, shuffle=True, drop_last=False)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pl_a, pl_b = pl_pairs
    num_pl = len(pl_a)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step_idx, (u, pos, w, negs) in enumerate(loader):
            u = u.to(device)
            pos = pos.to(device)
            w = w.to(device)
            negs = negs.to(device)  # [B, K]
            B, K = negs.shape

            all_items = torch.cat([pos.unsqueeze(1), negs], dim=1)  # [B, 1+K]
            labels = torch.zeros((B, 1 + K), device=device, dtype=torch.float32)
            labels[:, 0] = 1.0

            with torch.amp.autocast("cuda", enabled=use_amp):
                neumf_logits, shared_cos = model.score_pairs(u, all_items)
                bce_loss = F.binary_cross_entropy_with_logits(neumf_logits, labels, reduction="none").mean(dim=1)
                # Contrastive alignment in R^128 space (temperature = 0.10)
                ce_targets = torch.zeros(B, device=device, dtype=torch.long)
                contrast_loss = F.cross_entropy(shared_cos / 0.10, ce_targets, reduction="none")
                loss = ((0.5 * bce_loss + 0.5 * contrast_loss) * w).mean()

                # Co-occurrence playlist alignment step every 2 batches
                if num_pl > 256 and step_idx % 2 == 0:
                    ridx = np.random.randint(0, num_pl, size=256)
                    pa = torch.tensor(pl_a[ridx], device=device, dtype=torch.long)
                    pb = torch.tensor(pl_b[ridx], device=device, dtype=torch.long)
                    ea = track_encoder(pa)
                    eb = track_encoder(pb)
                    pl_logits = (ea @ eb.T) / 0.10
                    pl_targets = torch.arange(len(pa), device=device)
                    pl_loss = F.cross_entropy(pl_logits, pl_targets)
                    loss = loss + 0.25 * pl_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())

        if ep in (1, epochs // 2, epochs):
            print(f"  [NeuMF + MultimodalEncoder] Epoch {ep}/{epochs} | Loss: {total_loss / max(1, len(loader)):.4f}")

    model.eval()
    return model


def train_dual_horizon_transformer(
    neumf: NeuralCollaborativeFiltering,
    track_encoder: MultimodalTrackEncoder,
    seq_samples: list[tuple[int, list[int], list[int], list[int], int]],
    num_items: int,
    device: torch.device,
    epochs: int = 12,
) -> DualHorizonSessionTransformer:
    print("\n--- Training Model 3: Dual-Horizon Causal Session Transformer ---")
    model = DualHorizonSessionTransformer(
        track_encoder=track_encoder,
        max_seq_len=50,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.15,
    ).to(device)

    ds = SequentialWindowDataset(seq_samples, num_items=num_items, max_len=50)
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

    # Fine-tune encoder gently while training Transformer & fusion gate cleanly
    opt = torch.optim.AdamW(
        [
            {"params": model.transformer.parameters(), "lr": 1.8e-3},
            {"params": model.action_emb.parameters(), "lr": 1.8e-3},
            {"params": model.time_emb.parameters(), "lr": 1.8e-3},
            {"params": model.pos_emb.parameters(), "lr": 1.8e-3},
            {"params": model.short_proj.parameters(), "lr": 1.8e-3},
            {"params": model.fusion_gate.parameters(), "lr": 1.8e-3},
            {"params": track_encoder.parameters(), "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for uid, it_seq, ac_seq, tb_seq, target in loader:
            uid = uid.to(device)
            it_seq = it_seq.to(device)
            ac_seq = ac_seq.to(device)
            tb_seq = tb_seq.to(device)
            target = target.to(device)

            # Sample extra random negatives to complement in-batch negatives
            extra_negs = torch.randint(0, num_items, (512,), device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    u_long = neumf.get_long_term_user_emb(uid)
                u_short_all = model.encode_short_term_sequence(it_seq, ac_seq, tb_seq)
                u_short = u_short_all[:, -1, :]
                u_fused, _ = model.fuse_horizons(u_long, u_short)

                pos_emb = track_encoder(target)  # [B, D]
                neg_pool_emb = torch.cat([pos_emb, track_encoder(extra_negs)], dim=0)  # [B+512, D]

                # Mask accidental false negatives in batch/sampled pool
                all_pool_ids = torch.cat([target, extra_negs], dim=0)  # [B+512]
                logits = (u_fused @ neg_pool_emb.T) / 0.08  # [B, B+512]
                same_target = target.unsqueeze(1) == all_pool_ids.unsqueeze(0)
                diag_mask = torch.zeros_like(same_target)
                diag_mask[:, : len(target)] = torch.eye(len(target), dtype=torch.bool, device=device)
                false_neg_mask = same_target & (~diag_mask)
                logits = logits.masked_fill(false_neg_mask, -1e4)

                labels = torch.arange(len(target), device=device)
                # Also supervise u_short directly so short-term context remains sharp
                short_logits = ((u_short @ neg_pool_emb.T) / 0.08).masked_fill(false_neg_mask, -1e4)
                loss = 0.70 * F.cross_entropy(logits, labels) + 0.30 * F.cross_entropy(short_logits, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())

        if ep in (1, epochs // 2, epochs):
            print(f"  [DualHorizonTransformer] Epoch {ep}/{epochs} | Loss: {total_loss / max(1, len(loader)):.4f}")

    model.eval()
    return model


@torch.no_grad()
def run_recommendation_benchmarks(
    test_cases: list[dict[str, Any]],
    num_items: int,
    tracks_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    item_cf: ItemCooccurrenceCF,
    bpr_mf: BPRMatrixFactorization,
    neumf: NeuralCollaborativeFiltering,
    transformer: DualHorizonSessionTransformer,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Evaluate 6 models on:
    1. Sampled Ranking (1 ground-truth target + 499 unseen negatives): HR@10, NDCG@10, MRR@10
    2. Top-20 Full-Catalog Recommendation Quality: ILD@20, Novelty@20, Coverage@20, UnseenArtist@20
    """
    print("\n--- Running Benchmark Suite A: Recommendation Models Evaluation ---")
    rng = np.random.default_rng(123)
    artist_ids = tracks_df["artist_idx"].to_numpy(dtype=np.int32)

    pop_raw = tracks_df["popularity"].to_numpy(dtype=np.float32) / 100.0
    lfm_freq = np.log1p(tracks_df["lfm_count"].to_numpy(dtype=np.float32))
    lfm_freq /= max(float(lfm_freq.max()), 1.0)
    exposure = np.clip(0.5 * pop_raw + 0.5 * lfm_freq, 0.01, 0.99)
    novelty_arr = -np.log2(exposure)
    novelty_arr = (novelty_arr - novelty_arr.min()) / (novelty_arr.max() - novelty_arr.min() + 1e-6)

    models = [
        "Popularity",
        "Item-KNN CF",
        "BPR-MF",
        "NeuMF (Multimodal)",
        "Dual-Horizon Transformer",
        "Proposed Hybrid + Discovery",
    ]
    acc = {m: {"hr10": [], "ndcg10": [], "mrr10": [], "ild20": [], "nov20": [], "unseen_art": []} for m in models}
    covered_items: dict[str, set[int]] = {m: set() for m in models}

    eval_subset = test_cases[: min(600, len(test_cases))]
    L = transformer.max_seq_len

    for case in eval_subset:
        uid = case["user_idx"]
        target = case["target"]
        seen_pos = case["seen_pos"]
        seen_artists = {int(artist_ids[i]) for i in seen_pos}

        # Sample 499 negatives not equal to target
        negs = []
        while len(negs) < 499:
            cand = rng.integers(0, num_items, size=550)
            for c in cand:
                ci = int(c)
                if ci != target and ci not in seen_pos:
                    negs.append(ci)
                    if len(negs) == 499:
                        break
        eval_items = np.array([target] + negs, dtype=np.int64)  # target is index 0

        # Prepare history tensors
        pos_hist = [it for it, act in zip(case["hist_items"], case["hist_actions"]) if act >= 2]
        neg_hist = [it for it, act in zip(case["hist_items"], case["hist_actions"]) if act == 0]

        pad = L - len(case["hist_items"][-L:])
        it_seq = torch.tensor([[num_items] * pad + case["hist_items"][-L:]], dtype=torch.long, device=device)
        ac_seq = torch.tensor([[PAD_ACTION_ID] * pad + case["hist_actions"][-L:]], dtype=torch.long, device=device)
        tb_seq = torch.tensor([[PAD_TIME_BUCKET] * pad + case["hist_tb"][-L:]], dtype=torch.long, device=device)
        u_t = torch.tensor([uid], dtype=torch.long, device=device)
        eval_t = torch.tensor(eval_items, dtype=torch.long, device=device).unsqueeze(0)

        # Compute full-catalog scores for each model
        s_pop = item_cf.item_popularity.copy()
        s_cf = item_cf.score_for_history(pos_hist, neg_hist)
        s_cf_norm = s_cf / (float(s_cf.max()) + 1e-6)

        u_bpr = bpr_mf.user_emb(u_t).squeeze(0).cpu().numpy()
        s_bpr = (bpr_mf.item_emb.weight.detach().cpu().numpy() @ u_bpr) + bpr_mf.item_bias.weight.detach().cpu().numpy().ravel()

        neumf_eval_logits, _ = neumf.score_pairs(u_t, eval_t)
        u_long = neumf.get_long_term_user_emb(u_t)
        s_neumf_full = (item_embeddings @ u_long.squeeze(0).cpu().numpy()).astype(np.float32)

        u_short = transformer.encode_short_term_sequence(it_seq, ac_seq, tb_seq)[:, -1, :]
        u_fused, _ = transformer.fuse_horizons(u_long, u_short)
        u_fused_np = u_fused.squeeze(0).cpu().numpy()
        s_tr_full = (item_embeddings @ u_fused_np).astype(np.float32)

        # Proposed Hybrid: combines Dual-Horizon Transformer + Item-KNN CF + Novelty-aware MMR diversification
        s_hyb_acc = 0.68 * s_tr_full + 0.32 * s_cf_norm
        s_hyb_disc = 0.62 * s_tr_full + 0.24 * s_cf_norm + 0.14 * novelty_arr

        cand_scores_map = {
            "Popularity": s_pop[eval_items],
            "Item-KNN CF": s_cf_norm[eval_items],
            "BPR-MF": s_bpr[eval_items],
            "NeuMF (Multimodal)": (0.5 * torch.sigmoid(neumf_eval_logits).squeeze(0).cpu().numpy() + 0.5 * s_neumf_full[eval_items]),
            "Dual-Horizon Transformer": s_tr_full[eval_items],
            "Proposed Hybrid + Discovery": s_hyb_acc[eval_items],
        }

        full_scores_map = {
            "Popularity": s_pop,
            "Item-KNN CF": s_cf_norm,
            "BPR-MF": s_bpr,
            "NeuMF (Multimodal)": s_neumf_full,
            "Dual-Horizon Transformer": s_tr_full,
            "Proposed Hybrid + Discovery": s_hyb_disc,
        }

        for m in models:
            cs = cand_scores_map[m]
            # Rank of index 0 (higher score = better; break ties conservatively)
            rank = int(np.sum(cs[1:] >= cs[0])) + 1
            hr = 1.0 if rank <= 10 else 0.0
            ndcg = (1.0 / np.log2(rank + 1.0)) if rank <= 10 else 0.0
            mrr = (1.0 / rank) if rank <= 10 else 0.0
            acc[m]["hr10"].append(hr)
            acc[m]["ndcg10"].append(ndcg)
            acc[m]["mrr10"].append(mrr)

            # Top-20 unseen catalog recommendations for diversity/novelty/coverage
            fs = full_scores_map[m].copy()
            if seen_pos:
                fs[list(seen_pos)] = -1e9
            if m == "Proposed Hybrid + Discovery":
                # Fast MMR top-20 selection from top-120 candidates
                top120 = np.argpartition(fs, -120)[-120:]
                top120 = top120[np.argsort(-fs[top120])]
                s120 = fs[top120]
                s120 = (s120 - s120.min()) / (s120.max() - s120.min() + 1e-6)
                picked: list[int] = []
                avail = np.ones(len(top120), dtype=bool)
                e120 = item_embeddings[top120]
                for _ in range(20):
                    cpos = np.flatnonzero(avail)
                    if not picked:
                        util = s120[cpos]
                    else:
                        msim = (e120[cpos] @ item_embeddings[picked].T).max(axis=1)
                        util = 0.65 * s120[cpos] - 0.35 * msim
                    b = int(cpos[int(np.argmax(util))])
                    avail[b] = False
                    picked.append(int(top120[b]))
                top20 = np.array(picked, dtype=np.int32)
            else:
                top20 = np.argpartition(fs, -20)[-20:]
                top20 = top20[np.argsort(-fs[top20])]

            covered_items[m].update(top20.tolist())
            e20 = item_embeddings[top20]
            sim20 = np.clip(e20 @ e20.T, -1.0, 1.0)
            tri = np.triu(np.ones((20, 20), dtype=bool), k=1)
            acc[m]["ild20"].append(float(np.mean(1.0 - sim20[tri])))
            acc[m]["nov20"].append(float(np.mean(novelty_arr[top20])))
            acc[m]["unseen_art"].append(
                float(np.mean([int(artist_ids[i]) not in seen_artists for i in top20]))
            )

    summary: dict[str, dict[str, float]] = {}
    for m in models:
        summary[m] = {
            "HR@10": round(float(np.mean(acc[m]["hr10"])), 4),
            "NDCG@10": round(float(np.mean(acc[m]["ndcg10"])), 4),
            "MRR@10": round(float(np.mean(acc[m]["mrr10"])), 4),
            "ILD@20": round(float(np.mean(acc[m]["ild20"])), 4),
            "Novelty@20": round(float(np.mean(acc[m]["nov20"])), 4),
            "CatalogCoverage@20": round(len(covered_items[m]) / num_items, 4),
            "UnseenArtistRatio@20": round(float(np.mean(acc[m]["unseen_art"])), 4),
        }
        print(f"  {m:28s} -> {summary[m]}")
    return summary


def run_shuffle_benchmarks(
    playlists_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    neumf: NeuralCollaborativeFiltering,
    num_users: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Compare Fisher-Yates, Artist-Spaced, Greedy Relevance, and Proposed Adaptive Shuffle on 200 playlists."""
    print("\n--- Running Benchmark Suite B: Adaptive Shuffle vs. Baselines (200 Playlists) ---")
    engine = AdaptiveShuffleEngine()
    artist_ids = tracks_df["artist_idx"].to_numpy(dtype=np.int32)
    genre_ids = tracks_df["genre_idx"].to_numpy(dtype=np.int32)
    raw_audio = tracks_df[["danceability", "energy", "valence", "tempo", "key"]].to_numpy(dtype=np.float32)

    modes = {
        "Fisher-Yates Random": "fisher_yates",
        "Spotify-Style Artist Spaced": "artist_spaced",
        "Greedy Relevance Order": "greedy_relevance",
        "Proposed Adaptive Shuffle": "adaptive",
    }
    agg: dict[str, list[dict[str, float]]] = {k: [] for k in modes}

    sample_playlists = playlists_df[playlists_df["num_tracks"] >= 20].head(200)
    rng = np.random.default_rng(99)

    with torch.no_grad():
        all_u_long = neumf.get_long_term_user_emb(torch.arange(num_users, device=device)).cpu().numpy()

    for idx, row in enumerate(sample_playlists.itertuples(index=False)):
        pool = list(row.item_indices[:50])
        u_vec = all_u_long[idx % num_users]
        embs = item_embeddings[pool]
        arts = artist_ids[pool]
        gens = genre_ids[pool]
        aud = raw_audio[pool]
        rel = (embs @ u_vec).astype(np.float32)
        is_disc = rng.random(len(pool)) < 0.35

        for label, mode_key in modes.items():
            cfg = ShuffleConfig(mode=mode_key, seed=1000 + idx)
            res = engine.sequence(embs, arts, gens, aud, rel, is_disc, cfg)
            agg[label].append(res["metrics"])

    summary: dict[str, dict[str, float]] = {}
    metric_keys = [
        "artist_clump_rate",
        "min_artist_gap",
        "mean_artist_gap",
        "genre_switch_rate",
        "local_window_diversity",
        "acoustic_smoothness",
        "mean_bpm_jump",
        "session_ndcg",
        "discovery_pacing_std",
    ]
    for label, rows in agg.items():
        summary[label] = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in metric_keys}
        print(f"  {label:28s} -> {summary[label]}")
    return summary


def run_feedback_adaptation_benchmark(
    tracks_df: pd.DataFrame,
    item_embeddings: np.ndarray,
    neumf: NeuralCollaborativeFiltering,
    num_users: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Simulate 100 interactive sessions where a user skips 3 tracks from their dominant genre
    and likes 2 tracks from a new target discovery genre. Measure how rapidly top-20 recommendations
    suppress the fatigued genre and surface the newly liked genre.
    """
    print("\n--- Running Benchmark Suite C: Real-Time Feedback Adaptation Simulation ---")
    genre_ids = tracks_df["genre_idx"].to_numpy(dtype=np.int32)
    artist_ids = tracks_df["artist_idx"].to_numpy(dtype=np.int32)

    with torch.no_grad():
        all_u_long = neumf.get_long_term_user_emb(torch.arange(min(100, num_users), device=device)).cpu().numpy()

    before_fatigued_share, after_fatigued_share = [], []
    before_target_share, after_target_share = [], []
    rng = np.random.default_rng(77)

    unique_genres = np.unique(genre_ids)
    for uid in range(len(all_u_long)):
        u_base = all_u_long[uid]
        base_scores = item_embeddings @ u_base
        top20_before = np.argpartition(base_scores, -20)[-20:]
        top_genres, counts = np.unique(genre_ids[top20_before], return_counts=True)
        dom_genre = int(top_genres[int(np.argmax(counts))])

        other_genres = [int(g) for g in unique_genres if int(g) != dom_genre]
        target_genre = int(rng.choice(other_genres))

        before_fatigued_share.append(float(np.mean(genre_ids[top20_before] == dom_genre)))
        before_target_share.append(float(np.mean(genre_ids[top20_before] == target_genre)))

        fb = UserFeedbackState(user_idx=uid, embed_dim=128)
        dom_pool = np.flatnonzero(genre_ids == dom_genre)
        tgt_pool = np.flatnonzero(genre_ids == target_genre)

        # User early-skips 3 tracks of dom_genre and likes 2 + completes 1 of target_genre
        for it in rng.choice(dom_pool, size=min(3, len(dom_pool)), replace=False):
            fb.record_feedback(int(it), item_embeddings[it], int(artist_ids[it]), dom_genre, "early_skip", 0.10)
        picked_tgt = rng.choice(tgt_pool, size=min(3, len(tgt_pool)), replace=False)
        for idx_t, it in enumerate(picked_tgt):
            act = "like" if idx_t < 2 else "repeat"
            fb.record_feedback(int(it), item_embeddings[it], int(artist_ids[it]), target_genre, act, 1.0)

        u_adapted = u_base + fb.delta_u
        u_adapted /= np.linalg.norm(u_adapted) + 1e-8
        dyn = np.zeros(len(tracks_df), dtype=np.float32)
        for g_idx, pen in fb.genre_fatigue.items():
            dyn[genre_ids == g_idx] -= 0.30 * pen
        for g_idx, bst in fb.genre_boost.items():
            dyn[genre_ids == g_idx] += 0.12 * bst

        adapted_scores = (item_embeddings @ u_adapted) + 0.25 * fb.linucb_bonus(item_embeddings, 0.15) + dyn
        for skipped in fb.skipped_items:
            adapted_scores[skipped] = -1e9
        top20_after = np.argpartition(adapted_scores, -20)[-20:]

        after_fatigued_share.append(float(np.mean(genre_ids[top20_after] == dom_genre)))
        after_target_share.append(float(np.mean(genre_ids[top20_after] == target_genre)))

    res = {
        "Static_Before_Feedback": {
            "FatiguedGenreShare@20": round(float(np.mean(before_fatigued_share)), 4),
            "NewTargetGenreShare@20": round(float(np.mean(before_target_share)), 4),
        },
        "Adapted_After_Feedback": {
            "FatiguedGenreShare@20": round(float(np.mean(after_fatigued_share)), 4),
            "NewTargetGenreShare@20": round(float(np.mean(after_target_share)), 4),
        },
    }
    print(f"  Feedback Adaptation -> {res}")
    return res


def save_benchmark_charts(
    rec_res: dict[str, dict[str, float]],
    shuf_res: dict[str, dict[str, float]],
    reports_dir: Path,
) -> None:
    """Generate clean dark-themed benchmark figures for documentation & UI."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    # Chart 1: Recommendation Accuracy vs Diversity/Novelty
    models = list(rec_res.keys())
    ndcg = [rec_res[m]["NDCG@10"] for m in models]
    ild = [rec_res[m]["ILD@20"] for m in models]
    nov = [rec_res[m]["Novelty@20"] for m in models]

    x = np.arange(len(models))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    ax.bar(x - width, ndcg, width, label="NDCG@10 (Relevance)", color="#1DB954")
    ax.bar(x, ild, width, label="ILD@20 (Diversity)", color="#38BDF8")
    ax.bar(x + width, nov, width, label="Novelty@20 (Discovery)", color="#A855F7")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=18, ha="right", fontsize=9)
    ax.set_title("Recommendation Models: Relevance, Diversity & Discovery Comparison", fontsize=12, pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(reports_dir / "recommendation_benchmarks.png")
    plt.close(fig)

    # Chart 2: Shuffle Comparison
    s_modes = list(shuf_res.keys())
    clumps = [shuf_res[m]["artist_clump_rate"] * 100 for m in s_modes]
    smooth = [shuf_res[m]["acoustic_smoothness"] * 100 for m in s_modes]
    gswitch = [shuf_res[m]["genre_switch_rate"] * 100 for m in s_modes]

    xs = np.arange(len(s_modes))
    fig2, ax2 = plt.subplots(figsize=(10, 4.8), dpi=160)
    ax2.bar(xs - width, clumps, width, label="Artist Clump Rate % (Lower = Better)", color="#F43F5E")
    ax2.bar(xs, smooth, width, label="Acoustic Flow Smoothness % (Higher = Better)", color="#1DB954")
    ax2.bar(xs + width, gswitch, width, label="Genre Switch Rate % (Higher = Better)", color="#FBBF24")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(s_modes, rotation=12, ha="right", fontsize=9.5)
    ax2.set_title("Adaptive Shuffle vs. Traditional Baselines (200 Playlists)", fontsize=12, pad=12)
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(reports_dir / "shuffle_benchmarks.png")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all ML/DL models and run benchmark suites.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--epochs-neumf", type=int, default=12)
    parser.add_argument("--epochs-transformer", type=int, default=12)
    args = parser.parse_args()

    t_start = time.time()
    set_seed(42)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"=== Training & Evaluation Device: {device} ({gpu_name}) ===")

    tracks_df = pd.read_parquet(args.processed_dir / "tracks.parquet").sort_values("item_idx").reset_index(drop=True)
    interactions_df = pd.read_parquet(args.processed_dir / "interactions.parquet")
    playlists_df = pd.read_parquet(args.processed_dir / "playlists.parquet")
    stats = json.loads((args.processed_dir / "dataset_stats.json").read_text(encoding="utf-8"))

    num_items = len(tracks_df)
    num_artists = int(tracks_df["artist_idx"].max()) + 1
    num_genres = int(tracks_df["genre_idx"].max()) + 1
    num_users = int(interactions_df["user_idx"].max()) + 1

    splits = build_chronological_splits(interactions_df)
    print(
        f"Chronological splits built: {len(splits['train_users']):,} unique train (u,i) pairs | "
        f"{len(splits['seq_train_samples']):,} sequence windows | {len(splits['test_cases']):,} test users"
    )

    # 1. Fit Item-KNN Co-occurrence CF (User History + MPD Playlists)
    print("\n--- Fitting Baseline: Item-KNN Co-occurrence CF ---")
    item_cf = ItemCooccurrenceCF(num_items=num_items, top_k_neighbors=80).fit(
        user_item_pairs=(splits["train_users"], splits["train_items"], splits["train_weights"]),
        playlist_sequences=playlists_df["item_indices"].tolist(),
    )
    with open(args.artifacts_dir / "item_cf.pkl", "wb") as f:
        pickle.dump(item_cf, f)

    # 2. Train BPR Matrix Factorization
    pair_ds = InteractionPairDataset(
        users=splits["train_users"],
        pos_items=splits["train_items"],
        weights=splits["train_weights"],
        num_items=num_items,
        num_negatives=4,
    )
    bpr_mf = train_bpr_mf(num_users, num_items, pair_ds, device=device, epochs=10)
    torch.save(bpr_mf.state_dict(), args.artifacts_dir / "bpr_mf_checkpoint.pt")

    # 3. Initialize MultimodalTrackEncoder & Train NeuMF
    norm_cols = stats["norm_audio_cols"]
    track_encoder = MultimodalTrackEncoder(
        num_items=num_items,
        num_artists=num_artists,
        num_genres=num_genres,
        audio_dim=len(norm_cols),
        embed_dim=128,
        id_dropout_prob=0.15,
    )
    track_encoder.set_catalog_metadata(
        artist_indices=torch.tensor(tracks_df["artist_idx"].to_numpy(dtype=np.int64)),
        genre_indices=torch.tensor(tracks_df["genre_idx"].to_numpy(dtype=np.int64)),
        audio_features=torch.tensor(tracks_df[norm_cols].to_numpy(dtype=np.float32)),
    )
    pl_pairs = extract_playlist_adjacent_pairs(playlists_df)
    neumf = train_neumf_and_encoder(
        num_users=num_users,
        track_encoder=track_encoder,
        pair_ds=pair_ds,
        pl_pairs=pl_pairs,
        device=device,
        epochs=args.epochs_neumf,
    )

    # 4. Train Dual-Horizon Session Transformer
    transformer = train_dual_horizon_transformer(
        neumf=neumf,
        track_encoder=track_encoder,
        seq_samples=splits["seq_train_samples"],
        num_items=num_items,
        device=device,
        epochs=args.epochs_transformer,
    )

    # Save checkpoints AFTER joint fine-tuning so encoder weights match both NeuMF and Transformer
    torch.save(neumf.state_dict(), args.artifacts_dir / "neumf_checkpoint.pt")
    torch.save(transformer.state_dict(), args.artifacts_dir / "transformer_checkpoint.pt")

    item_embeddings = track_encoder.export_all_embeddings(device=device).numpy().astype(np.float32)
    np.save(args.artifacts_dir / "track_embeddings.npy", item_embeddings)
    print(f"Exported shared track embeddings matrix: {item_embeddings.shape}")

    # 5. Run Full Benchmark Suites A, B, C
    rec_benchmarks = run_recommendation_benchmarks(
        test_cases=splits["test_cases"],
        num_items=num_items,
        tracks_df=tracks_df,
        item_embeddings=item_embeddings,
        item_cf=item_cf,
        bpr_mf=bpr_mf,
        neumf=neumf,
        transformer=transformer,
        device=device,
    )
    shuffle_benchmarks = run_shuffle_benchmarks(
        playlists_df=playlists_df,
        tracks_df=tracks_df,
        item_embeddings=item_embeddings,
        neumf=neumf,
        num_users=num_users,
        device=device,
    )
    feedback_benchmarks = run_feedback_adaptation_benchmark(
        tracks_df=tracks_df,
        item_embeddings=item_embeddings,
        neumf=neumf,
        num_users=num_users,
        device=device,
    )

    save_benchmark_charts(rec_benchmarks, shuffle_benchmarks, args.reports_dir)

    full_report = {
        "device": gpu_name,
        "dataset_stats": stats,
        "recommendation_benchmarks": rec_benchmarks,
        "shuffle_benchmarks": shuffle_benchmarks,
        "feedback_adaptation_benchmarks": feedback_benchmarks,
        "total_runtime_seconds": round(time.time() - t_start, 2),
    }
    report_path = args.reports_dir / "evaluation_results.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\n=== All Models Trained & Evaluated in {full_report['total_runtime_seconds']}s! ===")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
