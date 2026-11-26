"""
Online Feedback Learner (LinUCB Contextual Bandit + Dynamic Fatigue Memory).

Enables sub-millisecond adaptation to implicit and explicit user feedback:
- Explicit: like (+1.0), dislike (-1.0)
- Implicit: repeat (+0.85), complete_play >=80% (+0.55), partial_play (+0.15), early_skip <25% (-0.75)

Maintains:
1. Diagonal/Ridge LinUCB posterior (A_diag, b) in R^128 for uncertainty-aware exploration + exploitation
2. Signed EMA feedback residual vector (delta_u) for shifting the fused user embedding immediately
3. Artist & Genre short-term fatigue / suppression counters triggered by early skips or dislikes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ACTION_TO_ID = {
    "early_skip": 0,
    "partial_play": 1,
    "complete_play": 2,
    "repeat": 3,
    "like": 4,
    "dislike": 5,
}

ACTION_DEFAULT_REWARD = {
    "early_skip": -0.75,
    "partial_play": 0.15,
    "complete_play": 0.55,
    "repeat": 0.85,
    "like": 1.00,
    "dislike": -1.00,
}


@dataclass
class FeedbackEvent:
    item_idx: int
    action: str
    action_id: int
    played_ratio: float
    reward: float
    artist_idx: int
    genre_idx: int
    timestamp: float = field(default_factory=time.time)


class UserFeedbackState:
    """Mutable real-time personalization state for an active listener."""

    def __init__(self, user_idx: int, embed_dim: int = 128, ridge_lambda: float = 2.0) -> None:
        self.user_idx = user_idx
        self.embed_dim = embed_dim
        self.A_diag = np.full(embed_dim, ridge_lambda, dtype=np.float32)
        self.b = np.zeros(embed_dim, dtype=np.float32)
        self.delta_u = np.zeros(embed_dim, dtype=np.float32)

        self.liked_items: set[int] = set()
        self.disliked_items: set[int] = set()
        self.skipped_items: set[int] = set()
        self.played_items_recent: list[int] = []

        # Dynamic negative fatigue & positive affinity counters per artist / genre
        self.artist_fatigue: dict[int, float] = {}
        self.genre_fatigue: dict[int, float] = {}
        self.artist_boost: dict[int, float] = {}
        self.genre_boost: dict[int, float] = {}

        self.events: list[FeedbackEvent] = []

    def record_feedback(
        self,
        item_idx: int,
        item_emb: np.ndarray,
        artist_idx: int,
        genre_idx: int,
        action: str,
        played_ratio: float = 1.0,
    ) -> FeedbackEvent:
        action = action.lower().strip()
        if action not in ACTION_TO_ID:
            # Infer implicit action from played_ratio if generic "listen" passed
            if played_ratio < 0.25:
                action = "early_skip"
            elif played_ratio < 0.75:
                action = "partial_play"
            else:
                action = "complete_play"

        base_reward = ACTION_DEFAULT_REWARD[action]
        # Modulate implicit reward smoothly by actual listening duration ratio
        if action in ("partial_play", "complete_play"):
            reward = float(np.clip(-0.4 + 1.05 * min(played_ratio, 1.1), -0.75, 0.75))
        else:
            reward = base_reward

        # Decay existing short-term fatigue slightly at every step
        for k in list(self.artist_fatigue.keys()):
            self.artist_fatigue[k] *= 0.88
            if self.artist_fatigue[k] < 0.05:
                del self.artist_fatigue[k]
        for k in list(self.genre_fatigue.keys()):
            self.genre_fatigue[k] *= 0.90
            if self.genre_fatigue[k] < 0.05:
                del self.genre_fatigue[k]

        # Diagonal LinUCB Ridge update: A <- A + x^2, b <- b + r * x
        x = item_emb.astype(np.float32)
        self.A_diag += x * x
        self.b += reward * x

        # Signed EMA residual shift on user taste vector
        step_lr = 0.28 if abs(reward) >= 0.7 else 0.14
        self.delta_u = 0.82 * self.delta_u + step_lr * reward * x
        norm = float(np.linalg.norm(self.delta_u))
        if norm > 0.65:
            self.delta_u = self.delta_u * (0.65 / norm)

        if action == "like":
            self.liked_items.add(item_idx)
            self.disliked_items.discard(item_idx)
            self.artist_boost[artist_idx] = min(1.5, self.artist_boost.get(artist_idx, 0.0) + 0.45)
            self.genre_boost[genre_idx] = min(1.2, self.genre_boost.get(genre_idx, 0.0) + 0.30)
        elif action == "repeat":
            self.artist_boost[artist_idx] = min(1.5, self.artist_boost.get(artist_idx, 0.0) + 0.35)
            self.genre_boost[genre_idx] = min(1.2, self.genre_boost.get(genre_idx, 0.0) + 0.25)
        elif action == "dislike":
            self.disliked_items.add(item_idx)
            self.liked_items.discard(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.90
            self.genre_fatigue[genre_idx] = self.genre_fatigue.get(genre_idx, 0.0) + 0.50
        elif action == "early_skip":
            self.skipped_items.add(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.55
            self.genre_fatigue[genre_idx] = self.genre_fatigue.get(genre_idx, 0.0) + 0.35

        self.played_items_recent.append(item_idx)
        if len(self.played_items_recent) > 100:
            self.played_items_recent = self.played_items_recent[-100:]

        ev = FeedbackEvent(
            item_idx=item_idx,
            action=action,
            action_id=ACTION_TO_ID[action],
            played_ratio=float(played_ratio),
            reward=float(reward),
            artist_idx=artist_idx,
            genre_idx=genre_idx,
        )
        self.events.append(ev)
        return ev

    def linucb_bonus(self, candidate_embs: np.ndarray, alpha_explore: float = 0.15) -> np.ndarray:
        """Compute contextual bandit mean + UCB exploration bonus for [M, D] candidates."""
        inv_A = 1.0 / self.A_diag  # [D]
        theta_hat = self.b * inv_A  # [D]
        mean_reward = candidate_embs @ theta_hat  # [M]
        if alpha_explore <= 0.0:
            return mean_reward
        # Variance x^T A^{-1} x under diagonal approximation
        var = (candidate_embs * candidate_embs) @ inv_A
        ucb = alpha_explore * np.sqrt(np.maximum(var, 1e-8))
        return mean_reward + ucb

    def summary(self) -> dict[str, Any]:
        return {
            "user_idx": self.user_idx,
            "num_feedback_events": len(self.events),
            "liked_count": len(self.liked_items),
            "disliked_count": len(self.disliked_items),
            "skipped_count": len(self.skipped_items),
            "residual_magnitude": round(float(np.linalg.norm(self.delta_u)), 4),
            "fatigued_artists": len(self.artist_fatigue),
            "fatigued_genres": len(self.genre_fatigue),
        }
