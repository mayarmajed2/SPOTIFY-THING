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

# Maps rich feedback actions onto the 0..5 action vocabulary expected by DualHorizonSessionTransformer
ACTION_TO_ID = {
    "early_skip": 0,
    "partial_play": 1,
    "complete_play": 2,
    "repeat": 3,
    "like": 4,
    "dislike": 5,
    "not_interested": 5,
    "more_like_this": 4,
    "less_like_this": 0,
    "artist_fatigue": 0,
    "genre_fatigue": 0,
}

ACTION_DEFAULT_REWARD = {
    "early_skip": -0.75,
    "partial_play": 0.15,
    "complete_play": 0.55,
    "repeat": 0.85,
    "like": 1.00,
    "dislike": -1.00,
    "not_interested": -0.45,
    "more_like_this": 0.90,
    "less_like_this": -0.55,
    "artist_fatigue": -0.30,
    "genre_fatigue": -0.30,
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
        self.ridge_lambda = float(ridge_lambda)
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
        self.genre_skip_counts: dict[int, int] = {}
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

        # Decay existing short-term fatigue smoothly at every step so temporary fatigue recovers
        for k in list(self.artist_fatigue.keys()):
            self.artist_fatigue[k] *= 0.80
            if self.artist_fatigue[k] < 0.05:
                del self.artist_fatigue[k]
        for k in list(self.genre_fatigue.keys()):
            self.genre_fatigue[k] *= 0.78
            if self.genre_fatigue[k] < 0.05:
                del self.genre_fatigue[k]

        # Diagonal LinUCB Ridge update: A <- A + x^2, b <- b + r * x
        x = item_emb.astype(np.float32)
        self.A_diag += x * x
        self.b += reward * x

        # Calibrated EMA residual shift: gentler on single negative actions so one negative click never shocks long-term taste
        if action in ("artist_fatigue", "genre_fatigue"):
            step_lr = 0.05
        elif action in ("not_interested", "less_like_this"):
            step_lr = 0.10
        elif action == "early_skip":
            prev_skips = self.genre_skip_counts.get(genre_idx, 0)
            step_lr = 0.10 if prev_skips == 0 else 0.22
        elif action == "dislike":
            step_lr = 0.18
        elif abs(reward) >= 0.7:
            step_lr = 0.26
        else:
            step_lr = 0.14
        self.delta_u = 0.82 * self.delta_u + step_lr * reward * x
        norm = float(np.linalg.norm(self.delta_u))
        if norm > 0.65:
            self.delta_u = self.delta_u * (0.65 / norm)

        if action == "like":
            self.liked_items.add(item_idx)
            self.disliked_items.discard(item_idx)
            self.genre_skip_counts.pop(genre_idx, None)
            self.artist_boost[artist_idx] = min(1.5, self.artist_boost.get(artist_idx, 0.0) + 0.45)
            self.genre_boost[genre_idx] = min(1.2, self.genre_boost.get(genre_idx, 0.0) + 0.30)
        elif action == "more_like_this":
            self.disliked_items.discard(item_idx)
            self.genre_skip_counts.pop(genre_idx, None)
            self.artist_boost[artist_idx] = min(1.5, self.artist_boost.get(artist_idx, 0.0) + 0.40)
            self.genre_boost[genre_idx] = min(1.2, self.genre_boost.get(genre_idx, 0.0) + 0.35)
        elif action == "repeat":
            self.genre_skip_counts.pop(genre_idx, None)
            self.artist_boost[artist_idx] = min(1.5, self.artist_boost.get(artist_idx, 0.0) + 0.35)
            self.genre_boost[genre_idx] = min(1.2, self.genre_boost.get(genre_idx, 0.0) + 0.25)
        elif action == "dislike":
            self.disliked_items.add(item_idx)
            self.liked_items.discard(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.90
            self.genre_fatigue[genre_idx] = self.genre_fatigue.get(genre_idx, 0.0) + 0.40
        elif action == "not_interested":
            self.disliked_items.add(item_idx)
            self.liked_items.discard(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.25
        elif action == "less_like_this":
            self.disliked_items.add(item_idx)
            self.liked_items.discard(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.45
            skip_cnt = self.genre_skip_counts.get(genre_idx, 0) + 1
            self.genre_skip_counts[genre_idx] = skip_cnt
            if skip_cnt >= 2:
                self.genre_fatigue[genre_idx] = self.genre_fatigue.get(genre_idx, 0.0) + 0.30
        elif action == "artist_fatigue":
            self.artist_fatigue[artist_idx] = min(2.0, self.artist_fatigue.get(artist_idx, 0.0) + 0.85)
        elif action == "genre_fatigue":
            self.genre_fatigue[genre_idx] = min(2.0, self.genre_fatigue.get(genre_idx, 0.0) + 0.75)
        elif action == "early_skip":
            self.skipped_items.add(item_idx)
            self.artist_fatigue[artist_idx] = self.artist_fatigue.get(artist_idx, 0.0) + 0.45
            skip_cnt = self.genre_skip_counts.get(genre_idx, 0) + 1
            self.genre_skip_counts[genre_idx] = skip_cnt
            # Require >= 2 skips of the same genre before activating genre-wide fatigue
            if skip_cnt >= 2:
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete per-user bandit and feedback memory for persistence."""
        return {
            "user_idx": int(self.user_idx),
            "embed_dim": int(self.embed_dim),
            "ridge_lambda": float(self.ridge_lambda),
            "A_diag": self.A_diag.astype(float).tolist(),
            "b": self.b.astype(float).tolist(),
            "delta_u": self.delta_u.astype(float).tolist(),
            "liked_items": sorted(int(i) for i in self.liked_items),
            "disliked_items": sorted(int(i) for i in self.disliked_items),
            "skipped_items": sorted(int(i) for i in self.skipped_items),
            "played_items_recent": [int(i) for i in self.played_items_recent[-100:]],
            "artist_fatigue": {str(k): float(v) for k, v in self.artist_fatigue.items()},
            "genre_fatigue": {str(k): float(v) for k, v in self.genre_fatigue.items()},
            "genre_skip_counts": {str(k): int(v) for k, v in self.genre_skip_counts.items()},
            "artist_boost": {str(k): float(v) for k, v in self.artist_boost.items()},
            "genre_boost": {str(k): float(v) for k, v in self.genre_boost.items()},
            "events": [
                {
                    "item_idx": int(ev.item_idx),
                    "action": ev.action,
                    "action_id": int(ev.action_id),
                    "played_ratio": float(ev.played_ratio),
                    "reward": float(ev.reward),
                    "artist_idx": int(ev.artist_idx),
                    "genre_idx": int(ev.genre_idx),
                    "timestamp": float(ev.timestamp),
                }
                for ev in self.events[-200:]
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserFeedbackState":
        """Restore a persisted UserFeedbackState from a serialized dictionary."""
        state = cls(
            user_idx=int(data["user_idx"]),
            embed_dim=int(data.get("embed_dim", 128)),
            ridge_lambda=float(data.get("ridge_lambda", 2.0)),
        )
        state.A_diag = np.asarray(data["A_diag"], dtype=np.float32)
        state.b = np.asarray(data["b"], dtype=np.float32)
        state.delta_u = np.asarray(data["delta_u"], dtype=np.float32)
        state.liked_items = {int(i) for i in data.get("liked_items", [])}
        state.disliked_items = {int(i) for i in data.get("disliked_items", [])}
        state.skipped_items = {int(i) for i in data.get("skipped_items", [])}
        state.played_items_recent = [int(i) for i in data.get("played_items_recent", [])]
        state.artist_fatigue = {int(k): float(v) for k, v in data.get("artist_fatigue", {}).items()}
        state.genre_fatigue = {int(k): float(v) for k, v in data.get("genre_fatigue", {}).items()}
        state.genre_skip_counts = {int(k): int(v) for k, v in data.get("genre_skip_counts", {}).items()}
        state.artist_boost = {int(k): float(v) for k, v in data.get("artist_boost", {}).items()}
        state.genre_boost = {int(k): float(v) for k, v in data.get("genre_boost", {}).items()}
        state.events = [
            FeedbackEvent(
                item_idx=int(ev["item_idx"]),
                action=str(ev["action"]),
                action_id=int(ev["action_id"]),
                played_ratio=float(ev["played_ratio"]),
                reward=float(ev["reward"]),
                artist_idx=int(ev["artist_idx"]),
                genre_idx=int(ev["genre_idx"]),
                timestamp=float(ev["timestamp"]),
            )
            for ev in data.get("events", [])
        ]
        return state
