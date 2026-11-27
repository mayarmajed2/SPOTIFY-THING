"""
SQLite Persistence Layer for Per-User Online Feedback & LinUCB Bandit State.

Designed for single-instance deployment with zero external infrastructure:
- Uses SQLite WAL mode + threading.RLock + IMMEDIATE transactions for safe concurrent updates
- Enforces strict per-user isolation keyed on user_idx
- Survives FastAPI process restarts cleanly

For multi-instance production deployments, replace this store with a Redis hash/JSON
store (with optimistic locking via WATCH/MULTI or Lua scripts) or PostgreSQL JSONB
with row-level locking (`SELECT ... FOR UPDATE`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from backend.app.engine.online_feedback import UserFeedbackState


class SQLiteFeedbackStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_feedback_state (
                            user_idx INTEGER PRIMARY KEY,
                            state_json TEXT NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
            finally:
                conn.close()

    def load_user_state(self, user_idx: int, embed_dim: int = 128) -> UserFeedbackState:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT state_json FROM user_feedback_state WHERE user_idx = ?",
                    (int(user_idx),),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            return UserFeedbackState(user_idx=int(user_idx), embed_dim=embed_dim)
        return UserFeedbackState.from_dict(json.loads(row[0]))

    def save_user_state(self, state: UserFeedbackState) -> None:
        payload = json.dumps(state.to_dict())
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """
                    INSERT INTO user_feedback_state (user_idx, state_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_idx) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at
                    """,
                    (int(state.user_idx), payload, now),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def reset_user_state(self, user_idx: int) -> None:
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        "DELETE FROM user_feedback_state WHERE user_idx = ?",
                        (int(user_idx),),
                    )
            finally:
                conn.close()

    def list_persisted_users(self) -> list[int]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT user_idx FROM user_feedback_state ORDER BY user_idx"
                ).fetchall()
            finally:
                conn.close()
        return [int(r[0]) for r in rows]
