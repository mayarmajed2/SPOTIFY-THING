"""
SQLite Persistence Layer for Chrome Companion Extension Users, Sessions, Preferences & Spotify Capability State.

Security & Privacy Design:
- Stores ONLY SHA-256 hashes of extension session tokens (raw bearer tokens are never persisted to disk).
- Never stores Spotify OAuth access tokens or refresh tokens on the backend; only permitted capability/scope metadata is stored.
- Assigns each extension user a unique `feedback_slot_idx` (>= 1_000_000) so their LinUCB bandit & fatigue memory
  in `SQLiteFeedbackStore` is 100% isolated from catalog personas and all other extension users.
- Thread-safe WAL mode + `threading.RLock` + `BEGIN IMMEDIATE` transactions.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_SHUFFLE_PRESETS: dict[str, Any] = {
    "artist_repetition": "balanced",   # strict | balanced | flexible
    "genre_diversity": "medium",       # low | medium | high
    "musical_coherence": "smooth",     # smooth | balanced | variety
    "discovery_mode": "balanced",      # familiar | balanced | exploratory
    "randomness": "balanced",          # predictable | balanced | varied
    "quota_urgency_weight": 0.0,
}


def default_preferences() -> dict[str, Any]:
    return {
        "onboarding_completed": False,
        "operating_mode": "independent",  # independent | spotify_connected
        "preferred_genres": [],
        "preferred_artists": [],
        "discovery_intensity": 0.40,
        "diversity_level": 0.50,
        "novelty_boost": 0.30,
        "short_term_weight": None,
        "shuffle_presets": dict(DEFAULT_SHUFFLE_PRESETS),
        "dismissed_item_indices": [],
    }


def default_spotify_state() -> dict[str, Any]:
    return {
        "connected": False,
        "mode": "independent",
        "account_label": None,
        "product_tier": None,
        "granted_scopes": [],
        "capabilities": {
            "can_read_profile": False,
            "can_read_playlists": False,
            "can_read_playback_state": False,
            "can_create_playlists": False,
            "direct_queue_reorder_supported": False,
            "native_shuffle_override_supported": False,
        },
        "last_verified_at": None,
    }


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


class ExtensionUserStore:
    FEEDBACK_SLOT_OFFSET = 1_000_000

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
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
                        CREATE TABLE IF NOT EXISTS extension_users (
                            seq_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT NOT NULL UNIQUE,
                            token_hash TEXT NOT NULL UNIQUE,
                            display_name TEXT NOT NULL,
                            anchor_user_idx INTEGER NOT NULL DEFAULT 0,
                            preferences_json TEXT NOT NULL,
                            spotify_json TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_ext_users_token ON extension_users(token_hash);"
                    )
            finally:
                conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        prefs = default_preferences()
        stored_prefs = json.loads(row["preferences_json"])
        prefs.update(stored_prefs)
        merged_shuffle = dict(DEFAULT_SHUFFLE_PRESETS)
        merged_shuffle.update(stored_prefs.get("shuffle_presets") or {})
        prefs["shuffle_presets"] = merged_shuffle

        sp = default_spotify_state()
        stored_sp = json.loads(row["spotify_json"])
        sp.update(stored_sp)
        caps = dict(default_spotify_state()["capabilities"])
        caps.update(stored_sp.get("capabilities") or {})
        sp["capabilities"] = caps

        return {
            "seq_id": int(row["seq_id"]),
            "user_id": str(row["user_id"]),
            "display_name": str(row["display_name"]),
            "feedback_slot_idx": self.FEEDBACK_SLOT_OFFSET + int(row["seq_id"]),
            "anchor_user_idx": int(row["anchor_user_idx"]),
            "preferences": prefs,
            "spotify": sp,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def create_session(
        self,
        display_name: str = "Spotify Companion Listener",
        anchor_user_idx: int = 0,
        initial_preferences: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        raw_token = f"st_ext_{secrets.token_urlsafe(32)}"
        token_hash = _hash_token(raw_token)
        user_id = f"usr_{uuid.uuid4().hex[:16]}"
        now = time.time()

        prefs = default_preferences()
        if initial_preferences:
            for k, v in initial_preferences.items():
                if v is not None and k in prefs:
                    prefs[k] = v
        sp = default_spotify_state()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                cur = conn.execute(
                    """
                    INSERT INTO extension_users (
                        user_id, token_hash, display_name, anchor_user_idx,
                        preferences_json, spotify_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        token_hash,
                        display_name.strip()[:80] or "Spotify Companion Listener",
                        int(anchor_user_idx),
                        json.dumps(prefs),
                        json.dumps(sp),
                        now,
                        now,
                    ),
                )
                row_id = cur.lastrowid
                row = conn.execute(
                    "SELECT * FROM extension_users WHERE seq_id = ?", (row_id,)
                ).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

        return raw_token, self._row_to_record(row)

    def get_by_token(self, raw_token: str) -> dict[str, Any] | None:
        if not raw_token or len(raw_token) < 16:
            return None
        token_hash = _hash_token(raw_token.strip())
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM extension_users WHERE token_hash = ?",
                    (token_hash,),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return self._row_to_record(row)

    def update_preferences(
        self,
        user_id: str,
        updates: dict[str, Any],
        anchor_user_idx: int | None = None,
        display_name: str | None = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM extension_users WHERE user_id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Extension user {user_id} not found.")
                rec = self._row_to_record(row)
                prefs = rec["preferences"]

                for key, val in updates.items():
                    if val is None and key != "short_term_weight":
                        continue
                    if key == "shuffle_presets" and isinstance(val, dict):
                        prefs["shuffle_presets"].update({k: v for k, v in val.items() if v is not None})
                    elif key in prefs:
                        prefs[key] = val

                new_anchor = int(anchor_user_idx) if anchor_user_idx is not None else rec["anchor_user_idx"]
                new_name = (display_name.strip()[:80] if display_name else rec["display_name"]) or rec["display_name"]

                conn.execute(
                    """
                    UPDATE extension_users
                    SET preferences_json = ?, anchor_user_idx = ?, display_name = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(prefs), new_anchor, new_name, now, user_id),
                )
                updated_row = conn.execute(
                    "SELECT * FROM extension_users WHERE user_id = ?", (user_id,)
                ).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return self._row_to_record(updated_row)

    def update_spotify_state(self, user_id: str, spotify_updates: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT * FROM extension_users WHERE user_id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"Extension user {user_id} not found.")
                rec = self._row_to_record(row)
                sp = rec["spotify"]
                prefs = rec["preferences"]

                for k, v in spotify_updates.items():
                    if k == "capabilities" and isinstance(v, dict):
                        sp["capabilities"].update(v)
                    elif k in sp:
                        sp[k] = v
                sp["last_verified_at"] = now
                sp["mode"] = "spotify_connected" if sp.get("connected") else "independent"
                prefs["operating_mode"] = sp["mode"]

                conn.execute(
                    """
                    UPDATE extension_users
                    SET spotify_json = ?, preferences_json = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(sp), json.dumps(prefs), now, user_id),
                )
                updated_row = conn.execute(
                    "SELECT * FROM extension_users WHERE user_id = ?", (user_id,)
                ).fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return self._row_to_record(updated_row)

    def delete_user(self, user_id: str) -> int | None:
        """Delete user session, preferences, and Spotify state; returns feedback_slot_idx for cleanup."""
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT seq_id FROM extension_users WHERE user_id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                slot_idx = self.FEEDBACK_SLOT_OFFSET + int(row["seq_id"])
                conn.execute("DELETE FROM extension_users WHERE user_id = ?", (user_id,))
                conn.commit()
                return slot_idx
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
