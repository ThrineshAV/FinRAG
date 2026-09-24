"""Refresh token utilities — opaque tokens stored server-side for revocation."""
from __future__ import annotations
import os
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("AUTH_DB_PATH", "data/auth_users.db"))
REFRESH_TOKEN_EXPIRE_DAYS = 7


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_refresh_table() -> None:
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            revoked INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def create_refresh_token(user_id: int) -> str:
    """Generate, persist, and return a new refresh token."""
    init_refresh_table()
    token = secrets.token_urlsafe(48)
    expires_at = (datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)).isoformat()
    conn = _connect()
    conn.execute(
        "INSERT INTO refresh_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires_at),
    )
    conn.commit()
    conn.close()
    return token


def verify_refresh_token(token: str) -> int | None:
    """Return user_id if token is valid, non-revoked, and non-expired; else None."""
    if not token:
        return None
    init_refresh_table()
    conn = _connect()
    row = conn.execute(
        "SELECT user_id, expires_at, revoked FROM refresh_tokens WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    if row is None or row["revoked"]:
        return None
    if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
        return None
    return row["user_id"]


def revoke_refresh_token(token: str) -> bool:
    """Mark a refresh token as revoked. Returns True if a row was updated."""
    if not token:
        return False
    init_refresh_table()
    conn = _connect()
    cursor = conn.execute(
        "UPDATE refresh_tokens SET revoked = 1 WHERE token = ?", (token,)
    )
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated
