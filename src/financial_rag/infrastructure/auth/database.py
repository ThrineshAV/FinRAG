"""SQLite database for JWT auth users."""
from __future__ import annotations
import sqlite3
import os
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_DB_PATH = Path(os.getenv("AUTH_DB_PATH", "data/auth_users.db"))


def _connect() -> sqlite3.Connection:
    """Open SQLite connection with row factory for dict-like access."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create users table if it does not exist."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'reader',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def seed_admin_from_env() -> None:
    """If ADMIN_EMAIL and ADMIN_PASSWORD are set and no admin exists,
    create the admin user using bcrypt."""
    email = os.getenv("ADMIN_EMAIL")
    password = os.getenv("ADMIN_PASSWORD")
    if not email or not password:
        return
    init_db()
    from passlib.hash import bcrypt
    conn = _connect()
    existing = conn.execute(
        "SELECT 1 FROM users WHERE email = ?", (email,)
    ).fetchone()
    if existing:
        conn.close()
        return
    conn.execute(
        "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (email, bcrypt.hash(password), "admin", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    logger.info("Seeded admin user from env: %s", email)


def get_user_by_email(email: str) -> sqlite3.Row | None:
    init_db()
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return row


def create_user(email: str, password: str, role: str = "reader") -> int:
    from passlib.hash import bcrypt
    init_db()
    conn = _connect()
    cursor = conn.execute(
        "INSERT INTO users (email, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
        (email, bcrypt.hash(password), role, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id
