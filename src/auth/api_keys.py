"""API key management — generation, hashing, validation, and file-based storage.

Keys are stored as SHA-256 hashes in a JSON file.  The raw key is returned
exactly once when created and is never persisted.

Storage location defaults to ``data/api_keys.json`` and is configurable via
the ``API_KEYS_FILE`` environment variable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from pathlib import Path

from src.auth.models import APIKeyCreate, APIKeyRecord, APIKeyResponse, APIKeyInfo, Role

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_KEY_PREFIX = "fsr_"  # FinSight-RAG key prefix for easy identification
_KEY_LENGTH = 32  # bytes of randomness (produces 64 hex chars)
_DEFAULT_KEYS_FILE = "data/api_keys.json"

def _keys_path() -> Path:
    """Resolve the API key storage file path."""
    return Path(os.getenv("API_KEYS_FILE", _DEFAULT_KEYS_FILE))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of *raw_key*."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _load_records() -> list[APIKeyRecord]:
    """Load all key records from the JSON file."""
    path = _keys_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [APIKeyRecord(**record) for record in data]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Failed to load API key store at %s: %s", path, exc)
        return []


def _save_records(records: list[APIKeyRecord]) -> None:
    """Persist key records to the JSON file."""
    path = _keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([record.model_dump() for record in records], indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_api_key(request: APIKeyCreate) -> APIKeyResponse:
    """Generate a new API key, store its hash, and return the raw key once."""
    raw_key = _KEY_PREFIX + secrets.token_hex(_KEY_LENGTH)
    key_id = secrets.token_hex(4)  # 8-char short ID
    hashed = hash_key(raw_key)

    record = APIKeyRecord(
        key_id=key_id,
        key_hash=hashed,
        name=request.name,
        role=request.role,
    )

    records = _load_records()
    records.append(record)
    _save_records(records)
    logger.info("Created API key key_id=%s name=%s role=%s", key_id, request.name, request.role)

    return APIKeyResponse(
        key_id=key_id,
        raw_key=raw_key,
        name=request.name,
        role=request.role,
        created_at=record.created_at,
    )


def validate_key(raw_key: str) -> APIKeyRecord | None:
    """Validate a raw API key and return its record, or ``None`` if invalid."""
    hashed = hash_key(raw_key)
    for record in _load_records():
        if record.key_hash == hashed and not record.revoked:
            return record
    return None


def list_api_keys() -> list[APIKeyInfo]:
    """Return public metadata for all keys (no secret material)."""
    return [
        APIKeyInfo(
            key_id=record.key_id,
            name=record.name,
            role=record.role,
            created_at=record.created_at,
            revoked=record.revoked,
        )
        for record in _load_records()
    ]


def revoke_api_key(key_id: str) -> bool:
    """Revoke a key by its short ID.  Returns ``True`` if the key was found."""
    records = _load_records()
    found = False
    for record in records:
        if record.key_id == key_id and not record.revoked:
            record.revoked = True
            found = True
            logger.info("Revoked API key key_id=%s", key_id)
            break
    if found:
        _save_records(records)
    return found


# ---------------------------------------------------------------------------
# Bootstrap helper  — ``python -m src.auth.api_keys``
# ---------------------------------------------------------------------------

def _bootstrap_admin_key() -> None:
    """Create an initial admin key from ``ADMIN_API_KEY`` env var if set,
    or generate a new one and print it to stdout."""
    admin_env = os.getenv("ADMIN_API_KEY")
    if admin_env:
        # Store the hash of the user-supplied key.
        records = _load_records()
        key_id = secrets.token_hex(4)
        records.append(
            APIKeyRecord(
                key_id=key_id,
                key_hash=hash_key(admin_env),
                name="bootstrap-admin",
                role=Role.ADMIN,
            )
        )
        _save_records(records)
        print(f"Registered bootstrap admin key (key_id={key_id}).")
    else:
        response = create_api_key(APIKeyCreate(name="bootstrap-admin", role=Role.ADMIN))
        print("Generated admin API key (copy it now — it will not be shown again):")
        print(f"  key_id  : {response.key_id}")
        print(f"  raw_key : {response.raw_key}")


if __name__ == "__main__":
    _bootstrap_admin_key()
