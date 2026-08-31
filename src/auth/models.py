"""Pydantic models for API key authentication."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """User roles for authorization."""

    READER = "reader"
    ADMIN = "admin"


# Permissions granted to each role.
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.READER: {"query"},
    Role.ADMIN: {"query", "upload", "admin"},
}


class APIKeyRecord(BaseModel):
    """Stored representation of an API key.

    The raw key is never persisted — only the SHA-256 hash.
    """

    key_id: str = Field(description="Short unique identifier for the key")
    key_hash: str = Field(description="SHA-256 hex digest of the raw key")
    name: str = Field(description="Human-readable label")
    role: Role = Field(default=Role.READER, description="Authorization role")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 creation timestamp",
    )
    revoked: bool = Field(default=False, description="Whether the key has been revoked")

    def has_permission(self, permission: str) -> bool:
        """Return *True* if this key's role grants *permission*."""
        return permission in ROLE_PERMISSIONS.get(self.role, set())


class APIKeyCreate(BaseModel):
    """Request body for creating a new API key."""

    name: str = Field(min_length=1, max_length=128, description="Label for the key")
    role: Role = Field(default=Role.READER, description="Role to assign")


class APIKeyResponse(BaseModel):
    """Response returned when a new key is created.

    ``raw_key`` is shown exactly once and must be copied by the caller.
    """

    key_id: str
    raw_key: str
    name: str
    role: Role
    created_at: str


class APIKeyInfo(BaseModel):
    """Public metadata about an existing key (no secret material)."""

    key_id: str
    name: str
    role: Role
    created_at: str
    revoked: bool
