"""Pydantic models for JWT user authentication and authorization."""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """User roles for authorization."""

    READER = "reader"
    ADMIN = "admin"


# Permissions granted to each role.
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.READER: {"query"},
    Role.ADMIN: {"query", "upload", "admin"},
}
