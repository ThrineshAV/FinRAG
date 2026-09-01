"""FastAPI dependencies for API key authentication and role checks.

Usage in endpoints::

    from src.auth.dependencies import require_api_key, require_admin, require_upload

    @app.post("/query")
    async def query(request: Request, key: APIKeyRecord = Depends(require_api_key)):
        ...

    @app.post("/upload")
    async def upload(request: Request, key: APIKeyRecord = Depends(require_upload)):
        ...

    @app.post("/admin/keys")
    async def create_key(request: Request, key: APIKeyRecord = Depends(require_admin)):
        ...
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request, status

from src.auth.api_keys import validate_key
from src.auth.models import APIKeyRecord


def _is_auth_required() -> bool:
    """Return *True* when authentication is enabled.

    Controlled by the ``AUTH_REQUIRED`` env var (default ``true``).
    Set to ``false`` for local development or backward compatibility.
    """
    return os.getenv("AUTH_REQUIRED", "true").lower() in ("true", "1", "yes")


def _extract_api_key(request: Request) -> str | None:
    """Extract the API key from the ``X-API-Key`` header."""
    return request.headers.get("X-API-Key")


async def require_api_key(request: Request) -> APIKeyRecord | None:
    """Validate the ``X-API-Key`` header and return the key record.

    When ``AUTH_REQUIRED`` is ``false`` this dependency returns ``None``
    so callers continue to work without credentials.

    Raises:
        HTTPException: 401 if key is missing or invalid, 503 if validation fails
    """
    if not _is_auth_required():
        return None

    raw_key = _extract_api_key(request)
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass it via the X-API-Key header.",
        )

    record = validate_key(raw_key)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )
    return record


async def require_admin(request: Request) -> APIKeyRecord:
    """Require an authenticated admin key.

    Always enforced, even when ``AUTH_REQUIRED`` is ``false``.
    Admin role is required for key management operations.

    Raises:
        HTTPException: 401 if key missing/invalid, 403 if not admin role
    """
    raw_key = _extract_api_key(request)
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass it via the X-API-Key header.",
        )

    record = validate_key(raw_key)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )
    if not record.has_permission("admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin privileges required. Your key has role '{record.role.value}'.",
        )
    return record


async def require_upload(request: Request) -> APIKeyRecord | None:
    """Require at least ``upload`` permission (admin role).

    When ``AUTH_REQUIRED`` is ``false`` this returns ``None``.
    Upload operations require admin role.

    Raises:
        HTTPException: 401 if key missing/invalid, 403 if no upload permission
    """
    if not _is_auth_required():
        return None

    raw_key = _extract_api_key(request)
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Pass it via the X-API-Key header.",
        )

    record = validate_key(raw_key)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )
    if not record.has_permission("upload"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Upload permission required. Your key has role '{record.role.value}'; 'admin' role needed.",
        )
    return record

