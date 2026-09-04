"""FastAPI dependencies for JWT Bearer authentication and role checks.

Usage in endpoints::

    from src.auth.dependencies import require_auth, require_admin, require_upload

    @app.post("/query")
    async def query(request: Request, claims = Depends(require_auth)):
        ...

    @app.post("/upload")
    async def upload(request: Request, claims = Depends(require_upload)):
        ...
"""

from __future__ import annotations

import os

from fastapi import Depends, HTTPException, Request, status

from src.auth import jwt_utils
from src.auth.models import Role


def _is_auth_required() -> bool:
    """Return *True* when authentication is enabled.

    Controlled by the ``AUTH_REQUIRED`` env var (default ``true``).
    Set to ``false`` for local development or backward compatibility.
    """
    return os.getenv("AUTH_REQUIRED", "true").lower() in ("true", "1", "yes")


def _extract_bearer(request: Request) -> str | None:
    """Extract the JWT token from the Authorization: Bearer header."""
    header = request.headers.get("Authorization")
    if header and header.startswith("Bearer "):
        return header.split(" ", 1)[1]
    return None


async def require_auth(request: Request) -> dict:
    """Validate the JWT Bearer token and return the claims.

    When ``AUTH_REQUIRED`` is ``false`` this dependency returns ``None``
    so callers continue to work without credentials.

    Raises:
        HTTPException: 401 if token is missing or invalid
    """
    if not _is_auth_required():
        return None

    token = _extract_bearer(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token. Pass it via the Authorization: Bearer header.",
        )

    claims = jwt_utils.verify_access_token(token)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    return claims


async def require_admin(request: Request) -> dict:
    """Require an authenticated user with admin role.

    Always enforced, even when ``AUTH_REQUIRED`` is ``false``.

    Raises:
        HTTPException: 401 if token missing/invalid, 403 if not admin role
    """
    token = _extract_bearer(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
        )

    claims = jwt_utils.verify_access_token(token)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    if claims.get("role") != Role.ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin privileges required. Your role is '{claims.get('role')}'.",
        )
    return claims


async def require_upload(request: Request) -> dict | None:
    """Require at least uploader or admin role.

    When ``AUTH_REQUIRED`` is ``false`` this returns ``None``.

    Raises:
        HTTPException: 401 if token missing/invalid, 403 if no upload permission
    """
    if not _is_auth_required():
        return None

    token = _extract_bearer(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token.",
        )

    claims = jwt_utils.verify_access_token(token)
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        )
    role = claims.get("role")
    if role not in (Role.ADMIN.value, "uploader"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Upload permission required. Your role is '{role}'; 'admin' or 'uploader' needed.",
        )
    return claims
