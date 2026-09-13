"""JWT access token utilities (HS256)."""
from __future__ import annotations
import os
import time
from jose import jwt, JWTError

_default_secret = os.getenv("JWT_SECRET")
if _default_secret and len(_default_secret) >= 32:
    JWT_SECRET = _default_secret
else:
    # Fallback only for development / test collection when env is missing
    JWT_SECRET = "dev-test-secret-key-at-least-32-chars-long!!"  # nosec B105: test-only fallback
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15


def create_access_token(user_id: int, email: str, role: str) -> str:
    """Issue a signed JWT access token."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + (ACCESS_TOKEN_EXPIRE_MINUTES * 60),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> dict | None:
    """Verify and decode an access token. Return claims dict or None if invalid/expired."""
    try:
        claims = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return claims
    except JWTError:
        return None
