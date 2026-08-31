"""Cache management with Redis support and in-memory fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

try:  # Redis is optional at runtime so local tests can run without a server.
    import redis
except ImportError:  # pragma: no cover - depends on installed optional package
    redis = None

logger = logging.getLogger(__name__)

_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_DEFAULT_TTL_SECONDS = 3600


class CacheManager:
    """Manage cache reads and writes with graceful Redis fallback.

    Redis is used when the ``redis`` package is installed and ``REDIS_URL`` is
    reachable. If Redis is unavailable, the manager falls back to an in-memory
    cache so development and tests do not require external services.
    """

    def __init__(
        self,
        enabled: bool | None = None,
        redis_url: str | None = None,
        default_ttl_seconds: int | None = None,
    ) -> None:
        self.enabled = _env_bool("CACHE_ENABLED", True) if enabled is None else enabled
        self.redis_url = redis_url or os.getenv("REDIS_URL", _DEFAULT_REDIS_URL)
        self.default_ttl_seconds = default_ttl_seconds or int(
            os.getenv("CACHE_TTL_SECONDS", str(_DEFAULT_TTL_SECONDS))
        )
        self.backend = "disabled"
        self._redis_client: Any | None = None
        self._memory_cache: dict[str, tuple[float | None, str]] = {}

        if self.enabled:
            self._connect_redis()
            if self._redis_client is None:
                self.backend = "memory"

    def _connect_redis(self) -> None:
        """Connect to Redis if possible; otherwise keep fallback available."""
        if redis is None:
            logger.info("Redis package is not installed; using in-memory cache")
            return

        try:
            client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            client.ping()
        except Exception as exc:  # pragma: no cover - depends on external Redis
            logger.warning("Redis cache unavailable; using in-memory cache: %s", exc)
            return

        self._redis_client = client
        self.backend = "redis"

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` on miss/expiry."""
        if not self.enabled:
            return None

        if self._redis_client is not None:
            try:
                value = self._redis_client.get(key)
            except Exception as exc:  # pragma: no cover - depends on external Redis
                logger.warning("Redis cache read failed for %s: %s", key, exc)
                return None
            return _deserialize(value) if value is not None else None

        cached = self._memory_cache.get(key)
        if cached is None:
            return None

        expires_at, value = cached
        if expires_at is not None and expires_at <= time.time():
            self._memory_cache.pop(key, None)
            return None
        return _deserialize(value)

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Store *value* for *key* and return whether the write succeeded."""
        if not self.enabled:
            return False

        ttl_seconds = self.default_ttl_seconds if ttl is None else ttl
        serialized = _serialize(value)

        if self._redis_client is not None:
            try:
                self._redis_client.setex(key, ttl_seconds, serialized)
            except Exception as exc:  # pragma: no cover - depends on external Redis
                logger.warning("Redis cache write failed for %s: %s", key, exc)
                return False
            return True

        expires_at = time.time() + ttl_seconds if ttl_seconds > 0 else None
        self._memory_cache[key] = (expires_at, serialized)
        return True

    def delete(self, key: str) -> bool:
        """Delete *key* from the cache."""
        if not self.enabled:
            return False

        if self._redis_client is not None:
            try:
                return bool(self._redis_client.delete(key))
            except Exception as exc:  # pragma: no cover - depends on external Redis
                logger.warning("Redis cache delete failed for %s: %s", key, exc)
                return False

        return self._memory_cache.pop(key, None) is not None

    def clear(self, prefix: str | None = None) -> int:
        """Clear cache entries, optionally limited to keys starting with *prefix*."""
        if not self.enabled:
            return 0

        if self._redis_client is not None:
            pattern = f"{prefix}*" if prefix else "*"
            try:
                keys = list(self._redis_client.scan_iter(match=pattern))
                if not keys:
                    return 0
                return int(self._redis_client.delete(*keys))
            except Exception as exc:  # pragma: no cover - depends on external Redis
                logger.warning("Redis cache clear failed for pattern %s: %s", pattern, exc)
                return 0

        keys_to_delete = [
            key for key in self._memory_cache
            if prefix is None or key.startswith(prefix)
        ]
        for key in keys_to_delete:
            self._memory_cache.pop(key, None)
        return len(keys_to_delete)


def build_cache_key(namespace: str, payload: Any) -> str:
    """Build a stable cache key from a namespace and JSON-serializable payload."""
    normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"finsight:{namespace}:{digest}"


_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Return the process-wide cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def _env_bool(name: str, default: bool) -> bool:
    """Parse an environment variable as a boolean."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _serialize(value: Any) -> str:
    """Serialize a value for storage."""
    return json.dumps(value, sort_keys=True, default=str)


def _deserialize(value: str) -> Any:
    """Deserialize a cached value."""
    return json.loads(value)
