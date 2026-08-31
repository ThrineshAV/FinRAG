"""Tests for cache manager behavior."""

from __future__ import annotations

import time

from src.cache.manager import CacheManager, build_cache_key


def test_build_cache_key_is_stable_for_equivalent_payloads() -> None:
    """Equivalent payloads produce the same cache key."""
    first = build_cache_key("query", {"company": "Apple", "top_k": 5})
    second = build_cache_key("query", {"top_k": 5, "company": "Apple"})

    assert first == second
    assert first.startswith("finsight:query:")


def test_build_cache_key_changes_for_different_payloads() -> None:
    """Different payloads produce different cache keys."""
    first = build_cache_key("query", {"question": "What is revenue?"})
    second = build_cache_key("query", {"question": "What is net income?"})

    assert first != second


def test_memory_cache_set_and_get_round_trip() -> None:
    """In-memory cache stores JSON-serializable values."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    assert cache.set("test:key", {"answer": "Revenue increased", "score": 0.91}) is True
    assert cache.get("test:key") == {"answer": "Revenue increased", "score": 0.91}


def test_memory_cache_returns_none_for_missing_key() -> None:
    """Cache miss returns None."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    assert cache.get("missing:key") is None


def test_memory_cache_expires_values() -> None:
    """Expired in-memory values are removed and return None."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    assert cache.set("short:key", "value", ttl=1) is True
    assert cache.get("short:key") == "value"
    time.sleep(1.1)

    assert cache.get("short:key") is None


def test_disabled_cache_does_not_store_values() -> None:
    """Disabled cache ignores reads and writes."""
    cache = CacheManager(enabled=False)

    assert cache.set("test:key", "value") is False
    assert cache.get("test:key") is None
    assert cache.delete("test:key") is False
    assert cache.clear() == 0


def test_memory_cache_delete_removes_value() -> None:
    """Deleting a key removes it from the cache."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    cache.set("test:key", "value")

    assert cache.delete("test:key") is True
    assert cache.get("test:key") is None


def test_memory_cache_clear_removes_matching_prefix_only() -> None:
    """Clearing by prefix only removes matching entries."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    cache.set("query:first", "first")
    cache.set("query:second", "second")
    cache.set("embedding:first", "embedding")

    assert cache.clear(prefix="query:") == 2
    assert cache.get("query:first") is None
    assert cache.get("query:second") is None
    assert cache.get("embedding:first") == "embedding"


def test_memory_cache_clear_without_prefix_removes_all_values() -> None:
    """Clearing without a prefix removes all entries."""
    cache = CacheManager(enabled=True, redis_url="redis://invalid:6379/0")
    cache._redis_client = None
    cache.backend = "memory"

    cache.set("first", 1)
    cache.set("second", 2)

    assert cache.clear() == 2
    assert cache.get("first") is None
    assert cache.get("second") is None
