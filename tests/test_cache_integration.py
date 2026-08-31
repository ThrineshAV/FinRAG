"""Tests for API cache integration."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from src import api
from unittest.mock import MagicMock

# Disable authentication
import os
os.environ["AUTH_REQUIRED"] = "false"

client = TestClient(api.app)

def test_query_endpoint_cache_behaviour(monkeypatch) -> None:
    """Test that POST /query uses cache, updates headers, and tracks stats."""
    # Define mocks
    monkeypatch.setattr(api, "retrieve_documents", lambda question, top_k, filters: ([{"text": "mock chunk"}], {}))
    monkeypatch.setattr(api, "_build_query_response", lambda qr, results, openai_enabled: api.QueryResponse(
        answer="mock answer",
        retrieved_chunks=["mock chunk"],
        metadata=[{"text": "mock chunk"}],
        citations=[]
    ))
    monkeypatch.setattr(api, "is_openai_configured", lambda: False)

    # Clear cache
    api.get_cache_manager().clear()
    api._query_cache_stats.update({"query_hits": 0, "query_misses": 0})

    query = {"question": "What is revenue?"}

    # First request - MISS
    res1 = client.post("/query", json=query)
    assert res1.status_code == 200
    assert res1.headers["X-Cache"] == "MISS"
    assert api._query_cache_stats["query_misses"] == 1

    # Second request - HIT
    res2 = client.post("/query", json=query)
    assert res2.status_code == 200
    assert res2.headers["X-Cache"] == "HIT"
    assert api._query_cache_stats["query_hits"] == 1
    assert res2.json()["answer"] == "mock answer"

def test_query_stream_cache_behaviour(monkeypatch) -> None:
    """Test that POST /query/stream uses cache, updates headers, and tracks stats."""
    # Define mocks
    monkeypatch.setattr(api, "retrieve_documents", lambda question, top_k, filters: ([{"text": "mock chunk"}], {}))
    monkeypatch.setattr(api, "is_openai_configured", lambda: True)

    # Mock LLM stream
    def fake_stream(question, context):
        yield "mock "
        yield "answer"
    monkeypatch.setattr(api, "generate_openai_answer_stream", fake_stream)

    # Clear cache
    api.get_cache_manager().clear()
    api._query_cache_stats.update({"stream_hits": 0, "stream_misses": 0})

    query = {"question": "What is revenue stream?"}

    # First request - MISS (streaming response)
    res1 = client.post("/query/stream", json=query)
    assert res1.status_code == 200
    assert res1.headers["X-Cache"] == "MISS"
    assert api._query_cache_stats["stream_misses"] == 1

    # Second request - HIT (streaming response, served as cache hit with JSON)
    res2 = client.post("/query/stream", json=query)
    assert res2.status_code == 200
    assert res2.headers["X-Cache"] == "HIT"
    assert api._query_cache_stats["stream_hits"] == 1
    assert "cached" in res2.text or "true" in res2.text # Should be the cached response SSE
