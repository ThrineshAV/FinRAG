"""Test suite for authentication and authorization."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.auth.api_keys import create_api_key, hash_key, revoke_api_key, validate_key
from src.auth.models import APIKeyCreate, Role


@pytest.fixture
def temp_keys_file(monkeypatch):
    """Create a temporary API keys file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        tmp.write("[]")
        tmp_path = tmp.name
    monkeypatch.setenv("API_KEYS_FILE", tmp_path)
    yield tmp_path
    Path(tmp_path).unlink(missing_ok=True)


@pytest.fixture
def auth_enabled(monkeypatch):
    """Enable authentication for tests."""
    monkeypatch.setenv("AUTH_REQUIRED", "true")


@pytest.fixture
def auth_disabled(monkeypatch):
    """Disable authentication for backward compatibility tests."""
    monkeypatch.setenv("AUTH_REQUIRED", "false")


@pytest.fixture
def reader_key(temp_keys_file):
    """Create a reader-role API key for testing."""
    response = create_api_key(APIKeyCreate(name="test-reader", role=Role.READER))
    return response.raw_key


@pytest.fixture
def admin_key(temp_keys_file):
    """Create an admin-role API key for testing."""
    response = create_api_key(APIKeyCreate(name="test-admin", role=Role.ADMIN))
    return response.raw_key


# ---------------------------------------------------------------------------
# API Key Management Tests
# ---------------------------------------------------------------------------

def test_create_api_key_generates_valid_key(temp_keys_file):
    """Creating a key returns a raw key with the correct prefix."""
    response = create_api_key(APIKeyCreate(name="test-key", role=Role.READER))
    assert response.raw_key.startswith("fsr_")
    assert len(response.raw_key) > 20
    assert response.name == "test-key"
    assert response.role == Role.READER
    assert response.key_id


def test_validate_key_returns_record_for_valid_key(temp_keys_file):
    """Validating a raw key returns its record."""
    response = create_api_key(APIKeyCreate(name="valid-key", role=Role.ADMIN))
    record = validate_key(response.raw_key)
    assert record is not None
    assert record.name == "valid-key"
    assert record.role == Role.ADMIN
    assert not record.revoked


def test_validate_key_returns_none_for_invalid_key(temp_keys_file):
    """Validating a non-existent key returns None."""
    assert validate_key("fsr_invalid_key_12345") is None


def test_validate_key_returns_none_for_revoked_key(temp_keys_file):
    """Validating a revoked key returns None."""
    response = create_api_key(APIKeyCreate(name="revoked-key", role=Role.READER))
    revoke_api_key(response.key_id)
    assert validate_key(response.raw_key) is None


def test_revoke_api_key_marks_key_as_revoked(temp_keys_file):
    """Revoking a key prevents future validation."""
    response = create_api_key(APIKeyCreate(name="revoke-me", role=Role.READER))
    assert revoke_api_key(response.key_id) is True
    assert validate_key(response.raw_key) is None


def test_revoke_nonexistent_key_returns_false(temp_keys_file):
    """Revoking a non-existent key returns False."""
    assert revoke_api_key("nonexistent-id") is False


def test_hash_key_is_deterministic():
    """Hashing the same key twice produces the same hash."""
    key = "fsr_test_key_123"
    hash1 = hash_key(key)
    hash2 = hash_key(key)
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest


def test_hash_key_produces_different_hashes_for_different_keys():
    """Different keys produce different hashes."""
    hash1 = hash_key("fsr_key_1")
    hash2 = hash_key("fsr_key_2")
    assert hash1 != hash2


# ---------------------------------------------------------------------------
# Role-Based Access Control Tests
# ---------------------------------------------------------------------------

def test_reader_can_query(temp_keys_file, auth_enabled, reader_key):
    """Reader role can access query endpoints."""
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": "What is revenue?"},
        headers={"X-API-Key": reader_key},
    )
    # Expect 200 or 503 — either is fine, we're testing auth not retrieval
    assert response.status_code in (200, 503)


def test_reader_cannot_upload(temp_keys_file, auth_enabled, reader_key):
    """Reader role cannot upload documents."""
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        data={"company": "Test", "document_type": "10-K", "fiscal_year": "2024", "quarter": "Q1"},
        headers={"X-API-Key": reader_key},
    )
    assert response.status_code == 403
    assert "Upload permission required" in response.json()["detail"]


def test_admin_can_query(temp_keys_file, auth_enabled, admin_key):
    """Admin role can access query endpoints."""
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": "What is revenue?"},
        headers={"X-API-Key": admin_key},
    )
    # Expect 200 or 503 — either is fine, we're testing auth not retrieval
    assert response.status_code in (200, 503)


def test_admin_can_upload(temp_keys_file, auth_enabled, admin_key):
    """Admin role can upload documents (auth check only, not full upload)."""
    client = TestClient(app)
    # Create a minimal valid PDF
    minimal_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
    response = client.post(
        "/upload",
        files={"file": ("test.pdf", minimal_pdf, "application/pdf")},
        data={"company": "Test", "document_type": "10-K", "fiscal_year": "2024", "quarter": "Q1"},
        headers={"X-API-Key": admin_key},
    )
    # Should fail at PDF text extraction (422 - no text), not authorization (403)
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"].lower()


def test_admin_can_create_keys(temp_keys_file, auth_enabled, admin_key):
    """Admin can create new API keys."""
    client = TestClient(app)
    response = client.post(
        "/admin/keys",
        json={"name": "new-key", "role": "reader"},
        headers={"X-API-Key": admin_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "new-key"
    assert data["role"] == "reader"
    assert "raw_key" in data


def test_admin_can_list_keys(temp_keys_file, auth_enabled, admin_key):
    """Admin can list all API keys."""
    client = TestClient(app)
    response = client.get("/admin/keys", headers={"X-API-Key": admin_key})
    assert response.status_code == 200
    keys = response.json()
    assert isinstance(keys, list)
    assert len(keys) >= 1  # At least the admin key


def test_admin_can_revoke_keys(temp_keys_file, auth_enabled, admin_key, reader_key):
    """Admin can revoke API keys."""
    client = TestClient(app)
    # Get the reader key_id
    record = validate_key(reader_key)
    assert record is not None
    key_id = record.key_id

    response = client.delete(f"/admin/keys/{key_id}", headers={"X-API-Key": admin_key})
    assert response.status_code == 200
    assert "revoked successfully" in response.json()["detail"]

    # Verify the key is now invalid
    assert validate_key(reader_key) is None


def test_reader_cannot_access_admin_endpoints(temp_keys_file, auth_enabled, reader_key):
    """Reader role cannot access admin endpoints."""
    client = TestClient(app)
    response = client.post(
        "/admin/keys",
        json={"name": "unauthorized", "role": "reader"},
        headers={"X-API-Key": reader_key},
    )
    assert response.status_code == 403
    assert "Admin privileges required" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Authentication Bypass Tests
# ---------------------------------------------------------------------------

def test_query_without_key_fails_when_auth_enabled(temp_keys_file, auth_enabled):
    """Query without API key returns 401 when auth is enabled."""
    client = TestClient(app)
    response = client.post("/query", json={"question": "What is revenue?"})
    assert response.status_code == 401
    assert "Missing API key" in response.json()["detail"]


def test_query_with_invalid_key_fails(temp_keys_file, auth_enabled):
    """Query with invalid API key returns 401."""
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": "What is revenue?"},
        headers={"X-API-Key": "fsr_invalid_key"},
    )
    assert response.status_code == 401
    assert "Invalid or revoked" in response.json()["detail"]


def test_query_works_without_key_when_auth_disabled(temp_keys_file, auth_disabled):
    """Query without API key succeeds when AUTH_REQUIRED is false."""
    client = TestClient(app)
    response = client.post("/query", json={"question": "What is revenue?"})
    # Expect 200 or 503 (vector store ready or not), but NOT 401
    assert response.status_code in (200, 503)


def test_health_endpoint_always_public(temp_keys_file, auth_enabled):
    """Health endpoint is public even when auth is enabled."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_always_public(temp_keys_file, auth_enabled):
    """Ready endpoint is public even when auth is enabled."""
    client = TestClient(app)
    response = client.get("/ready")
    # Expect 200 or 503 depending on whether vector store exists, but NOT 401
    assert response.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Admin Endpoint Security Tests
# ---------------------------------------------------------------------------

def test_admin_endpoints_always_require_auth_even_when_disabled(temp_keys_file, auth_disabled):
    """Admin endpoints require authentication even when AUTH_REQUIRED is false."""
    client = TestClient(app)
    response = client.post("/admin/keys", json={"name": "test", "role": "reader"})
    assert response.status_code == 401


def test_admin_endpoints_reject_revoked_keys(temp_keys_file, auth_enabled, admin_key):
    """Admin endpoints reject revoked keys."""
    client = TestClient(app)
    record = validate_key(admin_key)
    revoke_api_key(record.key_id)

    response = client.get("/admin/keys", headers={"X-API-Key": admin_key})
    assert response.status_code == 401
    assert "Invalid or revoked" in response.json()["detail"]


def test_delete_nonexistent_key_returns_404(temp_keys_file, auth_enabled, admin_key):
    """Deleting a non-existent key returns 404."""
    client = TestClient(app)
    response = client.delete("/admin/keys/nonexistent-id", headers={"X-API-Key": admin_key})
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
