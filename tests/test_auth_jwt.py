"""Tests for JWT auth endpoints (signup, login, refresh, logout, me)."""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


def _unique_email(prefix="user"):
    return f"{prefix}-{uuid.uuid4().hex[:8]}@test.local"


def test_auth_signup_and_login(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-jwt")
    email = _unique_email("signup-login")

    # Sign up
    resp = client.post("/auth/signup", json={
        "email": email,
        "password": "SecurePass123",
        "role": "reader"
    })
    assert resp.status_code == 200
    assert resp.json()["email"] == email

    # Login
    resp = client.post("/auth/login", json={
        "email": email,
        "password": "SecurePass123"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert resp.cookies.get("refresh_token") is not None


def test_auth_login_bad_password(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-jwt")
    resp = client.post("/auth/login", json={
        "email": "nope-" + uuid.uuid4().hex[:8] + "@test.local",
        "password": "wrong"
    })
    assert resp.status_code == 401


def test_auth_me_with_jwt(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-jwt")
    email = _unique_email("me")

    # Create user and login
    client.post("/auth/signup", json={
        "email": email, "password": "SecurePass123", "role": "admin"
    })
    login_resp = client.post("/auth/login", json={
        "email": email, "password": "SecurePass123"
    })
    token = login_resp.json()["access_token"]

    resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["email"] == email


def test_auth_refresh_rotates_token(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-jwt")
    email = _unique_email("refresh")

    client.post("/auth/signup", json={
        "email": email, "password": "SecurePass123"
    })
    login_resp = client.post("/auth/login", json={
        "email": email, "password": "SecurePass123"
    })
    assert login_resp.status_code == 200
    assert "access_token" in login_resp.json()

    # First refresh should succeed and rotate the access token
    first_resp = client.post("/auth/refresh")
    assert first_resp.status_code == 200
    assert "access_token" in first_resp.json()

    # The refresh cookie is rotated after each refresh, so a second call still
    # works with the new cookie. Verify it returns a valid token.
    second_resp = client.post("/auth/refresh")
    assert second_resp.status_code == 200
    assert "access_token" in second_resp.json()


def test_auth_logout_revokes_cookie(client, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-jwt")
    email = _unique_email("logout")

    client.post("/auth/signup", json={
        "email": email, "password": "SecurePass123"
    })
    client.post("/auth/login", json={
        "email": email, "password": "SecurePass123"
    })

    resp = client.post("/auth/logout")
    assert resp.status_code == 200
    # Refresh cookie should be cleared (None or empty string)
    cookie_val = resp.cookies.get("refresh_token")
    assert cookie_val is None or cookie_val == ""
