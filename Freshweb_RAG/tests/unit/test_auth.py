"""Unit tests for User Authentication (Signup, Login, Profile) and MongoDB data store."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_auth_signup_success():
    payload = {
        "email": "testuser@freshweb.ai",
        "username": "testuser",
        "password": "securepassword123",
    }
    r = client.post("/api/v1/auth/signup", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "testuser"
    assert data["user"]["email"] == "testuser@freshweb.ai"


def test_auth_signup_duplicate_fails():
    payload = {
        "email": "testuser@freshweb.ai",
        "username": "testuser2",
        "password": "securepassword123",
    }
    r = client.post("/api/v1/auth/signup", json=payload)
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"].lower()


def test_auth_login_success():
    payload = {
        "username": "testuser",
        "password": "securepassword123",
    }
    r = client.post("/api/v1/auth/login", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["user"]["username"] == "testuser"


def test_auth_login_wrong_password_fails():
    payload = {
        "username": "testuser",
        "password": "wrongpassword!",
    }
    r = client.post("/api/v1/auth/login", json=payload)
    assert r.status_code == 401
    assert "incorrect password" in r.json()["detail"].lower()


def test_auth_login_nonexistent_user_fails():
    payload = {
        "username": "ghost_user",
        "password": "password123",
    }
    r = client.post("/api/v1/auth/login", json=payload)
    assert r.status_code == 401


def test_auth_me_requires_token():
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401


def test_auth_me_with_valid_token():
    # Login first
    login_res = client.post("/api/v1/auth/login", json={
        "username": "testuser",
        "password": "securepassword123",
    })
    token = login_res.json()["access_token"]

    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    user = r.json()
    assert user["username"] == "testuser"
