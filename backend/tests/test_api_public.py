"""Testes de API pública — smoke tests sem OpenAI."""
from __future__ import annotations

import uuid


def test_health_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    body = res.json()
    assert "manutencao" in body or "maintenance" in body or isinstance(body, dict)


def test_billing_status_free_mode(client):
    res = client.get("/api/billing/status")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is False
    assert data["pricing_hidden"] is True
    assert data["checkout_ready"] is False


def test_frontend_config(client, origin_headers):
    res = client.get("/api/frontend-config", headers=origin_headers)
    assert res.status_code == 200
    data = res.json()
    assert "apiBase" in data
    assert data.get("pricingHidden") is True


def test_usage_anonymous(client):
    res = client.get("/api/usage")
    assert res.status_code == 200
    data = res.json()
    assert data["tier"] == "anonymous"
    assert data["limit"] == 3
    assert data["remaining"] >= 0


def test_auth_register_and_login(client):
    email = f"pytest-{uuid.uuid4().hex[:10]}@ouviescrevi.test"
    password = "TesteSeguro123!"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Pytest"},
    )
    assert reg.status_code == 200, reg.text
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json().get("sessionToken")
    assert token
    me = client.get("/api/auth/me", headers={"X-Site-Session": token})
    assert me.status_code == 200
    assert me.json().get("email") == email


def test_history_requires_login(client):
    res = client.get("/api/auth/history")
    assert res.status_code == 403


def test_export_docx_disabled_without_billing(client):
    res = client.post("/api/export/docx", json={"text": "Texto de teste."})
    assert res.status_code in (401, 403, 503)
