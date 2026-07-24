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


def test_history_search_and_rename(client):
    email = f"hist-{uuid.uuid4().hex[:10]}@ouviescrevi.test"
    password = "TesteSeguro123!"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Hist"},
    )
    assert reg.status_code == 200, reg.text
    login = client.post("/api/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["sessionToken"]
    headers = {"X-Site-Session": token}

    import admin_store as store

    item_id = store.save_user_transcription(
        email,
        filename="reuniao-meet.mp4",
        transcription="João e Daniela falaram sobre o projeto Ouviescrevi.",
        formatted="[00:01] João: Olá Daniela.\n[00:05] Daniela: Vamos ao projeto.",
    )
    assert item_id > 0

    listed = client.get("/api/auth/history", headers=headers)
    assert listed.status_code == 200
    assert any(i["id"] == item_id for i in listed.json()["items"])

    search = client.get("/api/auth/history?q=Daniela", headers=headers)
    assert search.status_code == 200
    ids = [i["id"] for i in search.json()["items"]]
    assert item_id in ids

    miss = client.get("/api/auth/history?q=zzzz-nao-existe", headers=headers)
    assert miss.status_code == 200
    assert all(i["id"] != item_id for i in miss.json()["items"])

    renamed = client.patch(
        f"/api/auth/history/{item_id}",
        headers=headers,
        json={"filename": "Reunião João e Daniela"},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["filename"] == "Reunião João e Daniela"

    by_name = client.get("/api/auth/history?q=Reuni", headers=headers)
    assert by_name.status_code == 200
    assert any(i["id"] == item_id for i in by_name.json()["items"])


def test_export_docx_disabled_without_billing(client):
    res = client.post("/api/export/docx", json={"text": "Texto de teste."})
    assert res.status_code in (401, 403, 503)
