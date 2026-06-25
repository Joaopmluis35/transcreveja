"""Testes — sugestões públicas e notificação por email."""
from __future__ import annotations

import admin_store


def test_suggestion_api_saves_and_notifies(client, origin_headers, monkeypatch):
    monkeypatch.setenv("TEST_SYNC_NOTIFICATIONS", "1")
    calls: list[dict] = []

    def fake_notify(suggestion_id, nome, mensagem, lang="pt", referer=None):
        calls.append(
            {
                "id": suggestion_id,
                "nome": nome,
                "mensagem": mensagem,
                "lang": lang,
                "referer": referer,
            }
        )
        return True, None

    monkeypatch.setattr("email_notify.send_suggestion_notification", fake_notify)

    res = client.post(
        "/api/suggestions",
        headers={**origin_headers, "Referer": "http://testserver/"},
        json={"mensagem": "Melhorar botões de exportação", "nome": "João"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] is True
    assert body["id"] > 0
    assert len(calls) == 1
    assert calls[0]["mensagem"] == "Melhorar botões de exportação"
    assert calls[0]["nome"] == "João"
    assert calls[0]["referer"] == "http://testserver/"

    rows = admin_store.list_suggestions(limit=5)
    assert any(r["id"] == body["id"] for r in rows)


def test_suggestion_rejects_empty(client, origin_headers):
    res = client.post(
        "/api/suggestions",
        headers=origin_headers,
        json={"mensagem": "   "},
    )
    assert res.status_code == 400


def test_suggestion_rejects_bad_origin(client):
    res = client.post(
        "/api/suggestions",
        headers={"Origin": "https://evil.example"},
        json={"mensagem": "spam"},
    )
    assert res.status_code == 403
