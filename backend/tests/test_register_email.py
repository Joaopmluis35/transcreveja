"""Testes — email de boas-vindas no registo."""
from __future__ import annotations

import uuid


def test_register_sends_welcome_email(client, monkeypatch):
    monkeypatch.setenv("TEST_SYNC_NOTIFICATIONS", "1")
    welcome_calls: list[tuple[str, str | None]] = []

    def fake_welcome(to, name=None):
        welcome_calls.append((to, name))
        return True, None

    monkeypatch.setattr("email_notify.send_welcome_email", fake_welcome)

    email = f"pytest-{uuid.uuid4().hex[:10]}@ouviescrevi.test"
    res = client.post(
        "/api/auth/register",
        json={"email": email, "password": "TesteSeguro123!", "name": "Novo Utilizador"},
    )
    assert res.status_code == 200, res.text
    assert len(welcome_calls) == 1
    assert welcome_calls[0][0] == email
    assert welcome_calls[0][1] == "Novo Utilizador"
