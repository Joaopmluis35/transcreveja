"""Testes — quotas diárias e autenticação de passwords."""
from __future__ import annotations

import admin_store


class _FakeRequest:
  def __init__(self, ip: str = "203.0.113.10"):
    self.client = type("C", (), {"host": ip})()
    self.headers = {}


def test_password_hash_roundtrip():
    stored = admin_store._hash_password("segredo-forte-123")
    assert admin_store._verify_password("segredo-forte-123", stored)
    assert not admin_store._verify_password("errado", stored)


def test_anonymous_quota_defaults():
    actor = {"type": "anonymous"}
    status = admin_store.transcribe_quota_status(_FakeRequest(), actor)
    assert status["tier"] == "anonymous"
    assert status["limit"] == 3
    assert status["remaining"] == 3


def test_registered_quota_defaults():
    actor = {"type": "user", "email": "aluno@example.com"}
    status = admin_store.transcribe_quota_status(_FakeRequest(), actor)
    assert status["tier"] == "registered"
    assert status["limit"] == 20


def test_quota_message_no_pro_when_pricing_hidden(monkeypatch):
    monkeypatch.setenv("PRICING_HIDDEN", "1")
    monkeypatch.setenv("BILLING_ENABLED", "0")
    actor = {"type": "user", "email": "cheio@example.com"}
    key, _ = admin_store.usage_key_for_request(_FakeRequest("203.0.113.99"), actor)
    for _ in range(25):
        admin_store.increment_daily_transcribe(key)
    status = admin_store.transcribe_quota_status(_FakeRequest("203.0.113.99"), actor)
    assert status["remaining"] == 0
    assert "Pro" not in (status.get("message") or "")


def test_anonymous_quota_reads_backoffice_config(monkeypatch):
    admin_store.set_config({"quota_anonymous_daily": "7"}, "pytest")
    try:
        actor = {"type": "anonymous"}
        status = admin_store.transcribe_quota_status(_FakeRequest(), actor)
        assert status["limit"] == 7
        assert status["remaining"] == 7
    finally:
        admin_store.set_config({"quota_anonymous_daily": "3"}, "pytest")
