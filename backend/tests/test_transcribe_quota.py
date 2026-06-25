"""Testes — limite diário no endpoint /transcribe."""
from __future__ import annotations

import admin_store


class _FakeRequest:
    def __init__(self, ip: str = "testclient"):
        self.client = type("C", (), {"host": ip})()
        self.headers = {}


def test_transcribe_returns_429_when_quota_exhausted(client):
    actor = {"type": "anonymous"}
    key, _ = admin_store.usage_key_for_request(_FakeRequest("testclient"), actor)
    for _ in range(3):
        admin_store.increment_daily_transcribe(key)

    res = client.post(
        "/transcribe",
        data={"token": "test-api-token"},
        files={"file": ("tiny.wav", b"RIFFxxxx", "audio/wav")},
    )
    assert res.status_code == 429
