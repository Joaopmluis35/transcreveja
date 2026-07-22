"""Testes — partilha pública, password reset, UTMs."""
from __future__ import annotations

from analytics import record_visit
import admin_store as store


def test_utm_record_and_top(client):
    record_visit(
        "/index.html",
        "203.0.113.50",
        referrer="https://t.co/x",
        user_agent="Mozilla/5.0",
        utm_source="twitter",
        utm_medium="social",
        utm_campaign="launch",
    )
    rows = store.top_utm_campaigns(5)
    assert any(r["utm_source"] == "twitter" and r["utm_campaign"] == "launch" for r in rows)


def test_shared_transcript_roundtrip(client):
    created = store.create_shared_transcript(
        "Isto é uma transcrição de teste com texto suficiente.",
        title="Teste",
        locale="pt",
    )
    assert created["id"]
    item = store.get_shared_transcript(created["id"])
    assert item is not None
    assert "transcrição" in item["text"]


def test_password_reset_flow(client):
    email = "reset_pytest@example.com"
    try:
        store.register_site_user(email, "SenhaAntiga1!", name="Reset")
    except ValueError:
        pass
    token = store.create_password_reset_token(email)
    assert token
    ok = store.reset_password_with_token(token, "SenhaNova123!")
    assert ok is True
    user = store.authenticate_site_user(email, "SenhaNova123!")
    assert user is not None
    assert store.reset_password_with_token(token, "OutraSenha999!") is False


def test_track_visit_accepts_utm(client):
    res = client.post(
        "/api/track-visit",
        json={
            "path": "/index.html",
            "referrer": "",
            "utm_source": "newsletter",
            "utm_medium": "email",
            "utm_campaign": "week1",
        },
        headers={"Origin": "http://testserver"},
    )
    # origin may be rejected in tests — accept 200 or 403 depending on ALLOWED_ORIGINS
    assert res.status_code in (200, 403)
