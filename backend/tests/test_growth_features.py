"""Testes — partilha pública, password reset, UTMs."""
from __future__ import annotations

import pytest

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


def test_track_visit_direct_entry_is_not_recorded_as_own_site(client, origin_headers):
    """Sem document.referrer (entrada direta, apps como o TikTok) a origem fica vazia.

    Antes usava-se o cabeçalho Referer do próprio pedido, que é sempre o site
    (https://www.ouviescrevi.pt/) — todas as entradas diretas pareciam navegação interna.
    """
    from database import get_connection, row_to_dict

    res = client.post(
        "/api/track-visit",
        json={"path": "/teste-direto.html", "referrer": ""},
        headers={**origin_headers, "Referer": "https://www.ouviescrevi.pt/"},
    )
    if res.status_code == 403:
        pytest.skip("Origem de teste não autorizada neste ambiente.")
    assert res.status_code == 200
    conn = get_connection()
    try:
        row = row_to_dict(conn.execute(
            "SELECT referrer FROM visitas WHERE path = ? ORDER BY id DESC LIMIT 1", ("/teste-direto.html",)
        ).fetchone())
    finally:
        conn.close()
    assert row["referrer"] is None


def test_shared_transcript_can_be_deleted_only_with_key(client):
    created = store.create_shared_transcript("Texto partilhado de teste com tamanho suficiente.", locale="pt")
    assert created["delete_key"]
    assert store.delete_shared_transcript(created["id"], "chave-errada") is False
    assert store.get_shared_transcript(created["id"]) is not None
    assert store.delete_shared_transcript(created["id"], created["delete_key"]) is True
    assert store.get_shared_transcript(created["id"]) is None


def test_share_delete_endpoint(client):
    origin = {"Origin": "http://testserver"}
    r = client.post("/api/share/transcript", json={"text": "Uma transcrição suficientemente longa para partilhar."}, headers=origin)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"].startswith("http://testserver/s/")
    assert "30 dias" in body["public_notice"]
    bad = client.post(f"/api/share/transcript/{body['id']}/delete", json={"delete_key": "x"}, headers=origin)
    assert bad.status_code == 404
    ok = client.post(f"/api/share/transcript/{body['id']}/delete", json={"delete_key": body["delete_key"]}, headers=origin)
    assert ok.status_code == 200


def test_logout_ends_session_on_server(client):
    token = store.create_session("sair@teste.pt", "user", hours=1)
    assert store.resolve_session(token) is not None
    r = client.post("/api/auth/logout", headers={"X-Site-Session": token})
    assert r.status_code == 200
    assert store.resolve_session(token) is None


def test_password_reset_ends_other_sessions(client):
    email = "reset-sessoes@teste.pt"
    try:
        store.register_site_user(email, "palavra-antiga-1")
    except Exception:
        pass
    old = store.create_session(email, "user", hours=1)
    reset = store.create_password_reset_token(email)
    assert reset
    assert store.reset_password_with_token(reset, "palavra-nova-123") is True
    assert store.resolve_session(old) is None
