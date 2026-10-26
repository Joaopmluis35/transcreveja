"""IP real atrás de Cloudflare/Render, tokens e limite de tentativas no login admin."""
from __future__ import annotations

from types import SimpleNamespace

from security import client_ip

CF_EDGE = "172.69.10.20"


def _req(xff: str | None, peer: str = "10.1.2.3"):
    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=peer))


def test_client_ip_without_forwarded_uses_peer():
    assert client_ip(_req(None, peer="203.0.113.5")) == "203.0.113.5"


def test_client_ip_via_cloudflare_ignores_spoofed_left_entries():
    assert client_ip(_req(f"1.1.1.1, 198.51.100.7, {CF_EDGE}")) == "198.51.100.7"


def test_client_ip_direct_to_render_uses_rightmost():
    assert client_ip(_req("1.1.1.1, 198.51.100.7")) == "198.51.100.7"


def test_client_ip_spoofed_cloudflare_ip_does_not_win():
    # Acesso direto ao Render: o cliente finge vir do Cloudflare, mas o Render acrescenta o IP real.
    assert client_ip(_req(f"{CF_EDGE}, 198.51.100.7")) == "198.51.100.7"


def test_api_token_is_never_admin_token():
    import main

    assert main.API_TOKEN != main.ADMIN_TOKEN


def test_admin_login_is_rate_limited(client):
    headers = {"X-Forwarded-For": "198.51.100.200"}
    codes = [
        client.post("/api/admin/login", json={"password": "errada"}, headers=headers).status_code
        for _ in range(11)
    ]
    assert codes[:10] == [403] * 10
    assert codes[10] == 429
