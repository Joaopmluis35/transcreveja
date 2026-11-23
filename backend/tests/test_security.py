"""IP real atrás de Cloudflare/Render, tokens e limite de tentativas no login admin."""
from __future__ import annotations

import ipaddress
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import security
from security import client_ip, safe_http_get

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


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.5", "169.254.169.254", "100.64.0.1", "::ffff:127.0.0.1", "::1", "224.0.0.1"],
)
def test_ssrf_blocks_non_public_ips(ip):
    assert security._ip_is_blocked(ipaddress.ip_address(ip))


def test_ssrf_allows_public_ip():
    assert not security._ip_is_blocked(ipaddress.ip_address("93.184.216.34"))


class _FakeResp:
    def __init__(self, status, location=None, body=b"ok"):
        self.status_code = status
        self.headers = {"location": location} if location else {}
        self.is_redirect = location is not None
        self.raw = SimpleNamespace(read=lambda n, decode_content=True: body)

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_dns(monkeypatch):
    ips = {"public.example": "93.184.216.34", "evil.example": "93.184.216.35"}
    monkeypatch.setattr(
        security, "_resolve_host_ips", lambda host: [ipaddress.ip_address(ips.get(host, "10.0.0.1"))]
    )


def test_ssrf_redirect_to_metadata_is_blocked(monkeypatch):
    _fake_dns(monkeypatch)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        return _FakeResp(302, location="http://169.254.169.254/latest/meta-data/")

    monkeypatch.setattr(security.requests, "get", fake_get)
    monkeypatch.setattr(security, "_resolve_host_ips", lambda host: [
        ipaddress.ip_address("93.184.216.34" if host == "evil.example" else host)
    ])
    with pytest.raises(HTTPException) as exc:
        safe_http_get("http://evil.example/")
    assert exc.value.status_code == 400
    assert calls == ["http://evil.example/"]


def test_ssrf_follows_public_redirect(monkeypatch):
    _fake_dns(monkeypatch)
    responses = iter([_FakeResp(301, location="/artigo"), _FakeResp(200, body=b"conteudo")])
    monkeypatch.setattr(security.requests, "get", lambda url, **kw: next(responses))
    resp = safe_http_get("http://public.example/")
    assert resp._content == b"conteudo"


def test_ssrf_too_many_redirects(monkeypatch):
    _fake_dns(monkeypatch)
    monkeypatch.setattr(security.requests, "get", lambda url, **kw: _FakeResp(302, location="/loop"))
    with pytest.raises(HTTPException):
        safe_http_get("http://public.example/")


def test_public_site_origin_ignores_untrusted_origin():
    """Link de recuperação/partilha: uma Origin falsa nunca entra no link (levaria o token a outro site)."""
    from security import PUBLIC_SITE_ORIGIN, public_site_origin

    allowed = ["https://www.ouviescrevi.pt", "http://localhost:5500"]
    fake = SimpleNamespace(headers={"origin": "https://evil.example"})
    assert public_site_origin(fake, allowed) == PUBLIC_SITE_ORIGIN
    assert public_site_origin(SimpleNamespace(headers={}), allowed) == PUBLIC_SITE_ORIGIN
    ok = SimpleNamespace(headers={"origin": "http://localhost:5500/"})
    assert public_site_origin(ok, allowed) == "http://localhost:5500"
    # «*» na lista não pode abrir a porta a qualquer site.
    assert public_site_origin(fake, ["*"]) == PUBLIC_SITE_ORIGIN


def test_forgot_password_email_link_uses_trusted_site(client, monkeypatch):
    import admin_store
    import email_notify

    sent = {}
    monkeypatch.setattr(admin_store, "create_password_reset_token", lambda email: "tok123")
    monkeypatch.setattr(email_notify, "send_password_reset_email", lambda *a, **k: sent.setdefault("args", (a, k)), raising=False)
    r = client.post("/api/auth/forgot-password", json={"email": "a@b.pt"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    blob = repr(sent.get("args"))
    assert "evil.example" not in blob
    assert "https://www.ouviescrevi.pt/index.html?reset=tok123" in blob
