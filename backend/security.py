"""Utilitários de segurança: auth, rate limit, SSRF e validação de origem."""
from __future__ import annotations

import ipaddress
import socket
import time
from collections import defaultdict
from urllib.parse import urlparse

import requests
from fastapi import HTTPException, Request


def parse_csv_env(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


# Proxies à frente da API (Cloudflare → Render). Fonte: https://www.cloudflare.com/ips/
_TRUSTED_PROXY_NETS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
        "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
        "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
        "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
        "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
        "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
        # Rede interna do Render / loopback
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "::1/128", "fc00::/7",
    )
)


def _is_trusted_proxy(value: str) -> bool:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError:
        return False
    return any(addr in net for net in _TRUSTED_PROXY_NETS)


def client_ip(request: Request) -> str:
    """IP real do cliente: percorre o X-Forwarded-For da direita para a esquerda,
    saltando proxies de confiança. As entradas à esquerda são escritas pelo
    cliente e não podem ser usadas (permitiriam contornar o rate limit)."""
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if not forwarded:
        return peer
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    for hop in reversed(hops):
        if not _is_trusted_proxy(hop):
            return hop
    return hops[0] if hops else peer


def request_origin(request: Request) -> str | None:
    origin = request.headers.get("origin")
    if origin:
        return origin.rstrip("/")
    referer = request.headers.get("referer", "")
    if referer:
        parsed = urlparse(referer)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    return None


def origin_is_allowed(request: Request, allowed_origins: list[str]) -> bool:
    if "*" in allowed_origins:
        return True
    origin = request_origin(request)
    if not origin:
        return False
    normalized = {o.rstrip("/") for o in allowed_origins}
    return origin in normalized


class RateLimiter:
  def __init__(self) -> None:
      self._hits: dict[tuple[str, str], list[float]] = defaultdict(list)

  def check(self, ip: str, bucket: str, limit: int, window_sec: int) -> None:
      if limit <= 0:
          return
      now = time.time()
      if len(self._hits) > 10_000:
          self._prune(now, window_sec)
      key = (ip, bucket)
      self._hits[key] = [t for t in self._hits[key] if now - t < window_sec]
      if len(self._hits[key]) >= limit:
          raise HTTPException(status_code=429, detail="Demasiados pedidos. Tenta mais tarde.")
      self._hits[key].append(now)

  def _prune(self, now: float, window_sec: int) -> None:
      # Remove IPs sem pedidos recentes para a memória não crescer sem limite.
      max_age = max(window_sec, 86_400)
      for key in [k for k, hits in self._hits.items() if not hits or now - hits[-1] >= max_age]:
          del self._hits[key]


_PRIVATE_NETS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


def _ip_is_blocked(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return any(addr in net for net in _PRIVATE_NETS)


def _resolve_host_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    if hostname.lower() in ("localhost",):
        return [ipaddress.ip_address("127.0.0.1")]
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="URL inválida ou host inacessível.") from exc
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        ips.append(ipaddress.ip_address(info[4][0]))
    return ips


def validate_public_http_url(url: str, *, allow_localhost: bool = False) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="URL deve usar http ou https.")
    if not parsed.hostname:
        raise HTTPException(status_code=400, detail="URL inválida.")
    host = parsed.hostname.lower()
    if host in ("localhost", "127.0.0.1", "::1") and allow_localhost:
        return url
    for addr in _resolve_host_ips(host):
        if _ip_is_blocked(addr):
            raise HTTPException(status_code=400, detail="URL não permitida.")
    return url


def safe_http_get(
    url: str,
    *,
    timeout: int = 12,
    max_bytes: int = 2_000_000,
    headers: dict | None = None,
) -> requests.Response:
    validate_public_http_url(url)
    hdrs = {
        "User-Agent": "Mozilla/5.0 (compatible; OuviescreviBot/1.0; +https://ouviescrevi.pt)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "pt-PT,pt;q=0.9,en;q=0.8",
    }
    if headers:
        hdrs.update(headers)
    with requests.get(url, timeout=timeout, stream=True, allow_redirects=True, headers=hdrs) as resp:
        resp.raise_for_status()
        content = resp.raw.read(max_bytes + 1, decode_content=True)
        if len(content) > max_bytes:
            raise HTTPException(status_code=400, detail="Resposta da URL demasiado grande.")
        resp._content = content
        return resp
