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


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


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
      key = (ip, bucket)
      self._hits[key] = [t for t in self._hits[key] if now - t < window_sec]
      if len(self._hits[key]) >= limit:
          raise HTTPException(status_code=429, detail="Demasiados pedidos. Tenta mais tarde.")
      self._hits[key].append(now)


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


def safe_http_get(url: str, *, timeout: int = 12, max_bytes: int = 2_000_000) -> requests.Response:
    validate_public_http_url(url)
    with requests.get(url, timeout=timeout, stream=True, allow_redirects=True) as resp:
        resp.raise_for_status()
        content = resp.raw.read(max_bytes + 1, decode_content=True)
        if len(content) > max_bytes:
            raise HTTPException(status_code=400, detail="Resposta da URL demasiado grande.")
        resp._content = content
        return resp
