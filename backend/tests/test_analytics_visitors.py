"""Testes de analytics — visitantes distintos e IP mascarado."""
from __future__ import annotations

from analytics import (
    get_owner_traffic_today,
    get_visitor_breakdown,
    is_bot_ip_label,
    is_bot_visit,
    mask_ip_label,
    parse_owner_visitor_uids,
    record_visit,
    visitor_uid,
)


def test_mask_ip_label_ipv4():
    assert mask_ip_label("89.123.45.67") == "89.123.45.x"
    assert mask_ip_label("unknown") == "desconhecido"


def test_visitor_uid_stable():
    a = visitor_uid("203.0.113.10")
    b = visitor_uid("203.0.113.10")
    c = visitor_uid("203.0.113.11")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_parse_owner_visitor_uids():
    assert parse_owner_visitor_uids("abc, def ,ghi") == {"abc", "def", "ghi"}
    assert parse_owner_visitor_uids("") == set()


def test_bot_detection():
    assert is_bot_ip_label("66.249.79.x") is True
    assert is_bot_ip_label("89.123.45.x") is False
    assert is_bot_visit("89.123.45.x", "Mozilla/5.0") is False
    assert is_bot_visit("89.123.45.x", "Mozilla/5.0 (compatible; Googlebot/2.1)") is True


def test_record_visit_and_breakdown(client):
    """Smoke: grava visita e agrupa por visitor_uid."""
    uid_a = visitor_uid("198.51.100.20")
    uid_b = visitor_uid("198.51.100.99")
    record_visit("/index.html", "198.51.100.20", referrer="https://google.com", user_agent="Mozilla/5.0")
    record_visit("/precos.html", "198.51.100.20", referrer=None, user_agent="Mozilla/5.0")
    record_visit("/en/index.html", "198.51.100.99", referrer=None, user_agent="Mozilla/5.0 (iPhone)")
    record_visit("/robots.txt", "66.249.79.68", referrer=None, user_agent="Mozilla/5.0 (compatible; Googlebot/2.1)")

    rows = get_visitor_breakdown(days=7, limit=20, owner_uids={uid_a})
    ids = {r["visitor_id"] for r in rows}
    assert uid_a in ids
    assert uid_b in ids
    owner_row = next(r for r in rows if r["visitor_id"] == uid_a)
    other_row = next(r for r in rows if r["visitor_id"] == uid_b)
    assert owner_row["is_owner"] is True
    assert other_row["is_owner"] is False
    assert owner_row["pageviews"] >= 2
    assert owner_row["ip_label"] == "198.51.100.x"
    bot_rows = [r for r in rows if r.get("is_bot")]
    assert any(r.get("ip_label") == "66.249.79.x" for r in bot_rows)

    traf = get_owner_traffic_today({uid_a})
    assert traf["visitas_tuas_hoje"] >= 2
    assert traf["visitas_outros_hoje"] >= 1
