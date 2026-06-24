"""Registo de visitas ao site (page views) com referrer e dispositivo."""
from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Any

from database import get_connection, row_to_dict, scalar_int


def _day_str(when: date | None = None) -> str:
    return (when or date.today()).isoformat()


def _device_type(user_agent: str | None) -> str:
    ua = (user_agent or "").lower()
    if not ua:
        return "desconhecido"
    if "mobile" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    if "tablet" in ua or "ipad" in ua:
        return "tablet"
    return "desktop"


def record_visit(
    path: str,
    client_ip: str,
    *,
    referrer: str | None = None,
    user_agent: str | None = None,
) -> None:
    path = (path or "/").strip()[:500] or "/"
    day = _day_str()
    visitor_hash = hashlib.sha256(f"{client_ip}|{day}".encode()).hexdigest()[:32]
    ref = (referrer or "")[:500] or None
    ua = (user_agent or "")[:500] or None
    device = _device_type(ua)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO visitas (path, day, visitor_hash, created_at, referrer, user_agent, device_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (path, day, visitor_hash, now, ref, ua, device),
        )
        conn.commit()
    finally:
        conn.close()


def _count_visits(since_day: str | None = None) -> int:
    conn = get_connection()
    try:
        if since_day:
            row = conn.execute("SELECT COUNT(*) FROM visitas WHERE day >= ?", (since_day,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM visitas").fetchone()
        return scalar_int(row, "COUNT(*)", index=0) if row else 0
    finally:
        conn.close()


def _count_visits_on_day(day: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) FROM visitas WHERE day = ?", (day,)).fetchone()
        return scalar_int(row, "COUNT(*)", index=0) if row else 0
    finally:
        conn.close()


def _count_unique_visitors(since_day: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT visitor_hash) FROM visitas WHERE day >= ?",
            (since_day,),
        ).fetchone()
        return scalar_int(row, "COUNT(*)", index=0) if row else 0
    finally:
        conn.close()


def _count_unique_visitors_on_day(day: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(DISTINCT visitor_hash) FROM visitas WHERE day = ?",
            (day,),
        ).fetchone()
        return scalar_int(row, "COUNT(*)", index=0) if row else 0
    finally:
        conn.close()


def get_visit_stats() -> dict:
    today = date.today()
    day_7 = (today - timedelta(days=6)).isoformat()
    day_30 = (today - timedelta(days=29)).isoformat()
    today_s = _day_str(today)
    return {
        "visitas_hoje": _count_visits_on_day(today_s),
        "visitantes_unicos_hoje": _count_unique_visitors_on_day(today_s),
        "visitas_7_dias": _count_visits(day_7),
        "visitantes_unicos_7_dias": _count_unique_visitors(day_7),
        "visitas_30_dias": _count_visits(day_30),
        "visitas_total": _count_visits(),
    }


def get_recent_visits(limit: int = 20) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT path, day, created_at, referrer, device_type
            FROM visitas ORDER BY id DESC LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
        return [
            row_to_dict(row)
            for row in rows
        ]
    finally:
        conn.close()


def _fill_daily_series(rows: list[tuple[str, int]], days: int) -> list[dict]:
    today = date.today()
    start = today - timedelta(days=days - 1)
    by_day = {row[0]: int(row[1]) for row in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        out.append({"day": d, "total": by_day.get(d, 0)})
    return out


def _pair_day_count(row: Any) -> tuple[str, int]:
    d = row_to_dict(row)
    day = str(d.get("day") or "")
    total = d.get("total", d.get("unicos", 0))
    try:
        count = int(total or 0)
    except (TypeError, ValueError):
        count = 0
    return day, count


def get_daily_visit_series(days: int = 14) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT day, COUNT(*) AS total FROM visitas
            WHERE day >= ? GROUP BY day ORDER BY day
            """,
            (since,),
        ).fetchall()
        totals = _fill_daily_series([_pair_day_count(r) for r in rows], days)
        unicos_rows = conn.execute(
            """
            SELECT day, COUNT(DISTINCT visitor_hash) AS unicos
            FROM visitas WHERE day >= ? GROUP BY day ORDER BY day
            """,
            (since,),
        ).fetchall()
        unicos_map = {day: count for day, count in (_pair_day_count(r) for r in unicos_rows)}
        for item in totals:
            item["unicos"] = unicos_map.get(item["day"], 0)
        return totals
    finally:
        conn.close()


def get_daily_transcription_series(days: int = 14) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT substr(data, 1, 10) AS day, COUNT(*) AS total
            FROM transcricoes WHERE substr(data, 1, 10) >= ?
            GROUP BY substr(data, 1, 10) ORDER BY day
            """,
            (since,),
        ).fetchall()
        return _fill_daily_series([_pair_day_count(r) for r in rows], days)
    finally:
        conn.close()


def get_daily_transcription_outcomes(days: int = 14) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT substr(data, 1, 10) AS day,
                   SUM(CASE WHEN LOWER(COALESCE(status, 'ok')) = 'ok' THEN 1 ELSE 0 END) AS ok,
                   SUM(CASE WHEN LOWER(COALESCE(status, 'ok')) != 'ok' THEN 1 ELSE 0 END) AS erros
            FROM transcricoes
            WHERE substr(data, 1, 10) >= ?
            GROUP BY substr(data, 1, 10) ORDER BY day
            """,
            (since,),
        ).fetchall()
        by_day: dict[str, dict[str, int]] = {}
        for row in rows:
            d = row_to_dict(row)
            day = str(d.get("day") or "")
            ok = int(d.get("ok") or 0)
            erros = int(d.get("erros") or 0)
            by_day[day] = {"ok": ok, "erros": erros}
        out: list[dict] = []
        for i in range(days):
            d = (date.today() - timedelta(days=days - 1 - i)).isoformat()
            item = by_day.get(d, {"ok": 0, "erros": 0})
            total = item["ok"] + item["erros"]
            out.append(
                {
                    "day": d,
                    "ok": item["ok"],
                    "erros": item["erros"],
                    "total": total,
                    "taxa_ok_pct": round(100 * item["ok"] / total, 1) if total else 0,
                }
            )
        return out
    finally:
        conn.close()


def get_top_pages(limit: int = 8) -> list[dict]:
    conn = get_connection()
    try:
        since = (date.today() - timedelta(days=29)).isoformat()
        rows = conn.execute(
            """
            SELECT path, COUNT(*) AS total FROM visitas
            WHERE day >= ? GROUP BY path ORDER BY total DESC LIMIT ?
            """,
            (since, max(1, min(limit, 20))),
        ).fetchall()
        return [{"path": d.get("path"), "total": int(d.get("total") or 0)} for d in (row_to_dict(row) for row in rows)]
    finally:
        conn.close()
