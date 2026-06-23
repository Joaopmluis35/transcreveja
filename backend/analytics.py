"""Registo simples de visitas ao site (page views)."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import date, datetime, timedelta


def _db_path() -> str:
    return "ouviescrevi.db"


def _day_str(when: date | None = None) -> str:
    return (when or date.today()).isoformat()


def record_visit(path: str, client_ip: str) -> None:
    path = (path or "/").strip()[:500] or "/"
    day = _day_str()
    visitor_hash = hashlib.sha256(f"{client_ip}|{day}".encode()).hexdigest()[:32]
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO visitas (path, day, visitor_hash, created_at) VALUES (?, ?, ?, ?)",
            (path, day, visitor_hash, now),
        )
        conn.commit()
    finally:
        conn.close()


def _count_visits(since_day: str | None = None) -> int:
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        if since_day:
            cur.execute("SELECT COUNT(*) FROM visitas WHERE day >= ?", (since_day,))
        else:
            cur.execute("SELECT COUNT(*) FROM visitas")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _count_unique_visitors(since_day: str) -> int:
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(DISTINCT visitor_hash) FROM visitas WHERE day >= ?",
            (since_day,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def get_visit_stats() -> dict:
    today = date.today()
    day_7 = (today - timedelta(days=6)).isoformat()
    day_30 = (today - timedelta(days=29)).isoformat()
    today_s = _day_str(today)
    return {
        "visitas_hoje": _count_visits(today_s),
        "visitantes_unicos_hoje": _count_unique_visitors(today_s),
        "visitas_7_dias": _count_visits(day_7),
        "visitantes_unicos_7_dias": _count_unique_visitors(day_7),
        "visitas_30_dias": _count_visits(day_30),
        "visitas_total": _count_visits(),
    }


def get_recent_visits(limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT path, day, created_at
            FROM visitas
            ORDER BY id DESC
            LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        )
        return [
            {"path": row[0], "day": row[1], "created_at": row[2]}
            for row in cur.fetchall()
        ]
    finally:
        conn.close()


def _fill_daily_series(rows: list[tuple[str, int]], days: int) -> list[dict]:
    """Preenche dias em falta com zero."""
    today = date.today()
    start = today - timedelta(days=days - 1)
    by_day = {row[0]: int(row[1]) for row in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        out.append({"day": d, "total": by_day.get(d, 0)})
    return out


def get_daily_visit_series(days: int = 14) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT day, COUNT(*) AS total
            FROM visitas
            WHERE day >= ?
            GROUP BY day
            ORDER BY day
            """,
            (since,),
        )
        totals = _fill_daily_series(cur.fetchall(), days)
        cur.execute(
            """
            SELECT day, COUNT(DISTINCT visitor_hash) AS unicos
            FROM visitas
            WHERE day >= ?
            GROUP BY day
            ORDER BY day
            """,
            (since,),
        )
        unicos_map = {row[0]: int(row[1]) for row in cur.fetchall()}
        for item in totals:
            item["unicos"] = unicos_map.get(item["day"], 0)
        return totals
    finally:
        conn.close()


def get_daily_transcription_series(days: int = 14) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT substr(data, 1, 10) AS day, COUNT(*) AS total
            FROM transcricoes
            WHERE substr(data, 1, 10) >= ?
            GROUP BY day
            ORDER BY day
            """,
            (since,),
        )
        return _fill_daily_series(cur.fetchall(), days)
    finally:
        conn.close()


def get_top_pages(limit: int = 8) -> list[dict]:
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        since = (date.today() - timedelta(days=29)).isoformat()
        cur.execute(
            """
            SELECT path, COUNT(*) AS total
            FROM visitas
            WHERE day >= ?
            GROUP BY path
            ORDER BY total DESC
            LIMIT ?
            """,
            (since, max(1, min(limit, 20))),
        )
        return [{"path": row[0], "total": int(row[1])} for row in cur.fetchall()]
    finally:
        conn.close()
