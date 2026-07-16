"""Registo de visitas ao site (page views) com referrer e dispositivo."""
from __future__ import annotations

import hashlib
import ipaddress
from datetime import date, datetime, timedelta
from typing import Any

from database import get_connection, row_to_dict, scalar_int


def _day_str(when: date | None = None) -> str:
    return (when or date.today()).isoformat()


def mask_ip_label(client_ip: str) -> str:
    """Etiqueta legível sem guardar IP completo (ex.: 89.123.45.x)."""
    ip = (client_ip or "unknown").strip()
    if not ip or ip == "unknown":
        return "desconhecido"
    try:
        addr = ipaddress.ip_address(ip)
        if isinstance(addr, ipaddress.IPv4Address):
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.x"
        parts = addr.exploded.split(":")
        return ":".join(parts[:3]) + ":…"
    except ValueError:
        return ip[:24] + "…" if len(ip) > 24 else ip


def visitor_uid(client_ip: str) -> str:
    """Identificador estável por IP — permite ver o mesmo visitante em dias diferentes."""
    ip = (client_ip or "unknown").strip()
    return hashlib.sha256(f"uid|{ip}".encode()).hexdigest()[:16]


def parse_owner_visitor_uids(raw: str | None) -> set[str]:
    return {part.strip() for part in (raw or "").split(",") if part.strip()}


_BOT_UA_MARKERS = (
    "googlebot",
    "bingbot",
    "yandexbot",
    "duckduckbot",
    "baiduspider",
    "facebookexternalhit",
    "twitterbot",
    "linkedinbot",
    "slackbot",
    "semrushbot",
    "ahrefsbot",
    "petalbot",
    "applebot",
    "bytespider",
    "gptbot",
    "claudebot",
)

_BOT_IP_PREFIXES = (
    "66.249.",
    "66.102.",
    "64.233.",
    "72.14.",
    "209.85.",
    "157.55.",
    "40.77.",
    "207.46.",
    "17.58.",
)


def is_bot_ip_label(ip_label: str | None) -> bool:
    label = (ip_label or "").strip().lower()
    if not label or label in ("—", "desconhecido", "legado"):
        return False
    return any(label.startswith(prefix) for prefix in _BOT_IP_PREFIXES)


def is_bot_user_agent(user_agent: str | None) -> bool:
    ua = (user_agent or "").lower()
    return any(marker in ua for marker in _BOT_UA_MARKERS)


def is_bot_visit(ip_label: str | None, user_agent: str | None) -> bool:
    return is_bot_ip_label(ip_label) or is_bot_user_agent(user_agent)


def is_legacy_ip_label(ip_label: str | None) -> bool:
    label = (ip_label or "").strip()
    return not label or label in ("—", "desconhecido")


def _bot_sql_condition() -> str:
    ua_checks = " OR ".join(
        f"LOWER(COALESCE(user_agent, '')) LIKE '%{marker}%'" for marker in _BOT_UA_MARKERS[:8]
    )
    ip_checks = " OR ".join(
        f"COALESCE(ip_label, '') LIKE '{prefix}%'" for prefix in _BOT_IP_PREFIXES[:6]
    )
    return f"({ua_checks} OR {ip_checks})"


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
    uid = visitor_uid(client_ip)
    visitor_hash = hashlib.sha256(f"{client_ip}|{day}".encode()).hexdigest()[:32]
    ip_label = mask_ip_label(client_ip)
    ref = (referrer or "")[:500] or None
    ua = (user_agent or "")[:500] or None
    device = _device_type(ua)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO visitas (path, day, visitor_hash, visitor_uid, ip_label, created_at, referrer, user_agent, device_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (path, day, visitor_hash, uid, ip_label, now, ref, ua, device),
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


def get_recent_visits(limit: int = 20, owner_uids: set[str] | None = None) -> list[dict]:
    owner_uids = owner_uids or set()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT path, day, created_at, referrer, device_type, visitor_uid, ip_label, user_agent
            FROM visitas ORDER BY id DESC LIMIT ?
            """,
            (max(1, min(limit, 100)),),
        ).fetchall()
        out = []
        for row in rows:
            item = row_to_dict(row)
            uid = item.get("visitor_uid") or ""
            ip_label = item.get("ip_label")
            item["is_owner"] = uid in owner_uids if uid else False
            item["is_legacy"] = is_legacy_ip_label(ip_label)
            item["is_bot"] = not item["is_owner"] and is_bot_visit(ip_label, item.get("user_agent"))
            item["visitor_label"] = uid[:8] if uid else "—"
            out.append(item)
        return out
    finally:
        conn.close()


def get_visitor_breakdown(days: int = 14, limit: int = 40, owner_uids: set[str] | None = None) -> list[dict]:
    """Agrupa visitas por visitante (IP estável mascarado) nos últimos N dias."""
    owner_uids = owner_uids or set()
    days = max(1, min(days, 90))
    limit = max(1, min(limit, 100))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
              COALESCE(NULLIF(visitor_uid, ''), visitor_hash, '?') AS visitor_id,
              MAX(ip_label) AS ip_label,
              MAX(user_agent) AS user_agent,
              COUNT(*) AS pageviews,
              COUNT(DISTINCT day) AS dias_ativos,
              MIN(created_at) AS first_seen,
              MAX(created_at) AS last_seen,
              MAX(device_type) AS device_type
            FROM visitas
            WHERE day >= ?
            GROUP BY COALESCE(NULLIF(visitor_uid, ''), visitor_hash, '?')
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()
        out = []
        for row in rows:
            item = row_to_dict(row)
            vid = str(item.get("visitor_id") or "")
            ip_label = item.get("ip_label")
            item["visitor_short"] = vid[:8] if vid else "—"
            item["is_owner"] = vid in owner_uids
            item["is_legacy"] = is_legacy_ip_label(ip_label)
            item["is_bot"] = not item["is_owner"] and is_bot_visit(ip_label, item.get("user_agent"))
            if item["is_owner"]:
                item["tipo"] = "equipa"
            elif item["is_bot"]:
                item["tipo"] = "bot"
            else:
                item["tipo"] = "outro"
            out.append(item)
        return out
    finally:
        conn.close()


def get_owner_traffic_today(owner_uids: set[str]) -> dict:
    """Separa visitas/visitantes de hoje entre IPs marcados como equipa e restantes."""
    today_s = _day_str()
    if not owner_uids:
        stats = get_visit_stats()
        return {
            "visitas_tuas_hoje": 0,
            "visitas_outros_hoje": stats.get("visitas_hoje", 0),
            "unicos_tuas_hoje": 0,
            "unicos_outros_hoje": stats.get("visitantes_unicos_hoje", 0),
        }
    placeholders = ",".join("?" for _ in owner_uids)
    params = [today_s, *owner_uids]
    conn = get_connection()
    try:
        total_row = conn.execute(
            "SELECT COUNT(*) AS c FROM visitas WHERE day = ?",
            (today_s,),
        ).fetchone()
        owner_row = conn.execute(
            f"SELECT COUNT(*) AS c FROM visitas WHERE day = ? AND visitor_uid IN ({placeholders})",
            params,
        ).fetchone()
        unicos_row = conn.execute(
            "SELECT COUNT(DISTINCT COALESCE(NULLIF(visitor_uid, ''), visitor_hash)) AS c FROM visitas WHERE day = ?",
            (today_s,),
        ).fetchone()
        owner_unicos_row = conn.execute(
            f"""
            SELECT COUNT(DISTINCT COALESCE(NULLIF(visitor_uid, ''), visitor_hash)) AS c
            FROM visitas WHERE day = ? AND visitor_uid IN ({placeholders})
            """,
            params,
        ).fetchone()
        total = scalar_int(total_row, "c", index=0) if total_row else 0
        owner = scalar_int(owner_row, "c", index=0) if owner_row else 0
        unicos = scalar_int(unicos_row, "c", index=0) if unicos_row else 0
        owner_unicos = scalar_int(owner_unicos_row, "c", index=0) if owner_unicos_row else 0
        return {
            "visitas_tuas_hoje": owner,
            "visitas_outros_hoje": max(0, total - owner),
            "unicos_tuas_hoje": owner_unicos,
            "unicos_outros_hoje": max(0, unicos - owner_unicos),
        }
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


def get_daily_visit_series(days: int = 14, owner_uids: set[str] | None = None) -> list[dict]:
    days = max(1, min(days, 90))
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    owner_uids = owner_uids or set()
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
        bot_cond = _bot_sql_condition()
        tuas_map: dict[str, int] = {}
        bots_map: dict[str, int] = {}
        if owner_uids:
            placeholders = ",".join("?" for _ in owner_uids)
            split_rows = conn.execute(
                f"""
                SELECT day,
                  SUM(CASE WHEN visitor_uid IN ({placeholders}) THEN 1 ELSE 0 END) AS tuas,
                  SUM(CASE WHEN {bot_cond} THEN 1 ELSE 0 END) AS bots
                FROM visitas
                WHERE day >= ?
                GROUP BY day
                """,
                (*owner_uids, since),
            ).fetchall()
        else:
            split_rows = conn.execute(
                f"""
                SELECT day,
                  0 AS tuas,
                  SUM(CASE WHEN {bot_cond} THEN 1 ELSE 0 END) AS bots
                FROM visitas
                WHERE day >= ?
                GROUP BY day
                """,
                (since,),
            ).fetchall()
        for row in split_rows:
            item = row_to_dict(row)
            day = str(item.get("day") or "")
            tuas_map[day] = int(item.get("tuas") or 0)
            bots_map[day] = int(item.get("bots") or 0)
        for item in totals:
            day = item["day"]
            total = int(item.get("total") or 0)
            tuas = tuas_map.get(day, 0)
            bots = bots_map.get(day, 0)
            item["unicos"] = unicos_map.get(day, 0)
            item["tuas"] = tuas
            item["bots"] = bots
            item["outros"] = max(0, total - tuas - bots)
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


def _scalar_count(conn: Any, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    d = row_to_dict(row) if row else {}
    try:
        return int(d.get("c") or 0)
    except (TypeError, ValueError):
        return 0


def build_visit_report(owner_uids: set[str] | None = None) -> dict[str, Any]:
    """Relatório compacto ontem+hoje para análise (export JSON no backoffice)."""
    import admin_store as store
    from database import database_backend, use_turso

    owner_uids = owner_uids or set()
    today = date.today()
    yesterday = today - timedelta(days=1)
    days = [yesterday.isoformat(), today.isoformat()]
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    conn = get_connection()
    try:
        by_day: dict[str, dict[str, int]] = {}
        for d in days:
            pageviews = _scalar_count(conn, "SELECT COUNT(*) AS c FROM visitas WHERE day=?", (d,))
            unicos = _scalar_count(
                conn,
                """
                SELECT COUNT(DISTINCT COALESCE(NULLIF(visitor_uid,''), visitor_hash)) AS c
                FROM visitas WHERE day=?
                """,
                (d,),
            )
            humans = bots = owner = legacy = 0
            for row in conn.execute(
                "SELECT visitor_uid, ip_label, user_agent FROM visitas WHERE day=?",
                (d,),
            ).fetchall():
                item = row_to_dict(row)
                uid = item.get("visitor_uid") or ""
                ip = item.get("ip_label")
                is_owner = uid in owner_uids if uid else False
                is_bot = (not is_owner) and is_bot_visit(ip, item.get("user_agent"))
                if is_owner:
                    owner += 1
                elif is_bot:
                    bots += 1
                else:
                    humans += 1
                if is_legacy_ip_label(ip):
                    legacy += 1
            by_day[d] = {
                "pageviews": pageviews,
                "unicos": unicos,
                "human_pageviews": humans,
                "bot_pageviews": bots,
                "owner_pageviews": owner,
                "legacy_pageviews": legacy,
            }

        unicos_2d = _scalar_count(
            conn,
            """
            SELECT COUNT(DISTINCT COALESCE(NULLIF(visitor_uid,''), visitor_hash)) AS c
            FROM visitas WHERE day IN (?,?)
            """,
            (days[0], days[1]),
        )
        humans_unicos_2d = 0
        for row in conn.execute(
            """
            SELECT COALESCE(NULLIF(visitor_uid,''), visitor_hash,'?') AS vid,
                   MAX(ip_label) AS ip_label, MAX(user_agent) AS user_agent,
                   MAX(visitor_uid) AS visitor_uid
            FROM visitas WHERE day IN (?,?)
            GROUP BY COALESCE(NULLIF(visitor_uid,''), visitor_hash,'?')
            """,
            (days[0], days[1]),
        ).fetchall():
            item = row_to_dict(row)
            uid = item.get("visitor_uid") or ""
            if uid in owner_uids:
                continue
            if is_bot_visit(item.get("ip_label"), item.get("user_agent")):
                continue
            humans_unicos_2d += 1

        pages = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT day, path, COUNT(*) AS c FROM visitas
                WHERE day IN (?,?) GROUP BY day, path ORDER BY day, c DESC
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        devices = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT day, COALESCE(device_type,'?') AS device_type, COUNT(*) AS c
                FROM visitas WHERE day IN (?,?) GROUP BY day, device_type ORDER BY day, c DESC
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        refs = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT day, COALESCE(NULLIF(referrer,''),'(direct)') AS referrer, COUNT(*) AS c
                FROM visitas WHERE day IN (?,?) GROUP BY day, referrer ORDER BY day, c DESC
                LIMIT 40
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        hours = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT day, substr(created_at,12,2) AS hora, COUNT(*) AS c
                FROM visitas WHERE day IN (?,?) GROUP BY day, hora ORDER BY day, hora
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        locales = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT day,
                  CASE
                    WHEN path LIKE '/en/%' OR path='/en' THEN 'en'
                    WHEN path LIKE '/es/%' OR path='/es' THEN 'es'
                    WHEN path LIKE '/fr/%' OR path='/fr' THEN 'fr'
                    WHEN path LIKE '/de/%' OR path='/de' THEN 'de'
                    ELSE 'pt'
                  END AS locale,
                  COUNT(*) AS c
                FROM visitas WHERE day IN (?,?)
                GROUP BY day, locale ORDER BY day, c DESC
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        visitors = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT COALESCE(NULLIF(visitor_uid,''), visitor_hash,'?') AS visitor_id,
                       MAX(ip_label) AS ip_label, MAX(user_agent) AS user_agent,
                       COUNT(*) AS pageviews, COUNT(DISTINCT day) AS dias,
                       MIN(created_at) AS first_seen, MAX(created_at) AS last_seen,
                       MAX(device_type) AS device_type
                FROM visitas WHERE day IN (?,?)
                GROUP BY COALESCE(NULLIF(visitor_uid,''), visitor_hash,'?')
                ORDER BY pageviews DESC LIMIT 40
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
        for v in visitors:
            uid = str(v.get("visitor_id") or "")
            v["is_owner"] = uid in owner_uids
            v["is_legacy"] = is_legacy_ip_label(v.get("ip_label"))
            v["is_bot"] = (not v["is_owner"]) and is_bot_visit(v.get("ip_label"), v.get("user_agent"))
            v["user_agent"] = (v.get("user_agent") or "")[:80]
            v["visitor_id"] = uid[:8] + "…" if len(uid) > 8 else uid

        trans = [
            row_to_dict(r)
            for r in conn.execute(
                """
                SELECT substr(data,1,10) AS day, COUNT(*) AS c
                FROM transcricoes WHERE substr(data,1,10) IN (?,?) GROUP BY day
                """,
                (days[0], days[1]),
            ).fetchall()
        ]
    finally:
        conn.close()

    series = get_daily_visit_series(14, owner_uids)
    breakdown = get_visitor_breakdown(2, 40, owner_uids)
    try:
        conv_locale = store.conversion_by_locale(14)
    except Exception:
        conv_locale = []

    return {
        "exported_at": now,
        "purpose": "Análise ontem vs hoje — partilhar este JSON no chat Cursor",
        "source": {
            "database_backend": database_backend(),
            "use_turso": use_turso(),
            "note": "turso production" if use_turso() else "local sqlite",
        },
        "range": {"ontem": days[0], "hoje": days[1]},
        "by_day": by_day,
        "totals_2d": {
            "pageviews": sum(v["pageviews"] for v in by_day.values()),
            "unicos": unicos_2d,
            "human_unicos_approx": humans_unicos_2d,
            "human_pageviews": sum(v["human_pageviews"] for v in by_day.values()),
            "bot_pageviews": sum(v["bot_pageviews"] for v in by_day.values()),
            "owner_pageviews": sum(v["owner_pageviews"] for v in by_day.values()),
        },
        "visitas": get_visit_stats(),
        "trafego_hoje": get_owner_traffic_today(owner_uids),
        "conversao_hoje": store.conversion_stats(),
        "conversao_por_idioma_14d": conv_locale,
        "pages": pages,
        "devices": devices,
        "referrers": refs,
        "hours": hours,
        "locales": locales,
        "visitors_top": visitors,
        "transcriptions": trans,
        "series_14d": series,
        "breakdown_2d": [
            {
                "tipo": b.get("tipo"),
                "ip_label": b.get("ip_label"),
                "pageviews": b.get("pageviews"),
                "dias_ativos": b.get("dias_ativos"),
                "device_type": b.get("device_type"),
                "last_seen": b.get("last_seen"),
                "is_owner": b.get("is_owner"),
                "is_bot": b.get("is_bot"),
                "is_legacy": b.get("is_legacy"),
            }
            for b in breakdown
        ],
        "top_pages_30d": get_top_pages(10),
        "owner_uids_count": len(owner_uids),
        "owner_ip_labels": store.get_owner_ip_labels_list(),
    }
