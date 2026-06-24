"""Operações do backoffice: sessões, config, exportações, saúde."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import secrets
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from database import database_backend, db_path, get_connection, row_to_dict, use_turso

ROLE_LEVEL = {"viewer": 1, "editor": 2, "admin": 3}

DEFAULT_CONFIG: dict[str, str] = {
    "max_file_size_mb": "",
    "file_limit_message_pt": "Ficheiro demasiado grande. O limite é {limit} MB.",
    "file_limit_message_en": "File too large. The limit is {limit} MB.",
    "alert_email_enabled": "0",
    "alert_email_to": "",
    "alert_transcriptions_daily": "50",
    "alert_visits_daily": "500",
    "cloudflare_zone_id": "",
    "cloudflare_api_token": "",
    "whisper_cost_per_minute_usd": "0.006",
    "deploy_note": "",
}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return f"{salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
    except ValueError:
        return False
    check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return secrets.compare_digest(check, digest)


def log_audit(actor: str, action: str, detail: str | None = None) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO audit_log (actor, action, detail, created_at) VALUES (?, ?, ?, ?)",
            (actor, action, detail, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def log_audit_login(actor: str) -> None:
    """Evita dezenas de entradas «login» quando o browser dispara o formulário várias vezes."""
    conn = get_connection()
    try:
        since = (datetime.now() - timedelta(seconds=90)).isoformat(timespec="seconds") + "Z"
        row = conn.execute(
            """
            SELECT id FROM audit_log
            WHERE actor = ? AND action = 'login' AND created_at >= ?
            ORDER BY id DESC LIMIT 1
            """,
            (actor, since),
        ).fetchone()
        if row:
            return
    finally:
        conn.close()
    log_audit(actor, "login")


def log_api_error(path: str, status_code: int, message: str, client_ip: str = "") -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO api_errors (path, status_code, message, client_ip, created_at) VALUES (?, ?, ?, ?, ?)",
            (path[:300], status_code, (message or "")[:2000], client_ip[:80], _now()),
        )
        conn.commit()
    finally:
        conn.close()


def create_session(username: str, role: str, hours: int = 12) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=hours)).isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO admin_sessions (token, username, role, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token, username, role, _now(), expires),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def resolve_session(token: str | None) -> dict[str, str] | None:
    if not token:
        return None
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT username, role, expires_at FROM admin_sessions WHERE token = ?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] < _now():
            conn.execute("DELETE FROM admin_sessions WHERE token = ?", (token,))
            conn.commit()
            return None
        return {"username": row["username"], "role": row["role"]}
    finally:
        conn.close()


def require_role(session: dict[str, str] | None, minimum: str) -> None:
    from fastapi import HTTPException

    if not session:
        raise HTTPException(status_code=403, detail="Acesso negado.")
    if ROLE_LEVEL.get(session["role"], 0) < ROLE_LEVEL.get(minimum, 99):
        raise HTTPException(status_code=403, detail="Permissão insuficiente.")


def ensure_default_admin(env_password: str) -> None:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM admin_users").fetchone()
        if row and int(row["c"]) > 0:
            return
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("admin", _hash_password(env_password), "admin", _now()),
        )
        conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> dict[str, str] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT username, password_hash, role FROM admin_users WHERE username = ?",
            (username,),
        ).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        return {"username": row["username"], "role": row["role"]}
    finally:
        conn.close()


def list_users() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, username, role, created_at FROM admin_users ORDER BY id"
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def create_user(username: str, password: str, role: str) -> dict:
    if role not in ROLE_LEVEL:
        role = "editor"
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, _hash_password(password), role, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"username": username, "role": role}


def delete_user(user_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM admin_users WHERE id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def get_config() -> dict[str, str]:
    out = dict(DEFAULT_CONFIG)
    conn = get_connection()
    try:
        for row in conn.execute("SELECT key, value FROM site_config"):
            out[row["key"]] = row["value"]
    finally:
        conn.close()
    return out


def set_config(updates: dict[str, str], actor: str = "admin") -> dict[str, str]:
    now = _now()
    conn = get_connection()
    try:
        for key, value in updates.items():
            if key not in DEFAULT_CONFIG and not key.startswith("custom_"):
                continue
            conn.execute(
                """
                INSERT INTO site_config (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, str(value), now),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(actor, "config_update", json.dumps(list(updates.keys())))
    return get_config()


def get_maintenance() -> dict:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT manutencao, maintenance_message, block_transcribe_only FROM status WHERE id = 1"
        ).fetchone()
        if not row:
            return {
                "manutencao": False,
                "maintenance_message": "",
                "block_transcribe_only": True,
            }
        return {
            "manutencao": bool(row["manutencao"]),
            "maintenance_message": row["maintenance_message"] or "",
            "block_transcribe_only": bool(row["block_transcribe_only"] if row["block_transcribe_only"] is not None else 1),
        }
    finally:
        conn.close()


def set_maintenance(
    manutencao: bool,
    message: str | None = None,
    block_transcribe_only: bool | None = None,
    actor: str = "admin",
) -> dict:
    conn = get_connection()
    try:
        fields = ["manutencao = ?"]
        params: list[Any] = [1 if manutencao else 0]
        if message is not None:
            fields.append("maintenance_message = ?")
            params.append(message)
        if block_transcribe_only is not None:
            fields.append("block_transcribe_only = ?")
            params.append(1 if block_transcribe_only else 0)
        params.append(1)
        conn.execute(f"UPDATE status SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()
    log_audit(actor, "maintenance", f"on={manutencao}")
    return get_maintenance()


def record_transcription(
    nome_ficheiro: str,
    *,
    language: str | None = None,
    size_bytes: int | None = None,
    duration_sec: float | None = None,
    processing_sec: float | None = None,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO transcricoes (
                ficheiro, data, language, size_bytes, duration_sec,
                processing_sec, status, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nome_ficheiro,
                datetime.now().isoformat(),
                language,
                size_bytes,
                duration_sec,
                processing_sec,
                status,
                (error_message or "")[:2000] or None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_transcriptions(
    *,
    q: str | None = None,
    status: str | None = None,
    language: str | None = None,
    duplicates_only: bool = False,
    day_from: str | None = None,
    day_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    clauses, params = _transcription_filters(
        q, status, language, duplicates_only, day_from, day_to
    )
    params.extend([limit, offset])
    conn = get_connection()
    try:
        rows = conn.execute(
            f"""
            SELECT id, ficheiro, data, language, size_bytes, duration_sec,
                   processing_sec, status, error_message
            FROM transcricoes
            WHERE {' AND '.join(clauses)}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        items = [row_to_dict(r) for r in rows]
        dup_map: dict[str, int] = {}
        if items:
            dup_rows = conn.execute(
                f"""
                SELECT ficheiro, COUNT(*) AS c
                FROM transcricoes
                WHERE {' AND '.join(clauses)}
                  AND ficheiro IS NOT NULL AND TRIM(ficheiro) != ''
                GROUP BY ficheiro
                HAVING c > 1
                """,
                params[:-2],
            ).fetchall()
            dup_map = {
                r["ficheiro"] if hasattr(r, "keys") else r[0]: int(
                    r["c"] if hasattr(r, "keys") else r[1]
                )
                for r in dup_rows
            }
        for item in items:
            name = item.get("ficheiro") or ""
            cnt = dup_map.get(name, 1)
            item["duplicate_count"] = cnt
            item["is_duplicate"] = cnt > 1
        return items
    finally:
        conn.close()


def _transcription_filters(
    q: str | None,
    status: str | None,
    language: str | None,
    duplicates_only: bool,
    day_from: str | None,
    day_to: str | None,
) -> tuple[list[str], list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if q:
        clauses.append("ficheiro LIKE ?")
        params.append(f"%{q}%")
    if status:
        clauses.append("COALESCE(status, 'ok') = ?")
        params.append(status)
    if language:
        clauses.append("COALESCE(language, 'auto') = ?")
        params.append(language)
    if duplicates_only:
        clauses.append(
            """ficheiro IN (
                SELECT ficheiro FROM transcricoes
                WHERE ficheiro IS NOT NULL AND TRIM(ficheiro) != ''
                GROUP BY ficheiro HAVING COUNT(*) > 1
            )"""
        )
    if day_from:
        clauses.append("substr(data, 1, 10) >= ?")
        params.append(day_from)
    if day_to:
        clauses.append("substr(data, 1, 10) <= ?")
        params.append(day_to)
    return clauses, params


def count_transcriptions(
    *,
    q: str | None = None,
    status: str | None = None,
    language: str | None = None,
    duplicates_only: bool = False,
    day_from: str | None = None,
    day_to: str | None = None,
) -> int:
    clauses, params = _transcription_filters(
        q, status, language, duplicates_only, day_from, day_to
    )
    conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT COUNT(*) AS c FROM transcricoes WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()
        return int(row["c"] if hasattr(row, "keys") else row[0])
    finally:
        conn.close()


def transcription_stats(
    *,
    q: str | None = None,
    status: str | None = None,
    language: str | None = None,
    duplicates_only: bool = False,
    day_from: str | None = None,
    day_to: str | None = None,
) -> dict[str, Any]:
    clauses, params = _transcription_filters(
        q, status, language, duplicates_only, day_from, day_to
    )
    conn = get_connection()
    try:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN COALESCE(status, 'ok') != 'ok' THEN 1 ELSE 0 END) AS falhas,
                ROUND(AVG(COALESCE(processing_sec, 0)), 2) AS media_proc_s,
                ROUND(AVG(COALESCE(duration_sec, 0)), 1) AS media_dur_s
            FROM transcricoes
            WHERE {' AND '.join(clauses)}
            """,
            params,
        ).fetchone()
        dup_groups = conn.execute(
            f"""
            SELECT COUNT(*) AS c FROM (
                SELECT ficheiro FROM transcricoes
                WHERE {' AND '.join(clauses)}
                  AND ficheiro IS NOT NULL AND TRIM(ficheiro) != ''
                GROUP BY ficheiro
                HAVING COUNT(*) > 1
            )
            """,
            params,
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "falhas": int(row["falhas"] or 0),
            "media_proc_s": float(row["media_proc_s"] or 0),
            "media_dur_s": float(row["media_dur_s"] or 0),
            "ficheiros_duplicados": int(
                dup_groups["c"] if hasattr(dup_groups, "keys") else dup_groups[0] or 0
            ),
        }
    finally:
        conn.close()


def add_suggestion(nome: str | None, mensagem: str, lang: str = "pt") -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sugestoes (nome, mensagem, lang, created_at) VALUES (?, ?, ?, ?)",
            (nome, mensagem, lang, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_suggestions(
    unread_only: bool = False,
    lang: str | None = None,
    limit: int = 100,
) -> list[dict]:
    conn = get_connection()
    try:
        clauses = []
        params: list[Any] = []
        if unread_only:
            clauses.append("lida = 0")
        if lang:
            clauses.append("COALESCE(lang, 'pt') = ?")
            params.append(lang)
        q = "SELECT * FROM sugestoes"
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, min(limit, 200)))
        rows = conn.execute(q, params).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def mark_suggestion_read(suggestion_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE sugestoes SET lida = 1 WHERE id = ?", (suggestion_id,))
        conn.commit()
    finally:
        conn.close()


def delete_suggestion(suggestion_id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM sugestoes WHERE id = ?", (suggestion_id,))
        conn.commit()
    finally:
        conn.close()


def send_test_alert_email(actor: str = "admin") -> dict[str, str]:
    import smtplib
    from email.message import EmailMessage

    cfg = get_config()
    to = cfg.get("alert_email_to") or os.getenv("SMTP_TO", "")
    if not to:
        return {"ok": False, "error": "Configure o email de alertas em Sistema."}
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not (smtp_user and smtp_password):
        return {"ok": False, "error": "SMTP_USER/SMTP_PASSWORD não configurados no servidor."}
    try:
        msg = EmailMessage()
        msg["Subject"] = "Teste de alertas — Ouviescrevi"
        msg["From"] = os.getenv("SMTP_FROM", "notificacoes@ouviescrevi.pt")
        msg["To"] = to
        msg.set_content(
            f"Email de teste enviado pelo backoffice ({actor}). "
            "Se recebeste isto, os alertas por email estão configurados corretamente."
        )
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as smtp:
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
        log_audit(actor, "alert_email_test", to)
        return {"ok": True, "to": to}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:200]}


def get_active_banner() -> dict | None:
    now = _now()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, texto, link FROM site_banners
            WHERE ativo = 1
              AND (starts_at IS NULL OR starts_at <= ?)
              AND (ends_at IS NULL OR ends_at >= ?)
            ORDER BY id DESC LIMIT 1
            """,
            (now, now),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_banners() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM site_banners ORDER BY id DESC").fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def save_banner(data: dict, actor: str = "admin") -> dict:
    now = _now()
    conn = get_connection()
    try:
        banner_id = data.get("id")
        if banner_id:
            conn.execute(
                """
                UPDATE site_banners SET texto=?, link=?, ativo=?, starts_at=?, ends_at=?, updated_at=?
                WHERE id=?
                """,
                (
                    data["texto"],
                    data.get("link"),
                    1 if data.get("ativo") else 0,
                    data.get("starts_at"),
                    data.get("ends_at"),
                    now,
                    banner_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO site_banners (texto, link, ativo, starts_at, ends_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    data["texto"],
                    data.get("link"),
                    1 if data.get("ativo") else 0,
                    data.get("starts_at"),
                    data.get("ends_at"),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(actor, "banner_save", str(data.get("id") or "new"))
    banners = list_banners()
    return banners[0] if banners else {}


def get_audit_log(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_api_errors(limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM api_errors ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def estimate_costs(config: dict | None = None) -> dict:
    config = config or get_config()
    rate = float(config.get("whisper_cost_per_minute_usd") or "0.006")
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(duration_sec), 0) AS secs,
                   COALESCE(SUM(CASE WHEN substr(data,1,10)=? THEN 1 ELSE 0 END), 0) AS hoje
            FROM transcricoes WHERE COALESCE(status,'ok')='ok'
            """,
            (date.today().isoformat(),),
        ).fetchone()
        mins = float(row["secs"] or 0) / 60.0
        return {
            "transcricoes_total": int(row["total"] or 0),
            "transcricoes_hoje": int(row["hoje"] or 0),
            "minutos_audio_total": round(mins, 2),
            "custo_estimado_usd": round(mins * rate, 4),
            "taxa_por_minuto_usd": rate,
        }
    finally:
        conn.close()


def conversion_stats() -> dict:
    conn = get_connection()
    try:
        hoje = date.today().isoformat()
        visitas = conn.execute(
            "SELECT COUNT(*) AS c FROM visitas WHERE day = ?", (hoje,)
        ).fetchone()["c"]
        trans = conn.execute(
            "SELECT COUNT(*) AS c FROM transcricoes WHERE substr(data,1,10) = ?", (hoje,)
        ).fetchone()["c"]
        rate = round((trans / visitas * 100), 2) if visitas else 0.0
        return {"visitas_hoje": visitas, "transcricoes_hoje": trans, "taxa_conversao_pct": rate}
    finally:
        conn.close()


def peak_hours(days: int = 7) -> list[dict]:
    since = (date.today() - timedelta(days=days - 1)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT substr(created_at, 12, 2) AS hora, COUNT(*) AS total
            FROM visitas WHERE day >= ?
            GROUP BY hora ORDER BY hora
            """,
            (since,),
        ).fetchall()
        return [{"hora": r["hora"], "total": int(r["total"])} for r in rows]
    finally:
        conn.close()


def top_referrers(limit: int = 8) -> list[dict]:
    since = (date.today() - timedelta(days=29)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(NULLIF(referrer,''), '(direto)') AS ref, COUNT(*) AS total
            FROM visitas WHERE day >= ?
            GROUP BY ref ORDER BY total DESC LIMIT ?
            """,
            (since, limit),
        ).fetchall()
        return [{"referrer": r["ref"], "total": int(r["total"])} for r in rows]
    finally:
        conn.close()


def device_breakdown() -> list[dict]:
    since = (date.today() - timedelta(days=29)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT COALESCE(device_type, 'desconhecido') AS device, COUNT(*) AS total
            FROM visitas WHERE day >= ?
            GROUP BY device ORDER BY total DESC
            """,
            (since,),
        ).fetchall()
        return [{"device": r["device"], "total": int(r["total"])} for r in rows]
    finally:
        conn.close()


def export_csv(table: str) -> str:
    conn = get_connection()
    buf = io.StringIO()
    try:
        if table == "visitas":
            rows = conn.execute(
                "SELECT path, day, referrer, device_type, created_at FROM visitas ORDER BY id DESC LIMIT 5000"
            ).fetchall()
            w = csv.writer(buf)
            w.writerow(["path", "day", "referrer", "device", "created_at"])
            for r in rows:
                w.writerow([r["path"], r["day"], r["referrer"], r["device_type"], r["created_at"]])
        elif table == "transcricoes":
            rows = conn.execute(
                """
                SELECT ficheiro, data, language, size_bytes, duration_sec,
                       processing_sec, status, error_message
                FROM transcricoes ORDER BY id DESC LIMIT 5000
                """
            ).fetchall()
            w = csv.writer(buf)
            w.writerow(["ficheiro", "data", "language", "size_bytes", "duration_sec", "processing_sec", "status", "error"])
            for r in rows:
                w.writerow(
                    [
                        r["ficheiro"],
                        r["data"],
                        r["language"],
                        r["size_bytes"],
                        r["duration_sec"],
                        r["processing_sec"],
                        r["status"],
                        r["error_message"],
                    ]
                )
        else:
            raise ValueError("Tabela inválida")
    finally:
        conn.close()
    return buf.getvalue()


def backup_json() -> dict:
    conn = get_connection()
    try:
        data: dict[str, Any] = {"exported_at": _now(), "db_path": db_path()}
        for table in (
            "transcricoes",
            "visitas",
            "site_content",
            "sugestoes",
            "site_config",
            "site_banners",
            "audit_log",
        ):
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            data[table] = [row_to_dict(r) for r in rows]
        return data
    finally:
        conn.close()


def system_health(openai_client=None) -> dict:
    import time

    health: dict[str, Any] = {
        "api": "ok",
        "database": "ok",
        "database_backend": database_backend(),
        "database_path": db_path(),
        "database_persistent": use_turso(),
        "database_bytes": None,
        "database_latency_ms": None,
        "openai": "unknown",
        "disk_free_mb": None,
        "app_env": os.getenv("APP_ENV", "development"),
        "public_api_base": os.getenv("PUBLIC_API_BASE", "").rstrip("/"),
        "checked_at": _now(),
        "table_counts": {},
        "last_transcription_at": None,
    }

    if use_turso():
        health["persistence_note"] = "Turso Cloud — os dados sobrevivem a redeploys no Render."
    else:
        health["persistence_note"] = (
            "SQLite local — no Render Free os dados podem perder-se em cada redeploy. "
            "Configura TURSO_DATABASE_URL + TURSO_AUTH_TOKEN ou faz backup regular."
        )

    try:
        t0 = time.monotonic()
        conn = get_connection()
        try:
            conn.execute("SELECT 1").fetchone()
            health["database_latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
            for table in (
                "transcricoes",
                "visitas",
                "site_content",
                "sugestoes",
                "admin_users",
                "audit_log",
            ):
                try:
                    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    health["table_counts"][table] = int(row[0]) if row else 0
                except Exception:
                    health["table_counts"][table] = None
            try:
                row = conn.execute("SELECT MAX(data) AS last FROM transcricoes").fetchone()
                last = row["last"] if hasattr(row, "keys") else row[0]
                health["last_transcription_at"] = last
            except sqlite3.Error:
                pass
        finally:
            conn.close()
    except Exception as exc:
        health["database"] = f"erro: {str(exc)[:120]}"

    if not use_turso():
        try:
            if os.path.exists(db_path()):
                health["database_bytes"] = os.path.getsize(db_path())
        except OSError:
            if health["database"] == "ok":
                health["database"] = "warn"
        try:
            import shutil

            health["disk_free_mb"] = round(
                shutil.disk_usage(os.path.dirname(os.path.abspath(db_path()))).free / 1_048_576,
                1,
            )
        except OSError:
            pass

    if openai_client:
        try:
            openai_client.models.list(timeout=8)
            health["openai"] = "ok"
        except Exception as exc:
            health["openai"] = f"erro: {str(exc)[:120]}"

    try:
        from cms import CONTENT_KEYS, get_page_schema

        pages = get_page_schema()
        locale_langs = ("es", "fr", "de")
        health["cms_locale_pages"] = len(
            [
                p
                for p in pages
                if p.get("lang") in locale_langs and p.get("category") != "seo"
            ]
        )
        health["cms_locale_seo_pages"] = len(
            [
                p
                for p in pages
                if p.get("lang") in locale_langs and p.get("category") == "seo"
            ]
        )
        health["cms_locale_keys"] = len(
            [
                k
                for k in CONTENT_KEYS
                if any(k.startswith(f"{lng}_") or k.endswith(f"_{lng}") for lng in locale_langs)
            ]
        )
        health["cms_locales_ready"] = (
            health["cms_locale_pages"] >= 21
            and health["cms_locale_seo_pages"] >= 16
            and "es_home_intro_html" in CONTENT_KEYS
            and "meta_home_title_es" in CONTENT_KEYS
            and "meta_home_title_en" in CONTENT_KEYS
        )
        health["cms_locales_note"] = (
            "API pronta para guardar conteúdo e SEO em ES/FR/DE."
            if health["cms_locales_ready"]
            else "API desatualizada — faz redeploy no Render para guardar textos ES/FR/DE."
        )
    except Exception as exc:
        health["cms_locales_ready"] = False
        health["cms_locales_note"] = f"Não foi possível verificar CMS: {str(exc)[:80]}"

    turso_url = os.getenv("TURSO_DATABASE_URL", "").strip()
    turso_token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    health["turso_url_set"] = bool(turso_url)
    health["turso_token_set"] = bool(turso_token)
    health["turso_url_valid"] = turso_url.startswith("libsql://") if turso_url else False
    health["turso_env_configured"] = bool(turso_url and turso_token)
    if not health["database_persistent"] and health["turso_env_configured"]:
        health["persistence_note"] = (
            "Variáveis TURSO_* existem no ambiente mas a API está em SQLite local — "
            "verifica o token e reinicia o serviço no Render."
        )

    return health


def maybe_send_alerts(transcricoes_hoje: int, visitas_hoje: int, send_fn) -> None:
    cfg = get_config()
    if cfg.get("alert_email_enabled") != "1":
        return
    to = cfg.get("alert_email_to") or os.getenv("SMTP_TO", "")
    if not to:
        return
    try:
        t_thresh = int(cfg.get("alert_transcriptions_daily") or "0")
        v_thresh = int(cfg.get("alert_visits_daily") or "0")
        if t_thresh and transcricoes_hoje >= t_thresh:
            send_fn(
                f"Alerta Ouviescrevi: {transcricoes_hoje} transcrições hoje (limite {t_thresh}).",
                "Alerta transcrições — Ouviescrevi",
            )
        if v_thresh and visitas_hoje >= v_thresh:
            send_fn(
                f"Alerta Ouviescrevi: {visitas_hoje} visitas hoje (limite {v_thresh}).",
                "Alerta visitas — Ouviescrevi",
            )
    except Exception:
        pass


def fetch_cloudflare_analytics() -> dict | None:
    cfg = get_config()
    zone = cfg.get("cloudflare_zone_id") or os.getenv("CF_ZONE_ID", "")
    token = cfg.get("cloudflare_api_token") or os.getenv("CF_API_TOKEN", "")
    if not zone or not token:
        return None
    try:
        import requests

        since = (date.today() - timedelta(days=6)).isoformat()
        until = date.today().isoformat()
        query = {
            "query": (
                "{ viewer { zones(filter: {zoneTag: $zone}) "
                "{ httpRequests1dGroups(limit: 7, filter: {date_geq: $since, date_leq: $until}) "
                "{ sum { requests } dimensions { date } } } } } }"
            ),
            "variables": {"zone": zone, "since": since, "until": until},
        }
        res = requests.post(
            "https://api.cloudflare.com/client/v4/graphql",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=query,
            timeout=12,
        )
        res.raise_for_status()
        return res.json()
    except Exception:
        return None
