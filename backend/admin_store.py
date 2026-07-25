"""Operações do backoffice: sessões, config, exportações, saúde."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from database import database_backend, db_path, get_connection, row_to_dict, scalar_float, scalar_int, use_turso

ROLE_LEVEL = {"viewer": 1, "editor": 2, "admin": 3}

_QUOTA_CONFIG_KEYS = frozenset({"quota_anonymous_daily", "quota_registered_daily", "pro_quota_daily"})


def _normalize_config_value(key: str, value: str) -> str:
    if key in _QUOTA_CONFIG_KEYS:
        try:
            return str(max(0, int(str(value).strip() or "0")))
        except ValueError:
            return "0"
    return str(value)


DEFAULT_CONFIG: dict[str, str] = {
    "max_file_size_mb": "",
    "file_limit_message_pt": "Ficheiro demasiado grande. O limite é {limit} MB.",
    "file_limit_message_en": "File too large. The limit is {limit} MB.",
    "alert_email_enabled": "0",
    "alert_email_to": "",
    "notify_activity_enabled": "1",
    "alert_transcriptions_daily": "50",
    "alert_visits_daily": "500",
    "owner_visitor_uids": "",
    "owner_ip_labels": "",
    "quota_anonymous_daily": "3",
    "quota_registered_daily": "20",
    "billing_enabled": "0",
    "pricing_hidden": "1",
    "stripe_public_key": "",
    "stripe_secret_key": "",
    "stripe_webhook_secret": "",
    "stripe_price_id_pro": "",
    "pro_quota_daily": "200",
    "pro_price_label": "9,99 €/mês",
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


def register_site_user(
    email: str,
    password: str,
    name: str | None = None,
    *,
    marketing_opt_in: bool = False,
) -> dict[str, str]:
    email = email.strip().lower()
    if not email or "@" not in email:
        raise ValueError("Email inválido.")
    if len(password or "") < 8:
        raise ValueError("A palavra-passe deve ter pelo menos 8 caracteres.")
    conn = get_connection()
    try:
        existing = conn.execute("SELECT id FROM site_users WHERE email = ?", (email,)).fetchone()
        if existing:
            raise ValueError("Já existe uma conta com este email.")
        display = (name or "").strip() or None
        conn.execute(
            """
            INSERT INTO site_users (email, password_hash, name, created_at, marketing_opt_in)
            VALUES (?, ?, ?, ?, ?)
            """,
            (email, _hash_password(password), display, _now(), 1 if marketing_opt_in else 0),
        )
        conn.commit()
    finally:
        conn.close()
    return {"email": email, "name": display, "marketing_opt_in": marketing_opt_in}


def set_marketing_opt_in(email: str, opt_in: bool) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE site_users SET marketing_opt_in = ? WHERE email = ?",
            (1 if opt_in else 0, email.strip().lower()),
        )
        conn.commit()
    finally:
        conn.close()


def list_marketing_opt_in_emails(limit: int = 500) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT email, name FROM site_users
            WHERE marketing_opt_in = 1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 2000)),),
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def create_password_reset_token(email: str, *, hours: int = 2) -> str | None:
    email = email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute("SELECT email FROM site_users WHERE email = ?", (email,)).fetchone()
        if not row:
            return None
        token = secrets.token_urlsafe(32)
        now = datetime.utcnow()
        expires = (now + timedelta(hours=max(1, hours))).isoformat(timespec="seconds") + "Z"
        conn.execute(
            """
            INSERT INTO password_reset_tokens (token, email, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, email, now.isoformat(timespec="seconds") + "Z", expires),
        )
        conn.commit()
        return token
    finally:
        conn.close()


def reset_password_with_token(token: str, new_password: str) -> bool:
    if len(new_password or "") < 8:
        raise ValueError("A palavra-passe deve ter pelo menos 8 caracteres.")
    token = (token or "").strip()
    if not token:
        return False
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT email, expires_at, used_at FROM password_reset_tokens WHERE token = ?
            """,
            (token,),
        ).fetchone()
        if not row or row["used_at"]:
            return False
        if str(row["expires_at"] or "") < now:
            return False
        email = row["email"]
        conn.execute(
            "UPDATE site_users SET password_hash = ? WHERE email = ?",
            (_hash_password(new_password), email),
        )
        conn.execute(
            "UPDATE password_reset_tokens SET used_at = ? WHERE token = ?",
            (now, token),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def create_shared_transcript(
    text: str,
    *,
    title: str | None = None,
    locale: str = "pt",
    days_valid: int = 30,
) -> dict:
    body = (text or "").strip()
    if len(body) < 20:
        raise ValueError("Texto demasiado curto para partilhar.")
    if len(body) > 100_000:
        body = body[:100_000]
    share_id = secrets.token_urlsafe(10)
    now = datetime.utcnow()
    expires = (now + timedelta(days=max(1, min(days_valid, 90)))).isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO shared_transcripts (id, title, text, locale, created_at, expires_at, view_count)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                share_id,
                (title or "Transcrição")[:120],
                body,
                (locale or "pt")[:8],
                now.isoformat(timespec="seconds") + "Z",
                expires,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": share_id, "expires_at": expires}


def get_shared_transcript(share_id: str) -> dict | None:
    sid = (share_id or "").strip()
    if not sid:
        return None
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, title, text, locale, created_at, expires_at, view_count FROM shared_transcripts WHERE id = ?",
            (sid,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] and str(row["expires_at"]) < now:
            return None
        conn.execute(
            "UPDATE shared_transcripts SET view_count = COALESCE(view_count,0) + 1 WHERE id = ?",
            (sid,),
        )
        conn.commit()
        item = row_to_dict(row)
        item["view_count"] = int(item.get("view_count") or 0) + 1
        return item
    finally:
        conn.close()


def authenticate_site_user(email: str, password: str) -> dict[str, str] | None:
    email = email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT email, password_hash, name FROM site_users WHERE email = ?",
            (email,),
        ).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        return {"email": row["email"], "name": row["name"]}
    finally:
        conn.close()


def _usage_day() -> str:
    return date.today().isoformat()


def _quota_limits() -> tuple[int, int]:
    cfg = get_config()
    anon = int(cfg.get("quota_anonymous_daily") or os.getenv("QUOTA_ANONYMOUS_DAILY", "3") or "0")
    reg = int(cfg.get("quota_registered_daily") or os.getenv("QUOTA_REGISTERED_DAILY", "20") or "0")
    return max(0, anon), max(0, reg)


def usage_key_for_request(request, actor: dict) -> tuple[str, str]:
    """Retorna (usage_key, tier) onde tier é anonymous|registered|staff."""
    actor_type = actor.get("type", "anonymous")
    if actor_type == "admin":
        return "", "staff"
    if actor_type == "user":
        email = (actor.get("email") or actor.get("username") or "").strip().lower()
        return f"user:{email}", "registered"
    from security import client_ip

    ip = client_ip(request)
    digest = hashlib.sha256(ip.encode("utf-8")).hexdigest()[:32]
    return f"anon:{digest}", "anonymous"


def get_daily_transcribe_count(usage_key: str) -> int:
    if not usage_key:
        return 0
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT transcribe_count FROM daily_usage WHERE usage_key = ? AND usage_day = ?",
            (usage_key, _usage_day()),
        ).fetchone()
        return int(row["transcribe_count"]) if row else 0
    finally:
        conn.close()


def increment_daily_transcribe(usage_key: str) -> int:
    if not usage_key:
        return 0
    now = _now()
    day = _usage_day()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO daily_usage (usage_key, usage_day, transcribe_count, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(usage_key, usage_day) DO UPDATE SET
                transcribe_count = transcribe_count + 1,
                updated_at = excluded.updated_at
            """,
            (usage_key, day, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT transcribe_count FROM daily_usage WHERE usage_key = ? AND usage_day = ?",
            (usage_key, day),
        ).fetchone()
        return int(row["transcribe_count"]) if row else 1
    finally:
        conn.close()


def transcribe_quota_status(request, actor: dict) -> dict:
    from billing import billing_enabled, get_user_plan, pricing_hidden, pro_quota_limit

    anon_limit, reg_limit = _quota_limits()
    actor_type = actor.get("type", "anonymous")
    if actor_type == "admin":
        return {
            "tier": "staff",
            "plan": "staff",
            "limit": 0,
            "used": 0,
            "remaining": None,
            "unlimited": True,
            "billing_enabled": billing_enabled(),
        }
    usage_key, tier = usage_key_for_request(request, actor)
    plan = "free"
    limit = reg_limit if tier == "registered" else anon_limit
    if tier == "registered":
        email = (actor.get("email") or "").strip().lower()
        if email and billing_enabled() and get_user_plan(email) == "pro":
            plan = "pro"
            limit = pro_quota_limit()
    used = get_daily_transcribe_count(usage_key)
    if limit <= 0:
        return {
            "tier": tier,
            "plan": plan,
            "limit": 0,
            "used": used,
            "remaining": None,
            "unlimited": True,
            "billing_enabled": billing_enabled(),
        }
    remaining = max(0, limit - used)
    out = {
        "tier": tier,
        "plan": plan,
        "limit": limit,
        "used": used,
        "remaining": remaining,
        "unlimited": False,
        "billing_enabled": billing_enabled(),
    }
    if remaining <= 0:
        if tier == "anonymous":
            out["message"] = (
                f"Limite diário atingido ({limit} transcrições). "
                "Cria uma conta gratuita para mais transcrições por dia."
            )
        elif plan != "pro":
            if billing_enabled() and not pricing_hidden():
                out["message"] = (
                    f"Limite diário atingido ({limit} transcrições). "
                    "Passa ao plano Pro para mais transcrições e exportação DOCX."
                )
            else:
                out["message"] = f"Limite diário atingido ({limit} transcrições). Tenta amanhã."
        else:
            out["message"] = f"Limite diário Pro atingido ({limit} transcrições). Tenta amanhã."
    return out


def save_user_transcription(
    user_email: str,
    *,
    filename: str | None = None,
    language: str | None = None,
    size_bytes: int | None = None,
    duration_sec: float | None = None,
    transcription: str | None = None,
    formatted: str | None = None,
    history_limit: int | None = None,
) -> int:
    email = user_email.strip().lower()
    now = _now()
    text = (transcription or "")[:200_000] or None
    fmt = (formatted or "")[:200_000] or None
    keep = history_limit if history_limit is not None else 100
    try:
        import billing as billing_mod

        if billing_mod.is_pro_user(email):
            keep = max(keep, 500)
    except Exception:
        pass
    keep = max(20, min(int(keep), 2000))
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO user_transcriptions (
                user_email, filename, language, size_bytes, duration_sec,
                transcription, formatted, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (email, filename, language, size_bytes, duration_sec, text, fmt, now),
        )
        new_id = int(cur.lastrowid)
        rows = conn.execute(
            "SELECT id FROM user_transcriptions WHERE user_email = ? ORDER BY id DESC",
            (email,),
        ).fetchall()
        if len(rows) > keep:
            drop_ids = [int(r["id"]) for r in rows[keep:]]
            placeholders = ",".join("?" * len(drop_ids))
            conn.execute(
                f"DELETE FROM user_transcriptions WHERE id IN ({placeholders})",
                drop_ids,
            )
        conn.commit()
        return new_id
    finally:
        conn.close()


def list_user_transcriptions(
    user_email: str,
    *,
    limit: int = 30,
    offset: int = 0,
    q: str | None = None,
) -> list[dict]:
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    email = user_email.strip().lower()
    needle = (q or "").strip()
    conn = get_connection()
    try:
        if needle:
            like = f"%{needle}%"
            rows = conn.execute(
                """
                SELECT id, filename, language, size_bytes, duration_sec, created_at,
                       substr(COALESCE(formatted, transcription, ''), 1, 160) AS preview
                FROM user_transcriptions
                WHERE user_email = ?
                  AND (
                    COALESCE(filename, '') LIKE ?
                    OR COALESCE(formatted, '') LIKE ?
                    OR COALESCE(transcription, '') LIKE ?
                  )
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (email, like, like, like, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, filename, language, size_bytes, duration_sec, created_at,
                       substr(COALESCE(formatted, transcription, ''), 1, 160) AS preview
                FROM user_transcriptions
                WHERE user_email = ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (email, limit, offset),
            ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def rename_user_transcription(user_email: str, item_id: int, filename: str) -> dict | None:
    email = user_email.strip().lower()
    name = (filename or "").strip()[:240] or "Sem nome"
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE user_transcriptions
            SET filename = ?
            WHERE user_email = ? AND id = ?
            """,
            (name, email, item_id),
        )
        conn.commit()
        if cur.rowcount <= 0:
            return None
    finally:
        conn.close()
    return get_user_transcription(email, item_id)


def get_user_transcription(user_email: str, item_id: int) -> dict | None:
    email = user_email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, filename, language, size_bytes, duration_sec,
                   transcription, formatted, created_at
            FROM user_transcriptions
            WHERE user_email = ? AND id = ?
            """,
            (email, item_id),
        ).fetchone()
        return row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_user_transcription(user_email: str, item_id: int) -> bool:
    email = user_email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM user_transcriptions WHERE user_email = ? AND id = ?",
            (email, item_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def save_user_correction(
    user_email: str,
    *,
    original_text: str,
    corrected_text: str,
    mode: str | None = None,
) -> int:
    email = user_email.strip().lower()
    now = _now()
    orig = (original_text or "")[:100_000]
    corr = (corrected_text or "")[:100_000]
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO user_corrections (
                user_email, original_text, corrected_text, mode, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (email, orig, corr, mode, now),
        )
        new_id = int(cur.lastrowid)
        rows = conn.execute(
            "SELECT id FROM user_corrections WHERE user_email = ? ORDER BY id DESC",
            (email,),
        ).fetchall()
        if len(rows) > 50:
            drop_ids = [int(r["id"]) for r in rows[50:]]
            placeholders = ",".join("?" * len(drop_ids))
            conn.execute(
                f"DELETE FROM user_corrections WHERE id IN ({placeholders})",
                drop_ids,
            )
        conn.commit()
        return new_id
    finally:
        conn.close()


def list_user_corrections(user_email: str, *, limit: int = 30, offset: int = 0) -> list[dict]:
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    email = user_email.strip().lower()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, mode, created_at,
                   substr(COALESCE(corrected_text, ''), 1, 160) AS preview
            FROM user_corrections
            WHERE user_email = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (email, limit, offset),
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_user_correction(user_email: str, item_id: int) -> dict | None:
    email = user_email.strip().lower()
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT id, original_text, corrected_text, mode, created_at
            FROM user_corrections
            WHERE user_email = ? AND id = ?
            """,
            (email, item_id),
        ).fetchone()
        return row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_user_correction(user_email: str, item_id: int) -> bool:
    email = user_email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM user_corrections WHERE user_email = ? AND id = ?",
            (email, item_id),
        )
        conn.commit()
        return cur.rowcount > 0
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
    username = (username or "").strip()
    if not username:
        raise ValueError("username_required")
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT id FROM admin_users WHERE username = ?",
            (username,),
        ).fetchone()
        if existing:
            raise ValueError("username_exists")
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


def update_user_role(user_id: int, role: str) -> dict:
    if role not in ROLE_LEVEL:
        raise ValueError("invalid_role")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, username, role, created_at FROM admin_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            raise ValueError("not_found")
        if row["role"] == "admin" and role != "admin":
            admin_count = scalar_int(
                conn.execute("SELECT COUNT(*) AS c FROM admin_users WHERE role = 'admin'").fetchone(),
                "c",
            )
            if admin_count <= 1:
                raise ValueError("last_admin")
        conn.execute("UPDATE admin_users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()
        updated = conn.execute(
            "SELECT id, username, role, created_at FROM admin_users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return row_to_dict(updated) if updated else {}
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
    if "alert_email_to" in updates:
        updates["alert_email_to"] = re.sub(r"\s+", "", (updates.get("alert_email_to") or "").strip())
    conn = get_connection()
    try:
        for key, value in updates.items():
            if key not in DEFAULT_CONFIG and not key.startswith("custom_"):
                continue
            normalized = _normalize_config_value(key, str(value))
            conn.execute(
                """
                INSERT INTO site_config (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, normalized, now),
            )
        conn.commit()
    finally:
        conn.close()
    log_audit(actor, "config_update", json.dumps(list(updates.keys())))
    return get_config()


def parse_owner_ip_labels(raw: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        uid, label = part.split(":", 1)
        uid = uid.strip()
        label = label.strip()
        if uid and label:
            out[uid] = label
    return out


def serialize_owner_ip_labels(mapping: dict[str, str]) -> str:
    return ",".join(f"{uid}:{label}" for uid, label in sorted(mapping.items()))


def add_owner_visitor_uid(uid: str, actor: str = "admin", ip_label: str = "") -> dict[str, str]:
    uid = (uid or "").strip()
    if not uid:
        return get_config()
    cfg = get_config()
    existing = {part.strip() for part in (cfg.get("owner_visitor_uids") or "").split(",") if part.strip()}
    existing.add(uid)
    labels = parse_owner_ip_labels(cfg.get("owner_ip_labels"))
    if ip_label:
        labels[uid] = ip_label
    return set_config(
        {
            "owner_visitor_uids": ",".join(sorted(existing)),
            "owner_ip_labels": serialize_owner_ip_labels(labels),
        },
        actor,
    )


def remove_owner_visitor_uid(uid: str, actor: str = "admin") -> dict[str, str]:
    uid = (uid or "").strip()
    cfg = get_config()
    existing = {part.strip() for part in (cfg.get("owner_visitor_uids") or "").split(",") if part.strip()}
    existing.discard(uid)
    labels = parse_owner_ip_labels(cfg.get("owner_ip_labels"))
    labels.pop(uid, None)
    return set_config(
        {
            "owner_visitor_uids": ",".join(sorted(existing)),
            "owner_ip_labels": serialize_owner_ip_labels(labels),
        },
        actor,
    )


def get_owner_ip_labels_list(cfg: dict[str, str] | None = None) -> list[str]:
    cfg = cfg or get_config()
    labels = parse_owner_ip_labels(cfg.get("owner_ip_labels"))
    return sorted({label for label in labels.values() if label})


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
    ui_locale: str | None = None,
    page_path: str | None = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO transcricoes (
                ficheiro, data, language, size_bytes, duration_sec,
                processing_sec, status, error_message, ui_locale, page_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                (ui_locale or "")[:16] or None,
                (page_path or "")[:500] or None,
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
                HAVING COUNT(*) > 1
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
        return int(scalar_int(row, "c", index=0))
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
            "total": scalar_int(row, "total", index=0),
            "falhas": scalar_int(row, "falhas", index=1),
            "media_proc_s": scalar_float(row, "media_proc_s", index=2),
            "media_dur_s": scalar_float(row, "media_dur_s", index=3),
            "ficheiros_duplicados": scalar_int(dup_groups, "c", index=0),
        }
    finally:
        conn.close()


def count_site_users() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM site_users").fetchone()
        return int(row["c"]) if row else 0
    finally:
        conn.close()


def count_site_users_today() -> int:
    today = date.today().isoformat() + "T00:00:00Z"
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM site_users WHERE created_at >= ?",
            (today,),
        ).fetchone()
        return int(row["c"]) if row else 0
    finally:
        conn.close()


def count_email_failures_since(since_iso: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM email_notifications
            WHERE status = 'failed' AND created_at >= ?
            """,
            (since_iso,),
        ).fetchone()
        return int(row["c"]) if row else 0
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


def save_ai_insights(
    suggestions: list[dict],
    *,
    run_id: str,
    source_days: int,
) -> list[dict]:
    now = _now()
    saved: list[dict] = []
    conn = get_connection()
    try:
        for item in suggestions:
            cur = conn.execute(
                """
                INSERT INTO ai_insights (
                    run_id, title, detail, priority, category, evidence,
                    cursor_prompt, status, source_days, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, ?)
                """,
                (
                    run_id,
                    item.get("title"),
                    item.get("detail"),
                    item.get("priority") or "media",
                    item.get("category") or "produto",
                    item.get("evidence") or "",
                    item.get("cursor_prompt") or "",
                    int(source_days),
                    now,
                    now,
                ),
            )
            saved.append(
                {
                    "id": int(cur.lastrowid),
                    "run_id": run_id,
                    "title": item.get("title"),
                    "detail": item.get("detail"),
                    "priority": item.get("priority") or "media",
                    "category": item.get("category") or "produto",
                    "evidence": item.get("evidence") or "",
                    "cursor_prompt": item.get("cursor_prompt") or "",
                    "status": "new",
                    "source_days": int(source_days),
                    "created_at": now,
                    "updated_at": now,
                }
            )
        conn.commit()
        return saved
    finally:
        conn.close()


def list_ai_insights(
    *,
    status: str | None = None,
    limit: int = 50,
) -> list[dict]:
    limit = max(1, min(int(limit or 50), 200))
    conn = get_connection()
    try:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM ai_insights
                WHERE status = ?
                ORDER BY id DESC LIMIT ?
                """,
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM ai_insights ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


def update_ai_insight_status(item_id: int, status: str) -> dict | None:
    allowed = {"new", "saved", "done", "dismissed"}
    st = (status or "").strip().lower()
    if st not in allowed:
        raise ValueError("status_invalido")
    now = _now()
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE ai_insights SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (st, now, item_id),
        )
        conn.commit()
        if cur.rowcount <= 0:
            return None
        row = conn.execute("SELECT * FROM ai_insights WHERE id = ?", (item_id,)).fetchone()
        return row_to_dict(row) if row else None
    finally:
        conn.close()


def delete_ai_insight(item_id: int) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM ai_insights WHERE id = ?", (item_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def send_test_alert_email(actor: str = "admin") -> dict[str, str]:
    from email_notify import notify_email_to, send_notification_email

    to = get_config().get("alert_email_to") or notify_email_to()
    if not to:
        return {"ok": False, "error": "Configure o email de destino no separador Emails."}
    ok, err = send_notification_email(
        f"Email de teste enviado pelo backoffice ({actor}). "
        "Se recebeste isto, os alertas por email estão configurados corretamente.",
        "Teste de alertas — Ouviescrevi",
        to=to,
        kind="alert_test",
        actor=actor,
    )
    if ok:
        log_audit(actor, "alert_email_test", to)
        return {"ok": True, "to": to}
    return {
        "ok": False,
        "error": err or "Falha ao enviar. Configura RESEND_API_KEY no Render.",
    }


def send_test_activity_email(actor: str = "admin") -> dict[str, str]:
    from email_notify import notify_email_to, send_notification_email

    to = get_config().get("alert_email_to") or notify_email_to()
    if not to:
        return {"ok": False, "error": "Configure o email de destino no separador Emails."}
    ok, err = send_notification_email(
        f"Teste de notificação de atividade (transcrição/IA).\n\nConta: admin:{actor}",
        "Teste de atividade — Ouviescrevi",
        to=to,
        kind="activity_test",
        actor=actor,
    )
    if ok:
        log_audit(actor, "activity_email_test", to)
        return {"ok": True, "to": to}
    return {
        "ok": False,
        "error": err or "Falha ao enviar. Configura RESEND_API_KEY no Render.",
    }


def log_email_notification(
    kind: str,
    recipient: str,
    subject: str,
    status: str,
    *,
    detail: str | None = None,
    actor: str | None = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO email_notifications (kind, recipient, subject, status, detail, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (kind[:40], recipient[:200], subject[:300], status[:20], detail, actor, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def list_email_notifications(limit: int = 50) -> list[dict]:
    limit = max(1, min(int(limit), 200))
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, kind, recipient, subject, status, detail, actor, created_at
            FROM email_notifications
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [row_to_dict(r) for r in rows]
    finally:
        conn.close()


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
        return row_to_dict(row) if row else None
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


def count_api_errors_since(since_iso: str) -> int:
    since = (since_iso or "").strip()
    if not since:
        return 0
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM api_errors WHERE created_at >= ?",
            (since,),
        ).fetchone()
        return int(scalar_int(row, "c", index=0))
    finally:
        conn.close()


def estimate_costs(config: dict | None = None) -> dict:
    config = config or get_config()
    raw_rate = config.get("whisper_cost_per_minute_usd")
    try:
        rate = float(raw_rate) if raw_rate not in (None, "") else 0.006
    except (TypeError, ValueError):
        rate = 0.006
    if rate <= 0:
        rate = 0.006
    conn = get_connection()
    try:
        today_s = date.today().isoformat()
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(COALESCE(duration_sec, 0)), 0) AS dur_secs,
                   COALESCE(SUM(CASE WHEN substr(data,1,10)=? THEN 1 ELSE 0 END), 0) AS hoje,
                   COALESCE(SUM(
                     CASE
                       WHEN COALESCE(duration_sec, 0) > 0 THEN duration_sec
                       WHEN COALESCE(size_bytes, 0) > 500000 THEN size_bytes / 16000.0
                       ELSE 90
                     END
                   ), 0) AS est_secs
            FROM transcricoes
            WHERE LOWER(COALESCE(status, 'ok')) = 'ok'
            """,
            (today_s,),
        ).fetchone()
        dur_secs = scalar_float(row, "dur_secs", index=1)
        est_secs = scalar_float(row, "est_secs", index=3)
        secs = dur_secs if dur_secs > 0 else est_secs
        mins = secs / 60.0
        estimated = dur_secs <= 0 and est_secs > 0
        return {
            "transcricoes_total": scalar_int(row, "total", index=0),
            "transcricoes_hoje": scalar_int(row, "hoje", index=2),
            "minutos_audio_total": round(mins, 2),
            "custo_estimado_usd": round(mins * rate, 4),
            "taxa_por_minuto_usd": rate,
            "custo_estimado": estimated,
        }
    finally:
        conn.close()


def conversion_stats() -> dict:
    conn = get_connection()
    try:
        hoje = date.today().isoformat()
        visitas = scalar_int(
            conn.execute("SELECT COUNT(*) AS c FROM visitas WHERE day = ?", (hoje,)).fetchone(),
            "c",
            index=0,
        )
        trans = scalar_int(
            conn.execute(
                "SELECT COUNT(*) AS c FROM transcricoes WHERE substr(data,1,10) = ?", (hoje,)
            ).fetchone(),
            "c",
            index=0,
        )
        rate = round((trans / visitas * 100), 2) if visitas else 0.0
        return {"visitas_hoje": visitas, "transcricoes_hoje": trans, "taxa_conversao_pct": rate}
    finally:
        conn.close()


def _locale_from_path_sql(path_col: str = "path") -> str:
    """SQL CASE expression mapping URL path → ui locale code."""
    return f"""
    CASE
      WHEN {path_col} LIKE '/en/%' OR {path_col} = '/en' OR {path_col} LIKE '/en' THEN 'en'
      WHEN {path_col} LIKE '/es/%' OR {path_col} = '/es' OR {path_col} LIKE '/es' THEN 'es'
      WHEN {path_col} LIKE '/fr/%' OR {path_col} = '/fr' OR {path_col} LIKE '/fr' THEN 'fr'
      WHEN {path_col} LIKE '/de/%' OR {path_col} = '/de' OR {path_col} LIKE '/de' THEN 'de'
      ELSE 'pt'
    END
    """


def conversion_by_locale(days: int = 14) -> list[dict]:
    """Visitas (por path) vs transcrições (por ui_locale) nos últimos N dias."""
    since = (date.today() - timedelta(days=max(1, min(days, 90)) - 1)).isoformat()
    locale_expr = _locale_from_path_sql("path")
    conn = get_connection()
    try:
        visit_rows = conn.execute(
            f"""
            SELECT {locale_expr} AS locale, COUNT(*) AS visitas
            FROM visitas WHERE day >= ?
            GROUP BY locale
            """,
            (since,),
        ).fetchall()
        visits = {
            (row_to_dict(r).get("locale") or "pt"): int(row_to_dict(r).get("visitas") or 0)
            for r in visit_rows
        }
        try:
            trans_rows = conn.execute(
                """
                SELECT COALESCE(NULLIF(TRIM(ui_locale), ''), 'pt') AS locale, COUNT(*) AS transcricoes
                FROM transcricoes
                WHERE substr(data,1,10) >= ?
                GROUP BY locale
                """,
                (since,),
            ).fetchall()
            trans = {
                (row_to_dict(r).get("locale") or "pt"): int(row_to_dict(r).get("transcricoes") or 0)
                for r in trans_rows
            }
        except Exception:
            # coluna ui_locale pode não existir ainda em DBs antigas a meio da migração
            trans = {}
        locales = sorted(set(visits) | set(trans) | {"pt", "en", "es", "fr", "de"})
        out = []
        for loc in locales:
            v = visits.get(loc, 0)
            t = trans.get(loc, 0)
            if v == 0 and t == 0:
                continue
            out.append(
                {
                    "locale": loc,
                    "visitas": v,
                    "transcricoes": t,
                    "taxa_conversao_pct": round((t / v * 100), 2) if v else 0.0,
                }
            )
        out.sort(key=lambda x: (-x["visitas"], x["locale"]))
        return out
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


def top_utm_campaigns(limit: int = 8) -> list[dict]:
    since = (date.today() - timedelta(days=29)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
              COALESCE(NULLIF(utm_source,''), '(none)') AS source,
              COALESCE(NULLIF(utm_medium,''), '(none)') AS medium,
              COALESCE(NULLIF(utm_campaign,''), '(none)') AS campaign,
              COUNT(*) AS total
            FROM visitas
            WHERE day >= ?
              AND (
                NULLIF(utm_source,'') IS NOT NULL
                OR NULLIF(utm_medium,'') IS NOT NULL
                OR NULLIF(utm_campaign,'') IS NOT NULL
              )
            GROUP BY source, medium, campaign
            ORDER BY total DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()
        return [
            {
                "utm_source": r["source"],
                "utm_medium": r["medium"],
                "utm_campaign": r["campaign"],
                "total": int(r["total"]),
            }
            for r in rows
        ]
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
            GROUP BY device_type ORDER BY total DESC
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
                "SELECT path, day, referrer, device_type, ip_label, visitor_uid, created_at FROM visitas ORDER BY id DESC LIMIT 5000"
            ).fetchall()
            w = csv.writer(buf)
            w.writerow(["path", "day", "referrer", "device", "ip_label", "visitor_uid", "created_at"])
            for r in rows:
                w.writerow([r["path"], r["day"], r["referrer"], r["device_type"], r["ip_label"], r["visitor_uid"], r["created_at"]])
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
                "ai_insights",
                "admin_users",
                "audit_log",
                "site_users",
                "user_transcriptions",
                "daily_usage",
                "user_subscriptions",
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
