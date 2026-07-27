"""Persistência de jobs assíncronos (transcribe / video-subs) para sobreviver a restarts."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from database import get_connection

logger = logging.getLogger(__name__)

_RESTART_ERROR = (
    "O servidor reiniciou durante o processamento. "
    "O ficheiro deve continuar no browser — volta a clicar em Transcrever (ou Legendar)."
)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Remove campos só-em-memória (monotonic clocks, logs longos)."""
    skip = {"updated_at", "created_at", "stage_started_at", "job_log"}
    out: dict[str, Any] = {}
    for k, v in job.items():
        if k in skip:
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def upsert_async_job(job_id: str, kind: str, job: dict[str, Any]) -> None:
    status = str(job.get("status") or "processing")
    payload = json.dumps(_safe_payload(job), ensure_ascii=False)
    now = _now()
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT created_at FROM async_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        created = existing["created_at"] if existing else now
        conn.execute(
            """
            INSERT INTO async_jobs (job_id, kind, status, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                kind = excluded.kind,
                status = excluded.status,
                payload_json = excluded.payload_json,
                updated_at = excluded.updated_at
            """,
            (job_id, kind, status, payload, created, now),
        )
        conn.commit()
    except Exception:
        logger.exception("async_jobs upsert falhou job=%s", job_id)
    finally:
        conn.close()


def load_async_job(job_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT job_id, kind, status, payload_json, created_at, updated_at "
            "FROM async_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload["status"] = row["status"] or payload.get("status") or "processing"
        payload["job_id"] = row["job_id"]
        payload["kind"] = row["kind"]
        payload["persisted_at"] = row["updated_at"]
        return payload
    except Exception:
        logger.exception("async_jobs load falhou job=%s", job_id)
        return None
    finally:
        conn.close()


def fail_orphaned_processing_jobs() -> int:
    """Após restart: jobs 'processing' já não têm worker — marcar como failed."""
    now = _now()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT job_id, payload_json FROM async_jobs WHERE status = 'processing'"
        ).fetchall()
        n = 0
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload["status"] = "failed"
            payload["error"] = _RESTART_ERROR
            payload["message"] = "Servidor reiniciou — tenta novamente."
            payload["progress"] = 100
            conn.execute(
                """
                UPDATE async_jobs
                SET status = 'failed', payload_json = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(payload, ensure_ascii=False), now, row["job_id"]),
            )
            n += 1
        if n:
            conn.commit()
            logger.warning("Marcou %d job(s) órfão(s) como failed após restart", n)
        return n
    except Exception:
        logger.exception("fail_orphaned_processing_jobs falhou")
        return 0
    finally:
        conn.close()


def prune_async_jobs(max_age_hours: int = 24) -> int:
    cutoff = (datetime.utcnow() - timedelta(hours=max(1, max_age_hours))).isoformat(
        timespec="seconds"
    ) + "Z"
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM async_jobs WHERE updated_at < ?",
            (cutoff,),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    except Exception:
        logger.exception("prune_async_jobs falhou")
        return 0
    finally:
        conn.close()
