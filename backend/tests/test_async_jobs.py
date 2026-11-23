"""Testes — persistência de async jobs."""
from __future__ import annotations

import async_jobs


def test_async_job_upsert_and_load():
    job_id = "test-job-persist-01"
    async_jobs.upsert_async_job(
        job_id,
        "transcribe",
        {
            "status": "processing",
            "message": "A converter…",
            "progress": 15,
            "transcription": None,
        },
    )
    loaded = async_jobs.load_async_job(job_id)
    assert loaded is not None
    assert loaded["status"] == "processing"
    assert loaded["progress"] == 15
    assert loaded["kind"] == "transcribe"

    async_jobs.upsert_async_job(
        job_id,
        "transcribe",
        {
            "status": "completed",
            "message": "OK",
            "progress": 100,
            "transcription": "olá mundo",
        },
    )
    loaded2 = async_jobs.load_async_job(job_id)
    assert loaded2["status"] == "completed"
    assert loaded2["transcription"] == "olá mundo"


def test_fail_orphaned_processing_jobs():
    job_id = "test-job-orphan-01"
    async_jobs.upsert_async_job(
        job_id,
        "transcribe",
        {"status": "processing", "message": "mid-run", "progress": 40},
    )
    n = async_jobs.fail_orphaned_processing_jobs()
    assert n >= 1
    loaded = async_jobs.load_async_job(job_id)
    assert loaded["status"] == "failed"
    assert "reiniciou" in (loaded.get("error") or "").lower()


def test_prune_only_old_transcribe_jobs():
    """Privacidade: os textos das transcrições saem da base de dados; os trabalhos de vídeo ficam até ao prazo geral."""
    from database import get_connection

    async_jobs.upsert_async_job("prune-tr-old", "transcribe", {"status": "completed", "transcription": "segredo"})
    async_jobs.upsert_async_job("prune-vid-old", "video-subs", {"status": "completed"})
    conn = get_connection()
    try:
        # 5 h atrás: mais velho que o prazo das transcrições (3 h), mais novo que o geral (24 h).
        from datetime import datetime, timedelta

        five_h = (datetime.utcnow() - timedelta(hours=5)).isoformat(timespec="seconds") + "Z"
        conn.execute(
            "UPDATE async_jobs SET updated_at = ? WHERE job_id IN (?, ?)",
            (five_h, "prune-tr-old", "prune-vid-old"),
        )
        conn.commit()
    finally:
        conn.close()
    async_jobs.prune_async_jobs(3, kind="transcribe")
    assert async_jobs.load_async_job("prune-tr-old") is None
    assert async_jobs.load_async_job("prune-vid-old") is not None
