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
