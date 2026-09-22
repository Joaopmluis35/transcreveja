"""Testes — fila assíncrona do endpoint /transcribe."""
from __future__ import annotations

import time

import main as app_main


def test_transcribe_returns_job_id(client):
    res = client.post(
        "/transcribe",
        data={"token": "test-api-token"},
        files={"file": ("tiny.wav", b"RIFFxxxx", "audio/wav")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("job_id")
    assert body.get("status") == "processing"
    assert "estimate_transcribe_sec" in body

    job_id = body["job_id"]
    deadline = time.time() + 30
    last = {}
    while time.time() < deadline:
        st = client.get(
            f"/transcribe/jobs/{job_id}",
            headers={"Authorization": "Bearer test-api-token"},
        )
        assert st.status_code == 200
        last = st.json()
        if last.get("status") in ("completed", "failed", "cancelled"):
            break
        time.sleep(0.5)

    assert last.get("status") in ("completed", "failed", "cancelled")
    assert last.get("message")


def test_transcribe_job_not_found(client):
    res = client.get(
        "/transcribe/jobs/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Bearer test-api-token"},
    )
    assert res.status_code == 404


def test_find_active_transcribe_duplicate_matches_same_file():
    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()
    job_id = "dup-job-1"
    app_main._transcribe_job_set(
        job_id,
        status="processing",
        usage_key="ip:1.2.3.4",
        filename="long.mp4",
        size_bytes=63741938,
        created_at=time.monotonic(),
        message="A processar",
        content_fingerprint="abc123",
        client_fingerprint="long.mp4|63741938|1",
    )
    found = app_main._find_active_transcribe_duplicate(
        "ip:1.2.3.4", "long.mp4", 63741938
    )
    assert found is not None
    assert found[0] == job_id

    found_fp = app_main._find_active_transcribe_duplicate(
        "ip:1.2.3.4",
        "other-name.mp4",
        999,
        content_fp="abc123",
    )
    assert found_fp is not None
    assert found_fp[0] == job_id

    # ficheiro diferente → sem dedupe
    assert (
        app_main._find_active_transcribe_duplicate(
            "ip:1.2.3.4", "other.mp4", 63741938
        )
        is None
    )
    # utilizador diferente → sem dedupe
    assert (
        app_main._find_active_transcribe_duplicate(
            "ip:9.9.9.9", "long.mp4", 63741938
        )
        is None
    )

    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()


def test_cancel_transcribe_job(client):
    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()
    job_id = "cancel-job-1"
    app_main._transcribe_job_set(
        job_id,
        status="processing",
        message="A processar",
        progress=20,
        cancel_requested=False,
    )
    res = client.post(
        f"/transcribe/jobs/{job_id}/cancel",
        headers={"Authorization": "Bearer test-api-token"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body.get("cancel_requested") is True
    assert app_main._transcribe_job_cancel_requested(job_id) is True

    app_main._finish_transcribe_cancelled(job_id, "rid-test")
    job = app_main._transcribe_job_get(job_id)
    assert job is not None
    assert job.get("status") == "cancelled"

    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()


def test_recent_completed_duplicate_reuse():
    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()
    job_id = "done-job-1"
    app_main._transcribe_job_set(
        job_id,
        status="completed",
        usage_key="ip:1.2.3.4",
        filename="aula.mp4",
        size_bytes=1000,
        updated_at=time.monotonic(),
        transcription="olá mundo",
        formatted="00:00 olá mundo",
        content_fingerprint="fp-done",
    )
    found = app_main._find_recent_completed_transcribe_duplicate(
        "ip:1.2.3.4",
        "aula.mp4",
        1000,
        content_fp="fp-done",
    )
    assert found is not None
    assert found[0] == job_id
    assert found[1].get("transcription") == "olá mundo"

    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs.clear()
