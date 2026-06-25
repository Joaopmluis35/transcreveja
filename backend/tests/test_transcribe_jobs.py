"""Testes — fila assíncrona do endpoint /transcribe."""
from __future__ import annotations

import time


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
        if last.get("status") in ("completed", "failed"):
            break
        time.sleep(0.5)

    assert last.get("status") in ("completed", "failed")
    assert last.get("message")


def test_transcribe_job_not_found(client):
    res = client.get(
        "/transcribe/jobs/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Bearer test-api-token"},
    )
    assert res.status_code == 404
