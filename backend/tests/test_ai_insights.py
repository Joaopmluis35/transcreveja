"""Testes — sugestões AI no backoffice."""
from __future__ import annotations

from unittest.mock import patch

import admin_store


def test_ai_insights_store_crud():
    saved = admin_store.save_ai_insights(
        [
            {
                "title": "Melhorar CTA",
                "detail": "Testar botão de transcrever na home.",
                "priority": "alta",
                "category": "conversao",
                "evidence": "poucas visitas na /precos",
                "cursor_prompt": "Melhora o CTA da home do Ouviescrevi.",
            }
        ],
        run_id="testrun01",
        source_days=7,
    )
    assert len(saved) == 1
    item_id = saved[0]["id"]
    listed = admin_store.list_ai_insights(limit=20)
    assert any(int(x["id"]) == int(item_id) for x in listed)

    updated = admin_store.update_ai_insight_status(item_id, "saved")
    assert updated and updated["status"] == "saved"

    assert admin_store.delete_ai_insight(item_id) is True
    assert admin_store.delete_ai_insight(item_id) is False


def test_ai_insights_list_endpoint(client):
    headers = {"Authorization": "Bearer test-admin-token"}
    saved = admin_store.save_ai_insights(
        [
            {
                "title": "Endpoint list",
                "detail": "Item de teste API.",
                "priority": "media",
                "category": "produto",
                "evidence": "",
                "cursor_prompt": "Lista OK",
            }
        ],
        run_id="apitest01",
        source_days=7,
    )
    item_id = saved[0]["id"]
    try:
        res = client.get("/api/admin/ai-insights", headers=headers)
        assert res.status_code == 200
        body = res.json()
        assert body.get("ok") is True
        assert any(int(x["id"]) == int(item_id) for x in body.get("items") or [])
    finally:
        admin_store.delete_ai_insight(item_id)


def test_ai_insights_generate_mocked(client):
    headers = {"Authorization": "Bearer test-admin-token"}
    fake = {
        "run_id": "mockrun01",
        "model": "gpt-4o-mini",
        "days": 7,
        "summary": "Resumo de teste.",
        "suggestions": [
            {
                "title": "Reduzir erros",
                "detail": "Investigar falhas Whisper.",
                "priority": "alta",
                "category": "tecnico",
                "evidence": "erros_hoje > 0",
                "cursor_prompt": "Investiga erros de transcrição.",
            }
        ],
        "snapshot_totals": {},
    }
    with patch("ai_insights.generate_ai_insights", return_value=fake):
        res = client.post(
            "/api/admin/ai-insights/generate?days=7&save=true",
            headers=headers,
        )
    assert res.status_code == 200
    body = res.json()
    assert body.get("ok") is True
    assert body.get("summary") == "Resumo de teste."
    assert body.get("count") == 1
    for item in body.get("suggestions") or []:
        if item.get("id"):
            admin_store.delete_ai_insight(int(item["id"]))
