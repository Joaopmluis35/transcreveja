"""Testes — Estudo AI no backoffice."""
from __future__ import annotations

from unittest.mock import patch

import admin_store
from ai_estudo import _linear_forecast, _merge_series, build_estudo_snapshot


def test_linear_forecast_basic():
    pts = _linear_forecast([10, 12, 14, 16, 18], horizon=3)
    assert len(pts) == 3
    assert pts[0]["forecast"] >= 18
    assert pts[0]["low"] <= pts[0]["forecast"] <= pts[0]["high"]


def test_merge_series_has_forecast_tail():
    hist = [{"day": "2026-07-01", "total": 5}, {"day": "2026-07-02", "total": 7}]
    merged = _merge_series(hist, "total", 2)
    assert len(merged) == 4
    assert merged[0]["actual"] == 5
    assert merged[0]["forecast"] is None
    assert merged[-1]["actual"] is None
    assert merged[-1]["forecast"] is not None


def test_build_estudo_snapshot_shape():
    snap = build_estudo_snapshot(days=14, horizon=7)
    assert snap["range_days"] == 14
    assert snap["horizon_days"] == 7
    assert "visitas" in snap["series"]
    assert "transcricoes" in snap["series"]
    assert "metrics" in snap


def test_ai_estudo_store_crud():
    saved = admin_store.save_ai_estudo_run(
        run_id="estudo01",
        source_days=30,
        horizon_days=14,
        model="gpt-4o-mini",
        summary="Tendência estável.",
        trend_label="estavel",
        risk_level="baixo",
        metrics={"visitas_humanas_media_7d": 10},
        series={
            "visitas": [{"day": "2026-07-01", "actual": 10, "forecast": None}],
            "transcricoes": [],
        },
        suggestions=[
            {
                "title": "Apostar em SEO",
                "detail": "Criar landing para professores.",
                "priority": "alta",
                "category": "crescimento",
                "evidence": "crescimento flat",
                "cursor_prompt": "Cria landing SEO professores.",
            }
        ],
    )
    assert saved["run_id"] == "estudo01"
    assert len(saved["suggestions"]) == 1
    item_id = saved["suggestions"][0]["id"]

    latest = admin_store.get_latest_ai_estudo_run()
    assert latest and latest["run_id"] == "estudo01"
    assert latest["metrics"]["visitas_humanas_media_7d"] == 10

    listed = admin_store.list_ai_estudo_suggestions(limit=20)
    assert any(int(x["id"]) == int(item_id) for x in listed)

    updated = admin_store.update_ai_estudo_suggestion_status(item_id, "saved")
    assert updated and updated["status"] == "saved"
    assert admin_store.delete_ai_estudo_suggestion(item_id) is True


def test_ai_estudo_generate_mocked(client):
    headers = {"Authorization": "Bearer test-admin-token"}
    fake = {
        "run_id": "mockestudo",
        "model": "gpt-4o-mini",
        "days": 30,
        "horizon": 14,
        "summary": "Previsão de crescimento ligeiro.",
        "trend_label": "crescimento",
        "risk_level": "medio",
        "metrics": {},
        "series": {
            "visitas": [
                {"day": "2026-07-01", "actual": 8, "forecast": None, "low": None, "high": None},
                {"day": "2026-07-02", "actual": None, "forecast": 9, "low": 7, "high": 11},
            ],
            "transcricoes": [],
        },
        "suggestions": [
            {
                "title": "Melhorar conversão",
                "detail": "Testar CTA na home.",
                "priority": "alta",
                "category": "conversao",
                "evidence": "taxa baixa",
                "cursor_prompt": "Melhora CTA da home.",
            }
        ],
    }
    with patch("ai_estudo.generate_ai_estudo", return_value=fake):
        res = client.post(
            "/api/admin/ai-estudo/generate?days=30&horizon=14&save=true",
            headers=headers,
        )
    assert res.status_code == 200
    body = res.json()
    assert body.get("ok") is True
    assert body.get("summary") == "Previsão de crescimento ligeiro."
    assert body.get("series", {}).get("visitas")
    assert body.get("count") == 1

    latest = client.get("/api/admin/ai-estudo/latest", headers=headers)
    assert latest.status_code == 200
    latest_body = latest.json()
    assert latest_body.get("run", {}).get("run_id") == "mockestudo"

    for item in body.get("suggestions") or []:
        if item.get("id"):
            admin_store.delete_ai_estudo_suggestion(int(item["id"]))
