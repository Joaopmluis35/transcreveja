"""Testes — aparelho da equipa em qualquer rede, percurso (carregou/terminou/voltou) e resumo semanal."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import admin_store as store
from analytics import build_marketing_report, parse_owner_visitor_uids, record_event, record_visit, visitor_uid
from database import get_connection

ADMIN = {"Authorization": "Bearer test-admin-token"}


def test_team_device_token_marks_new_network(client, origin_headers):
    res = client.post("/api/admin/team-device", headers=ADMIN)
    assert res.status_code == 200
    token = res.json()["token"]
    assert store.is_team_device_token(token)
    assert not store.is_team_device_token("inventado")
    assert client.post("/api/admin/team-device").status_code in (401, 403)

    # Simula «rede nova»: tira o IP atual da lista e volta a visitar com o token do aparelho.
    uid = visitor_uid("testclient")
    store.remove_owner_visitor_uid(uid)
    res = client.post("/api/track-visit", json={"path": "/", "team_device": token}, headers=origin_headers)
    if res.status_code == 403:
        pytest.skip("Origem de teste não autorizada neste ambiente.")
    assert uid in parse_owner_visitor_uids(store.get_config().get("owner_visitor_uids"))

    # Um token falso não marca nada.
    store.remove_owner_visitor_uid(uid)
    client.post("/api/track-visit", json={"path": "/", "team_device": "falso"}, headers=origin_headers)
    assert uid not in parse_owner_visitor_uids(store.get_config().get("owner_visitor_uids"))


def test_percurso_counts_steps_and_returning_visitors(client):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM eventos")
        conn.commit()
    finally:
        conn.close()
    a, b, equipa = "203.0.113.81", "203.0.113.82", "198.51.100.83"
    for ip in (a, b, equipa):
        record_visit("/", ip, user_agent="Mozilla/5.0")
    record_event("carregou", a)
    record_event("carregou", a)  # o mesmo visitante conta uma vez
    record_event("terminou", a)
    record_event("carregou", b)
    record_event("carregou", equipa)
    record_event("inventado", a)  # nomes fora da lista são ignorados

    # «Voltou» = visto em 2 dias diferentes.
    ontem = (datetime.now() - timedelta(days=1)).date().isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO visitas (path, day, visitor_hash, visitor_uid, ip_label, created_at, user_agent) "
            "VALUES ('/', ?, 'h', ?, '203.0.113.x', '2026-01-01T00:00:00Z', 'Mozilla/5.0')",
            (ontem, visitor_uid(a)),
        )
        conn.commit()
    finally:
        conn.close()

    p = build_marketing_report({visitor_uid(equipa)}, days=2)["percurso"]
    assert p["carregaram"] == 2
    assert p["terminaram"] == 1
    assert p["voltaram"] >= 1
    assert p["entraram"] >= 2


def test_weekly_report_text_and_once_per_week(client, monkeypatch):
    import email_notify
    import weekly_report

    report = {
        "por_dia": [{"visitantes": 1, "transcricoes": 0, "origens": {}}] * 7
        + [{"visitantes": 12, "transcricoes": 2, "origens": {"Google": 3}}] * 7,
        "percurso": {"entraram": 20, "carregaram": 8, "terminaram": 6, "voltaram": 2},
    }
    assunto, corpo = weekly_report.build_weekly_text(report)
    assert "84 visitas" in assunto and assunto.startswith("🟢")
    assert "semana anterior: 7" in corpo and "Google: 21" in corpo
    assert weekly_report.semaforo(5) == "🟡" and weekly_report.semaforo(1) == "🔴"

    enviados = []
    monkeypatch.setattr(email_notify, "send_notification_email", lambda *a, **k: (enviados.append(a), (True, None))[1])
    store.set_config({"weekly_report_week": ""})
    domingo = datetime(2026, 11, 29, 10, 0, tzinfo=timezone.utc)
    assert weekly_report.maybe_send_weekly(domingo - timedelta(days=1)) is False  # sábado
    assert weekly_report.maybe_send_weekly(domingo.replace(hour=7)) is False  # cedo demais
    assert weekly_report.maybe_send_weekly(domingo) is True
    assert weekly_report.maybe_send_weekly(domingo + timedelta(hours=1)) is False  # já enviado
    assert len(enviados) == 1
