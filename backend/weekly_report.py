"""Resumo semanal por email (domingo de manhã): visitas reais, Google, percurso e semáforo da meta."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

import admin_store
from analytics import build_marketing_report, parse_owner_visitor_uids

logger = logging.getLogger(__name__)

# Meta até ao fim de fevereiro de 2027 (média de visitantes reais por dia).
META_VERDE = 10
META_AMARELO = 3


def semaforo(media_dia: float) -> str:
    if media_dia >= META_VERDE:
        return "🟢"
    if media_dia >= META_AMARELO:
        return "🟡"
    return "🔴"


def build_weekly_text(report: dict) -> tuple[str, str]:
    dias = report.get("por_dia") or []
    ult, ant = dias[-7:], dias[-14:-7]
    v7 = sum(d["visitantes"] for d in ult)
    v_ant = sum(d["visitantes"] for d in ant)
    t7 = sum(d["transcricoes"] for d in ult)
    google = sum((d.get("origens") or {}).get("Google", 0) for d in ult)
    media = v7 / 7
    p = report.get("percurso") or {}
    sinal = semaforo(media)
    assunto = f"{sinal} Ouviescrevi: {v7} visitas reais esta semana"
    corpo = "\n".join([
        "Resumo da semana (só pessoas: sem bots nem a equipa).",
        "",
        f"Visitas reais (soma por dia): {v7}  (semana anterior: {v_ant})",
        f"Média por dia: {media:.1f}  {sinal}  meta: {META_VERDE} por dia até ao fim de fevereiro",
        f"Entradas vindas do Google: {google}",
        f"Transcrições: {t7}",
        "",
        "Percurso nos últimos 14 dias:",
        f"  entraram {p.get('entraram', 0)} → carregaram um ficheiro {p.get('carregaram', 0)}"
        f" → terminaram {p.get('terminaram', 0)} → voltaram noutro dia {p.get('voltaram', 0)}",
        "",
        "Mais detalhe: https://www.ouviescrevi.pt/backoffice.html",
    ])
    return assunto, corpo


def maybe_send_weekly(now: datetime | None = None) -> bool:
    """Envia uma vez por semana, ao domingo a partir das 9h UTC (9h ou 10h em Lisboa)."""
    now = now or datetime.now(timezone.utc)
    if now.weekday() != 6 or now.hour < 9:
        return False
    cfg = admin_store.get_config()
    if cfg.get("weekly_report_enabled", "1") == "0":
        return False
    semana = now.strftime("%G-W%V")
    if cfg.get("weekly_report_week") == semana:
        return False
    from email_notify import send_notification_email

    owners = parse_owner_visitor_uids(cfg.get("owner_visitor_uids"))
    assunto, corpo = build_weekly_text(build_marketing_report(owners, days=14))
    # Marca antes de enviar: se o email falhar, não insiste de 30 em 30 min (fica no registo de emails).
    admin_store.set_config({"weekly_report_week": semana}, "resumo-semanal")
    ok, _ = send_notification_email(corpo, assunto, kind="weekly_report", actor="resumo-semanal")
    return ok


def start_weekly_loop() -> None:
    def _loop() -> None:
        while True:
            try:
                maybe_send_weekly()
            except Exception:
                logger.exception("Resumo semanal falhou")
            time.sleep(1800)

    threading.Thread(target=_loop, daemon=True, name="resumo-semanal").start()
