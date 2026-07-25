"""Sugestões AI para o backoffice — analisa visitas/transcrições via OpenAI."""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

INSIGHT_MODEL = os.getenv("INSIGHT_MODEL") or os.getenv("SUM_MODEL", "gpt-4o-mini")


def _safe(label: str, fn, default):
    try:
        return fn()
    except Exception:
        logger.exception("ai_insights snapshot: %s falhou", label)
        return default


def build_insights_snapshot(days: int = 7) -> dict[str, Any]:
    """Pacote compacto de métricas para o modelo (sem PII completa)."""
    import admin_store as store
    from analytics import (
        build_visit_report,
        get_daily_transcription_outcomes,
        get_top_pages,
        parse_owner_visitor_uids,
    )

    n = max(2, min(int(days or 7), 30))
    cfg = _safe("config", store.get_config, {})
    owner_uids = parse_owner_visitor_uids(cfg.get("owner_visitor_uids"))
    report = _safe("visit_report", lambda: build_visit_report(owner_uids, days=n), {})
    outcomes = _safe("outcomes", lambda: get_daily_transcription_outcomes(n), [])
    today = date.today().isoformat()
    since_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat(timespec="seconds") + "Z"

    by_day = report.get("by_day") or {}
    totals = report.get("totals") or {}
    pages = (report.get("pages") or [])[:20]
    refs = (report.get("referrers") or [])[:15]
    devices = (report.get("devices") or [])[:10]
    locales = (report.get("locales") or [])[:10]
    top_pages = _safe("top_pages", lambda: get_top_pages(8), [])

    erros_hoje = _safe(
        "erros_hoje",
        lambda: store.count_transcriptions(day_from=today, day_to=today, status="error"),
        0,
    )
    ok_hoje = _safe(
        "ok_hoje",
        lambda: store.count_transcriptions(day_from=today, day_to=today, status="ok"),
        0,
    )
    api_errors = _safe("api_errors", lambda: store.count_api_errors_since(since_24h), 0)
    user_suggestions = _safe(
        "user_suggestions",
        lambda: store.list_suggestions(unread_only=True, limit=8),
        [],
    )

    return {
        "range_days": n,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "totals": {
            "pageviews": totals.get("pageviews"),
            "unicos": totals.get("unicos"),
            "human_pageviews": totals.get("human_pageviews"),
            "bot_pageviews": totals.get("bot_pageviews"),
            "owner_pageviews": totals.get("owner_pageviews"),
        },
        "by_day": by_day,
        "top_pages_period": pages,
        "top_pages_30d": top_pages,
        "referrers": refs,
        "devices": devices,
        "locales": locales,
        "transcriptions": {
            "ok_hoje": ok_hoje,
            "erros_hoje": erros_hoje,
            "outcomes_by_day": outcomes,
        },
        "api_errors_24h": api_errors,
        "conversion": report.get("conversao_hoje") or {},
        "user_feedback_unread": [
            {
                "nome": (s.get("nome") or "anónimo")[:40],
                "mensagem": (s.get("mensagem") or "")[:240],
                "lang": s.get("lang") or "pt",
            }
            for s in user_suggestions
        ],
        "site": "Ouviescrevi — transcrição de áudio/vídeo com IA (PT, freemium)",
    }


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def generate_ai_insights(days: int = 7, *, max_suggestions: int = 6) -> dict[str, Any]:
    """Chama OpenAI e devolve summary + lista de sugestões (ainda não persistidas)."""
    from openai import OpenAI

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key.startswith("sk-...") or "placeholder" in api_key.lower():
        raise RuntimeError("OPENAI_API_KEY em falta ou inválida no servidor.")

    client = OpenAI(api_key=api_key, timeout=60)
    snapshot = build_insights_snapshot(days)
    max_n = max(3, min(int(max_suggestions or 6), 10))
    system = (
        "És um consultor de produto para o Ouviescrevi (site PT de transcrição com IA). "
        "Analisas métricas reais e propões melhorias práticas, concretas e priorizadas. "
        "Responde APENAS com JSON válido (sem markdown)."
    )
    user = (
        f"Analisa estes dados dos últimos {snapshot.get('range_days')} dias e devolve JSON com:\n"
        '{\n'
        '  "summary": "2-3 frases sobre o estado do site",\n'
        '  "suggestions": [\n'
        "    {\n"
        '      "title": "título curto acionável",\n'
        '      "detail": "o que fazer e porquê (2-4 frases, em português de Portugal)",\n'
        '      "priority": "alta|media|baixa",\n'
        '      "category": "ux|produto|seo|tecnico|conversao|conteudo",\n'
        '      "evidence": "métrica ou padrão que motiva a sugestão",\n'
        '      "cursor_prompt": "frase pronta para colar no chat Cursor e pedir a implementação"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Máximo {max_n} sugestões. Evita genéricos. Foca em visitas, páginas, bots vs humanos, "
        "erros de transcrição, conversão e feedback de utilizadores se existir.\n\n"
        f"DADOS:\n{json.dumps(snapshot, ensure_ascii=False, default=str)[:12000]}"
    )

    resp = client.chat.completions.create(
        model=INSIGHT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.4,
        max_tokens=1800,
        response_format={"type": "json_object"},
    )
    content = ""
    try:
        content = resp.choices[0].message.content or ""
    except Exception:
        content = ""
    parsed = _extract_json(content)
    suggestions_raw = parsed.get("suggestions") or []
    if not isinstance(suggestions_raw, list):
        suggestions_raw = []

    cleaned: list[dict[str, Any]] = []
    for item in suggestions_raw[:max_n]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        detail = str(item.get("detail") or "").strip()
        if not title or not detail:
            continue
        priority = str(item.get("priority") or "media").strip().lower()
        if priority not in ("alta", "media", "baixa"):
            priority = "media"
        category = str(item.get("category") or "produto").strip().lower()[:40]
        cleaned.append(
            {
                "title": title[:160],
                "detail": detail[:2000],
                "priority": priority,
                "category": category,
                "evidence": str(item.get("evidence") or "")[:500],
                "cursor_prompt": str(item.get("cursor_prompt") or title)[:800],
            }
        )

    run_id = uuid.uuid4().hex[:12]
    return {
        "run_id": run_id,
        "model": INSIGHT_MODEL,
        "days": snapshot.get("range_days"),
        "summary": str(parsed.get("summary") or "").strip()[:1200],
        "suggestions": cleaned,
        "snapshot_totals": snapshot.get("totals") or {},
    }
