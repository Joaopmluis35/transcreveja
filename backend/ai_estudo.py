"""Estudo AI — tendências, previsão e sugestões forward-looking no backoffice."""
from __future__ import annotations

import json
import logging
import math
import os
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

ESTUDO_MODEL = os.getenv("ESTUDO_MODEL") or os.getenv("INSIGHT_MODEL") or os.getenv(
    "SUM_MODEL", "gpt-4o-mini"
)


def _safe(label: str, fn, default):
    try:
        return fn()
    except Exception:
        logger.exception("ai_estudo snapshot: %s falhou", label)
        return default


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


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _linear_forecast(values: list[float], horizon: int) -> list[dict[str, float]]:
    """Previsão linear simples + banda ±1 desvio (mín. 15% do valor)."""
    horizon = max(1, min(int(horizon or 7), 30))
    ys = [max(0.0, float(v or 0)) for v in values]
    n = len(ys)
    if n == 0:
        return [{"forecast": 0.0, "low": 0.0, "high": 0.0} for _ in range(horizon)]
    if n == 1:
        base = ys[0]
        band = max(1.0, base * 0.2)
        return [
            {"forecast": round(base, 1), "low": round(max(0, base - band), 1), "high": round(base + band, 1)}
            for _ in range(horizon)
        ]

    xs = list(range(n))
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs) or 1.0
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    variance = sum(r * r for r in residuals) / max(1, n - 2)
    stdev = math.sqrt(max(0.0, variance))

    out: list[dict[str, float]] = []
    for i in range(1, horizon + 1):
        x = n - 1 + i
        pred = max(0.0, intercept + slope * x)
        band = max(stdev, pred * 0.15, 1.0)
        out.append(
            {
                "forecast": round(pred, 1),
                "low": round(max(0.0, pred - band), 1),
                "high": round(pred + band, 1),
            }
        )
    return out


def _merge_series(
    historical: list[dict[str, Any]],
    value_key: str,
    horizon: int,
) -> list[dict[str, Any]]:
    """Junta histórico (actual) com dias futuros (forecast)."""
    points: list[dict[str, Any]] = []
    values: list[float] = []
    last_day: date | None = None
    for row in historical:
        day = str(row.get("day") or "")
        try:
            last_day = date.fromisoformat(day)
        except ValueError:
            continue
        val = float(row.get(value_key) or 0)
        values.append(val)
        points.append(
            {
                "day": day,
                "actual": round(val, 1),
                "forecast": None,
                "low": None,
                "high": None,
            }
        )

    if not last_day:
        last_day = date.today()
        values = []

    forecasts = _linear_forecast(values, horizon)
    for i, f in enumerate(forecasts, start=1):
        d = (last_day + timedelta(days=i)).isoformat()
        points.append(
            {
                "day": d,
                "actual": None,
                "forecast": f["forecast"],
                "low": f["low"],
                "high": f["high"],
            }
        )
    return points


def build_estudo_snapshot(days: int = 30, horizon: int = 14) -> dict[str, Any]:
    """Séries históricas + previsão baseline + contexto para o modelo."""
    import admin_store as store
    from analytics import (
        get_daily_transcription_outcomes,
        get_daily_transcription_series,
        get_daily_visit_series,
        get_top_pages,
        parse_owner_visitor_uids,
    )

    n = max(7, min(int(days or 30), 90))
    h = max(3, min(int(horizon or 14), 30))
    cfg = _safe("config", store.get_config, {})
    owner_uids = parse_owner_visitor_uids(cfg.get("owner_visitor_uids"))

    visits = _safe(
        "visits",
        lambda: get_daily_visit_series(n, owner_uids=owner_uids),
        [],
    )
    trans = _safe("trans", lambda: get_daily_transcription_series(n), [])
    outcomes = _safe("outcomes", lambda: get_daily_transcription_outcomes(n), [])
    top_pages = _safe("top_pages", lambda: get_top_pages(8), [])

    visit_human = [
        {"day": v.get("day"), "total": v.get("outros") if v.get("outros") is not None else v.get("total")}
        for v in visits
    ]
    series_visitas = _merge_series(visit_human, "total", h)
    series_trans = _merge_series(trans, "total", h)

    human_vals = [float(v.get("outros") or 0) for v in visits]
    bot_vals = [float(v.get("bots") or 0) for v in visits]
    trans_vals = [float(t.get("total") or 0) for t in trans]
    ok_vals = [float(o.get("ok") or 0) for o in outcomes]
    err_vals = [float(o.get("erros") or 0) for o in outcomes]

    last7 = human_vals[-7:] if human_vals else []
    prev7 = human_vals[-14:-7] if len(human_vals) >= 14 else human_vals[: max(0, len(human_vals) - 7)]
    last7_avg = _mean(last7)
    prev7_avg = _mean(prev7) if prev7 else last7_avg
    growth_pct = (
        round(((last7_avg - prev7_avg) / prev7_avg) * 100, 1) if prev7_avg > 0 else None
    )

    return {
        "range_days": n,
        "horizon_days": h,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {
            "visitas_humanas_media_7d": round(last7_avg, 1),
            "visitas_humanas_media_7d_anterior": round(prev7_avg, 1),
            "crescimento_visitas_pct": growth_pct,
            "bots_media_7d": round(_mean(bot_vals[-7:]), 1) if bot_vals else 0,
            "transcricoes_media_7d": round(_mean(trans_vals[-7:]), 1) if trans_vals else 0,
            "ok_media_7d": round(_mean(ok_vals[-7:]), 1) if ok_vals else 0,
            "erros_media_7d": round(_mean(err_vals[-7:]), 1) if err_vals else 0,
        },
        "series": {
            "visitas": series_visitas,
            "transcricoes": series_trans,
        },
        "top_pages_30d": top_pages,
        "outcomes_by_day": outcomes[-14:],
        "site": "Ouviescrevi — transcrição de áudio/vídeo com IA (PT, freemium)",
    }


def generate_ai_estudo(
    days: int = 30,
    horizon: int = 14,
    *,
    max_suggestions: int = 6,
) -> dict[str, Any]:
    """Calcula previsão + pede à OpenAI resumo e sugestões acionáveis."""
    from openai import OpenAI

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key or api_key.startswith("sk-...") or "placeholder" in api_key.lower():
        raise RuntimeError("OPENAI_API_KEY em falta ou inválida no servidor.")

    client = OpenAI(api_key=api_key, timeout=60)
    snapshot = build_estudo_snapshot(days, horizon)
    max_n = max(3, min(int(max_suggestions or 6), 10))

    system = (
        "És um analista de crescimento para o Ouviescrevi (site PT de transcrição com IA). "
        "Recebes séries históricas e uma previsão linear baseline (já calculada). "
        "Não inventes números novos para gráficos — usa a previsão fornecida. "
        "Propõe ações concretas para o futuro (crescimento, risco, conversão, produto). "
        "Responde APENAS com JSON válido (sem markdown), em português de Portugal."
    )
    user = (
        f"Analisa tendência e previsão ({snapshot.get('range_days')} dias de histórico, "
        f"{snapshot.get('horizon_days')} dias de horizonte) e devolve JSON:\n"
        "{\n"
        '  "summary": "2-4 frases: tendência atual + o que a previsão sugere",\n'
        '  "trend_label": "crescimento|estavel|queda|volatil",\n'
        '  "risk_level": "baixo|medio|alto",\n'
        '  "suggestions": [\n'
        "    {\n"
        '      "title": "título curto acionável (focado no futuro)",\n'
        '      "detail": "o que fazer nas próximas 1-2 semanas e porquê (2-4 frases)",\n'
        '      "priority": "alta|media|baixa",\n'
        '      "category": "crescimento|risco|produto|tecnico|conversao|seo|conteudo",\n'
        '      "evidence": "métrica ou padrão da série/previsão",\n'
        '      "cursor_prompt": "frase pronta para colar no chat Cursor"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Máximo {max_n} sugestões. Evita genéricos. Foca em visitas humanas vs bots, "
        "transcrições, erros e páginas top.\n\n"
        f"DADOS:\n{json.dumps(snapshot, ensure_ascii=False, default=str)[:14000]}"
    )

    resp = client.chat.completions.create(
        model=ESTUDO_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.35,
        max_tokens=2000,
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
        category = str(item.get("category") or "crescimento").strip().lower()[:40]
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

    trend = str(parsed.get("trend_label") or "estavel").strip().lower()[:20]
    if trend not in ("crescimento", "estavel", "queda", "volatil"):
        trend = "estavel"
    risk = str(parsed.get("risk_level") or "medio").strip().lower()[:12]
    if risk not in ("baixo", "medio", "alto"):
        risk = "medio"

    run_id = uuid.uuid4().hex[:12]
    return {
        "run_id": run_id,
        "model": ESTUDO_MODEL,
        "days": snapshot.get("range_days"),
        "horizon": snapshot.get("horizon_days"),
        "summary": str(parsed.get("summary") or "").strip()[:1500],
        "trend_label": trend,
        "risk_level": risk,
        "metrics": snapshot.get("metrics") or {},
        "series": snapshot.get("series") or {},
        "suggestions": cleaned,
    }
