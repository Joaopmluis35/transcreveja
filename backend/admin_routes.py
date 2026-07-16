"""Rotas administrativas do backoffice."""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import admin_store as store
from analytics import (
    build_visit_report,
    get_daily_transcription_outcomes,
    get_daily_transcription_series,
    get_daily_visit_series,
    get_owner_traffic_today,
    get_recent_visits,
    get_top_pages,
    get_visit_stats,
    get_visitor_breakdown,
    mask_ip_label,
    parse_owner_visitor_uids,
    visitor_uid,
)
from cms import get_all_content, get_page_schema, get_seo_overrides, keys_for_page, nav_defaults_for_admin, parse_nav_config, reset_content, update_content
from database import database_backend, use_turso
from security import client_ip

router = APIRouter(prefix="/api/admin", tags=["admin"])
logger = logging.getLogger(__name__)


def _safe(label: str, fn, default):
    try:
        return fn()
    except Exception as exc:
        logger.exception("dashboard: %s falhou", label)
        return default


def _actor(request: Request) -> str:
    return getattr(request.state, "admin_user", "admin")


@router.get("/dashboard")
def admin_dashboard(request: Request):
    today_s = date.today().isoformat()
    stats = _safe("visitas", get_visit_stats, {})
    costs = _safe("custos", store.estimate_costs, {})
    conv = _safe("conversao", store.conversion_stats, {})
    maint = _safe("manutencao", store.get_maintenance, {})
    cfg = _safe("config", store.get_config, {})
    owner_uids = parse_owner_visitor_uids(cfg.get("owner_visitor_uids"))
    trans_total = _safe("trans_total", store.count_transcriptions, 0)
    trans_hoje = _safe(
        "trans_hoje",
        lambda: store.count_transcriptions(day_from=today_s, day_to=today_s),
        0,
    )
    since_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat(timespec="seconds") + "Z"
    return {
        "manutencao": maint.get("manutencao", False),
        "maintenance_message": maint.get("maintenance_message", ""),
        "block_transcribe_only": maint.get("block_transcribe_only", True),
        "transcricoes_hoje": trans_hoje,
        "transcricoes_total": trans_total,
        "utilizadores_total": _safe("utilizadores_total", store.count_site_users, 0),
        "utilizadores_hoje": _safe("utilizadores_hoje", store.count_site_users_today, 0),
        "emails_falhados_24h": _safe(
            "emails_falhados",
            lambda: store.count_email_failures_since(since_24h),
            0,
        ),
        "visitas": stats,
        "visitas_total": stats.get("visitas_total", 0),
        "visitas_trafego": _safe("visitas_trafego", lambda: get_owner_traffic_today(owner_uids), {}),
        "visitas_recentes": _safe("visitas_recentes", lambda: get_recent_visits(10, owner_uids), []),
        "visitantes_distintos": _safe(
            "visitantes_distintos", lambda: get_visitor_breakdown(14, 25, owner_uids), []
        ),
        "owner_visitor_uids": sorted(owner_uids),
        "owner_ip_labels": _safe(
            "owner_ip_labels", lambda: store.get_owner_ip_labels_list(cfg), []
        ),
        "charts": {
            "visitas_diarias": _safe(
                "visitas_diarias", lambda: get_daily_visit_series(14, owner_uids), []
            ),
            "transcricoes_diarias": _safe("transcricoes_diarias", lambda: get_daily_transcription_series(14), []),
            "transcricoes_resultados": _safe(
                "transcricoes_resultados", lambda: get_daily_transcription_outcomes(14), []
            ),
            "horas_pico": _safe("horas_pico", lambda: store.peak_hours(7), []),
        },
        "top_paginas": _safe("top_paginas", lambda: get_top_pages(8), []),
        "top_referrers": _safe("top_referrers", lambda: store.top_referrers(8), []),
        "devices": _safe("devices", store.device_breakdown, []),
        "conversao": conv,
        "conversao_por_idioma": _safe(
            "conversao_por_idioma", lambda: store.conversion_by_locale(14), []
        ),
        "custos_openai": costs,
        "cloudflare": _safe("cloudflare", store.fetch_cloudflare_analytics, None),
        "banner": _safe("banner", store.get_active_banner, None),
        "sugestoes_nao_lidas": _safe(
            "sugestoes",
            lambda: len(store.list_suggestions(unread_only=True, limit=200)),
            0,
        ),
        "alert_transcriptions_daily": int(cfg.get("alert_transcriptions_daily") or 0),
        "alert_visits_daily": int(cfg.get("alert_visits_daily") or 0),
        "database_backend": database_backend(),
        "database_persistent": use_turso(),
    }


@router.get("/me")
def admin_me(request: Request):
    session = getattr(request.state, "admin_session", None) or {}
    return {
        "username": session.get("username") or getattr(request.state, "admin_user", "admin"),
        "role": session.get("role") or "admin",
    }


@router.get("/health")
def admin_health(request: Request):
    import os
    from openai import OpenAI

    oa_client = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            oa_client = OpenAI()
        except Exception:
            oa_client = None
    return store.system_health(oa_client)


@router.get("/transcricoes")
def admin_transcricoes(
    request: Request,
    q: str | None = None,
    status: str | None = None,
    language: str | None = None,
    duplicates_only: bool = False,
    day_from: str | None = None,
    day_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    filters = {
        "q": q,
        "status": status,
        "language": language,
        "duplicates_only": duplicates_only,
        "day_from": day_from,
        "day_to": day_to,
    }
    return {
        "items": store.list_transcriptions(**filters, limit=limit, offset=offset),
        "total": store.count_transcriptions(**filters),
        "stats": store.transcription_stats(**filters),
        "limit": limit,
        "offset": offset,
    }


@router.get("/sugestoes")
def admin_sugestoes(request: Request, unread_only: bool = False, lang: str | None = None):
    return {"items": store.list_suggestions(unread_only=unread_only, lang=lang)}


class SuggestionReadRequest(BaseModel):
    id: int


@router.post("/sugestoes/read")
def admin_sugestao_read(request: Request, body: SuggestionReadRequest):
    store.require_role(getattr(request.state, "admin_session", None), "editor")
    store.mark_suggestion_read(body.id)
    return {"ok": True}


@router.delete("/sugestoes/{suggestion_id}")
def admin_sugestao_delete(request: Request, suggestion_id: int):
    store.require_role(getattr(request.state, "admin_session", None), "editor")
    store.delete_suggestion(suggestion_id)
    store.log_audit(_actor(request), "suggestion_delete", str(suggestion_id))
    return {"ok": True}


@router.post("/test-alert-email")
def admin_test_alert_email(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    result = store.send_test_alert_email(_actor(request))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Falha no envio"))
    return result


@router.post("/test-activity-email")
def admin_test_activity_email(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    result = store.send_test_activity_email(_actor(request))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Falha no envio"))
    return result


@router.get("/email/status")
def admin_email_status(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    from email_notify import get_email_status

    status = get_email_status()
    cfg = store.get_config()
    raw_to = cfg.get("alert_email_to") or status.get("default_to") or ""
    status["alert_email_to"] = re.sub(r"\s+", "", raw_to.strip())
    return status


@router.get("/email/logs")
def admin_email_logs(request: Request, limit: int = 50):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    return {"items": store.list_email_notifications(limit=limit)}


@router.get("/config")
def admin_get_config(request: Request):
    return {"config": store.get_config()}


class ConfigUpdateRequest(BaseModel):
    updates: dict[str, str]


@router.put("/config")
def admin_put_config(request: Request, body: ConfigUpdateRequest):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    cfg = store.set_config(body.updates, _actor(request))
    return {"ok": True, "config": cfg}


@router.post("/visitors/mark-owner")
def admin_mark_owner_visitor(request: Request):
    """Marca o IP atual (backoffice) como visitante da equipa — para separar tráfego teu vs outros."""
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    ip = client_ip(request)
    uid = visitor_uid(ip)
    label = mask_ip_label(ip)
    cfg = store.add_owner_visitor_uid(uid, _actor(request), label)
    return {
        "ok": True,
        "visitor_uid": uid,
        "ip_label": label,
        "owner_visitor_uids": parse_owner_visitor_uids(cfg.get("owner_visitor_uids")),
        "owner_ip_labels": store.get_owner_ip_labels_list(cfg),
    }


@router.post("/visitors/unmark-owner")
def admin_unmark_owner_visitor(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    ip = client_ip(request)
    uid = visitor_uid(ip)
    cfg = store.remove_owner_visitor_uid(uid, _actor(request))
    return {
        "ok": True,
        "visitor_uid": uid,
        "owner_visitor_uids": parse_owner_visitor_uids(cfg.get("owner_visitor_uids")),
        "owner_ip_labels": store.get_owner_ip_labels_list(cfg),
    }


class MaintenanceRequest(BaseModel):
    manutencao: bool
    maintenance_message: str | None = None
    block_transcribe_only: bool | None = None


@router.put("/maintenance")
def admin_maintenance(request: Request, body: MaintenanceRequest):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    return store.set_maintenance(
        body.manutencao,
        body.maintenance_message,
        body.block_transcribe_only,
        _actor(request),
    )


@router.get("/banners")
def admin_banners(request: Request):
    return {"items": store.list_banners()}


@router.put("/banners")
def admin_save_banner(request: Request, body: dict):
    store.require_role(getattr(request.state, "admin_session", None), "editor")
    return {"ok": True, "banner": store.save_banner(body, _actor(request))}


@router.get("/audit")
def admin_audit(request: Request, limit: int = 50):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    return {"items": store.get_audit_log(limit)}


@router.get("/errors")
def admin_errors(request: Request, limit: int = 50):
    return {"items": store.get_api_errors(limit)}


@router.get("/server-logs")
def admin_server_logs(
    request: Request,
    limit: int = 300,
    q: str = "",
    level: str = "",
    fmt: str = "json",
):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    from log_buffer import format_logs_text, get_memory_logs, tail_log_file

    import main as app_main

    items = get_memory_logs(limit=limit, q=q, level=level)
    file_tail = tail_log_file(app_main.LOG_FILE, limit=min(limit, 400))
    text = format_logs_text(items)
    if file_tail:
        text = (text + "\n\n--- últimas linhas do ficheiro ---\n" + "\n".join(file_tail)).strip()
    if fmt == "text":
        return PlainTextResponse(text or "(sem registos)")
    return {
        "items": items,
        "file_tail": file_tail,
        "text": text,
        "log_file": app_main.LOG_FILE,
    }


@router.get("/processing-jobs")
def admin_processing_jobs(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    import main as app_main

    return {"items": app_main.export_video_sub_jobs()}


@router.get("/users")
def admin_users(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    return {"items": store.list_users()}


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str = "editor"


@router.post("/users")
def admin_create_user(request: Request, body: UserCreateRequest):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    try:
        user = store.create_user(body.username, body.password, body.role)
    except ValueError as exc:
        code = str(exc)
        if code == "username_exists":
            raise HTTPException(
                status_code=409,
                detail="Este nome de utilizador já existe.",
            ) from exc
        if code == "username_required":
            raise HTTPException(
                status_code=400,
                detail="Indica um nome de utilizador.",
            ) from exc
        raise HTTPException(status_code=400, detail="Dados inválidos.") from exc
    store.log_audit(_actor(request), "user_create", body.username)
    return {"ok": True, "user": user}


class UserDeleteRequest(BaseModel):
    id: int


@router.delete("/users/{user_id}")
def admin_delete_user(request: Request, user_id: int):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    store.delete_user(user_id)
    store.log_audit(_actor(request), "user_delete", str(user_id))
    return {"ok": True}


class UserUpdateRequest(BaseModel):
    role: str


@router.patch("/users/{user_id}")
def admin_update_user(request: Request, user_id: int, body: UserUpdateRequest):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    try:
        user = store.update_user_role(user_id, body.role)
    except ValueError as exc:
        code = str(exc)
        if code == "not_found":
            raise HTTPException(status_code=404, detail="Utilizador não encontrado.") from exc
        if code == "last_admin":
            raise HTTPException(
                status_code=400,
                detail="Não é possível alterar o papel do único administrador.",
            ) from exc
        raise HTTPException(status_code=400, detail="Papel inválido.") from exc
    store.log_audit(_actor(request), "user_role_update", f"{user.get('username')}:{body.role}")
    return {"ok": True, "user": user}


@router.get("/export/visit-report")
def admin_export_visit_report(request: Request):
    """JSON compacto ontem+hoje para análise (partilhar no chat)."""
    store.require_role(getattr(request.state, "admin_session", None), "viewer")
    try:
        cfg = store.get_config()
        owner_uids = parse_owner_visitor_uids(cfg.get("owner_visitor_uids"))
        report = build_visit_report(owner_uids)
    except Exception as exc:
        logger.exception("export visit-report falhou")
        raise HTTPException(status_code=500, detail=f"Falha ao gerar relatório: {exc}") from exc
    day = report.get("range", {}).get("hoje") or date.today().isoformat()
    return JSONResponse(
        report,
        headers={
            "Content-Disposition": f'attachment; filename="ouviescrevi-visitas-{day}.json"',
        },
    )


@router.get("/export/{table}")
def admin_export(request: Request, table: str):
    store.require_role(getattr(request.state, "admin_session", None), "viewer")
    if table not in ("visitas", "transcricoes"):
        raise HTTPException(status_code=400, detail="Tabela inválida.")
    csv_data = store.export_csv(table)
    return PlainTextResponse(csv_data, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{table}.csv"',
    })


@router.get("/backup")
def admin_backup(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    return JSONResponse(store.backup_json())


@router.get("/site-content")
def admin_get_site_content(request: Request):
    return {
        "content": get_all_content(),
        "keys": sorted(get_all_content().keys()),
        "pages": get_page_schema(),
        "seo": get_seo_overrides(),
        "nav_defaults": nav_defaults_for_admin(),
    }


@router.put("/site-content")
def admin_put_site_content(request: Request, body: dict):
    store.require_role(getattr(request.state, "admin_session", None), "editor")
    updates = body.get("updates") or {}
    if not updates:
        raise HTTPException(status_code=400, detail="Nenhuma alteração enviada.")
    content = update_content(updates)
    store.log_audit(_actor(request), "cms_update", json.dumps(list(updates.keys())[:20]))
    return {"ok": True, "content": content}


@router.post("/site-content/reset")
def admin_reset_site_content(request: Request, body: dict | None = None):
    store.require_role(getattr(request.state, "admin_session", None), "editor")
    page = (body or {}).get("page")
    keys = (body or {}).get("keys")
    if not keys and page:
        keys = keys_for_page(page)
    if not keys:
        raise HTTPException(status_code=400, detail="Nada a repor (indica página ou chaves).")
    if page and not keys_for_page(page) and not (body or {}).get("keys"):
        raise HTTPException(status_code=400, detail="Página desconhecida.")
    content = reset_content(keys)
    store.log_audit(_actor(request), "cms_reset", page or "all")
    return {"ok": True, "content": content}


@router.get("/billing")
def admin_billing_status(request: Request):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    from billing import billing_config, list_subscriptions

    cfg = store.get_config()
    return {
        "config": {
            "billing_enabled": cfg.get("billing_enabled", "0"),
            "pricing_hidden": cfg.get("pricing_hidden", "1"),
            "stripe_public_key": cfg.get("stripe_public_key", ""),
            "stripe_price_id_pro": cfg.get("stripe_price_id_pro", ""),
            "pro_quota_daily": cfg.get("pro_quota_daily", "200"),
            "pro_price_label": cfg.get("pro_price_label", "9,99 €/mês"),
            "quota_anonymous_daily": cfg.get("quota_anonymous_daily", "3"),
            "quota_registered_daily": cfg.get("quota_registered_daily", "20"),
            "stripe_secret_set": bool(cfg.get("stripe_secret_key") or ""),
            "stripe_webhook_set": bool(cfg.get("stripe_webhook_secret") or ""),
        },
        "status": billing_config(),
        "subscriptions": list_subscriptions(30),
    }


class BillingConfigUpdate(BaseModel):
    updates: dict[str, str]


@router.put("/billing")
def admin_billing_config(request: Request, body: BillingConfigUpdate):
    store.require_role(getattr(request.state, "admin_session", None), "admin")
    allowed = {
        "billing_enabled",
        "pricing_hidden",
        "stripe_public_key",
        "stripe_secret_key",
        "stripe_webhook_secret",
        "stripe_price_id_pro",
        "pro_quota_daily",
        "pro_price_label",
        "quota_anonymous_daily",
        "quota_registered_daily",
    }
    updates = {k: v for k, v in body.updates.items() if k in allowed}
    if not updates:
        raise HTTPException(status_code=400, detail="Nada para atualizar.")
    cfg = store.set_config(updates, actor=_actor(request))
    store.log_audit(_actor(request), "billing_config", json.dumps(list(updates.keys())))
    from billing import billing_config

    return {
        "ok": True,
        "status": billing_config(),
        "config": {
            "billing_enabled": cfg.get("billing_enabled", "0"),
            "pricing_hidden": cfg.get("pricing_hidden", "1"),
            "stripe_public_key": cfg.get("stripe_public_key", ""),
            "stripe_price_id_pro": cfg.get("stripe_price_id_pro", ""),
            "pro_quota_daily": cfg.get("pro_quota_daily", "200"),
            "pro_price_label": cfg.get("pro_price_label", "9,99 €/mês"),
            "quota_anonymous_daily": cfg.get("quota_anonymous_daily", "3"),
            "quota_registered_daily": cfg.get("quota_registered_daily", "20"),
            "stripe_secret_set": bool(cfg.get("stripe_secret_key") or ""),
            "stripe_webhook_set": bool(cfg.get("stripe_webhook_secret") or ""),
        },
    }
