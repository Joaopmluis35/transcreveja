"""Rotas administrativas do backoffice."""
from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

import admin_store as store
from analytics import (
    get_daily_transcription_outcomes,
    get_daily_transcription_series,
    get_daily_visit_series,
    get_recent_visits,
    get_top_pages,
    get_visit_stats,
)
from cms import get_all_content, get_page_schema, get_seo_overrides, keys_for_page, reset_content, update_content
from database import database_backend, use_turso

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _actor(request: Request) -> str:
    return getattr(request.state, "admin_user", "admin")


@router.get("/dashboard")
def admin_dashboard(request: Request):
    stats = get_visit_stats()
    costs = store.estimate_costs()
    conv = store.conversion_stats()
    maint = store.get_maintenance()
    cfg = store.get_config()
    return {
        "manutencao": maint["manutencao"],
        "maintenance_message": maint["maintenance_message"],
        "block_transcribe_only": maint["block_transcribe_only"],
        "transcricoes_hoje": costs["transcricoes_hoje"],
        "transcricoes_total": costs["transcricoes_total"],
        "visitas": stats,
        "visitas_recentes": get_recent_visits(15),
        "charts": {
            "visitas_diarias": get_daily_visit_series(14),
            "transcricoes_diarias": get_daily_transcription_series(14),
            "transcricoes_resultados": get_daily_transcription_outcomes(14),
            "horas_pico": store.peak_hours(7),
        },
        "top_paginas": get_top_pages(8),
        "top_referrers": store.top_referrers(8),
        "devices": store.device_breakdown(),
        "conversao": conv,
        "custos_openai": costs,
        "cloudflare": store.fetch_cloudflare_analytics(),
        "banner": store.get_active_banner(),
        "sugestoes_nao_lidas": len(store.list_suggestions(unread_only=True, limit=200)),
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
    user = store.create_user(body.username, body.password, body.role)
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
    keys = keys_for_page(page) if page else None
    if page and not keys:
        raise HTTPException(status_code=400, detail="Página desconhecida.")
    content = reset_content(keys)
    store.log_audit(_actor(request), "cms_reset", page or "all")
    return {"ok": True, "content": content}
