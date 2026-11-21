"""Rotas de partilha: transcrições (/s/{id}) e jogos de quiz da Aula Pronta (link, ranking, «os meus jogos»).

Separado do main.py em 21/11/2026. As dependências do main (origens, limites, sessão) chegam por configure().
"""
from __future__ import annotations

import re
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import admin_store
from security import origin_is_allowed

router = APIRouter()

ALLOWED_ORIGINS: list[str] = []
enforce_rate_limit: Callable[[Request, str, int, int], None]
resolve_site_actor: Callable[[Request], dict]


def configure(*, allowed_origins: list[str], rate_limit: Callable, site_actor: Callable) -> None:
    global ALLOWED_ORIGINS, enforce_rate_limit, resolve_site_actor
    ALLOWED_ORIGINS = allowed_origins
    enforce_rate_limit = rate_limit
    resolve_site_actor = site_actor


class ShareTranscriptRequest(BaseModel):
    text: str
    title: str | None = None
    locale: str = "pt"


class ShareQuizRequest(BaseModel):
    questions: list[dict]
    title: str | None = None
    locale: str = "pt"


class QuizScoreRequest(BaseModel):
    name: str
    score: int
    correct: int
    total: int


@router.post("/api/share/transcript")
def share_transcript(req: ShareTranscriptRequest, request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    enforce_rate_limit(request, "share", 30, 3600)
    try:
        created = admin_store.create_shared_transcript(
            req.text, title=req.title, locale=req.locale or "pt"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    origin = (request.headers.get("origin") or "https://www.ouviescrevi.pt").rstrip("/")
    # Prefer pretty /s/{id} (proxied to API HTML with OG tags via vercel.json)
    url = f"{origin}/s/{created['id']}"
    return {"ok": True, "id": created["id"], "url": url, "expires_at": created["expires_at"]}


@router.post("/api/share/quiz")
def share_quiz(req: ShareQuizRequest, request: Request):
    """Guarda as perguntas de um jogo (Aula Pronta) e devolve um link para outra pessoa jogar."""
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    enforce_rate_limit(request, "share", 30, 3600)
    actor = resolve_site_actor(request)
    try:
        created = admin_store.create_shared_quiz(
            req.questions,
            title=req.title,
            locale=req.locale or "pt",
            owner=actor.get("email") if actor["type"] == "user" else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    origin = (request.headers.get("origin") or "https://www.ouviescrevi.pt").rstrip("/")
    url = f"{origin}/jogo.html?id={created['id']}"
    return {"ok": True, "id": created["id"], "url": url, "expires_at": created["expires_at"]}


def _quiz_owner_or_401(request: Request) -> str:
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=401, detail="Inicia sessão para veres os teus jogos.")
    return actor["email"]


# Tem de vir antes de /api/share/quiz/{share_id}, senão "mine" seria lido como um id.
@router.get("/api/share/quiz/mine")
def my_shared_quizzes(request: Request):
    """Jogos partilhados pela conta com sessão iniciada (para os ver em qualquer dispositivo)."""
    owner = _quiz_owner_or_401(request)
    origin = (request.headers.get("origin") or "https://www.ouviescrevi.pt").rstrip("/")
    games = admin_store.list_owner_quizzes(owner)
    for g in games:
        g["url"] = f"{origin}/jogo.html?id={g['id']}"
    return {"games": games}


@router.delete("/api/share/quiz/{share_id}/mine")
def unlink_my_quiz(share_id: str, request: Request):
    """Tira o jogo da lista da conta; o link continua a funcionar até expirar."""
    owner = _quiz_owner_or_401(request)
    admin_store.unlink_owner_quiz(owner, share_id)
    return {"ok": True}


@router.get("/api/share/quiz/{share_id}")
def get_share_quiz(share_id: str, request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    item = admin_store.get_shared_quiz(share_id)
    if not item:
        raise HTTPException(status_code=404, detail="Jogo não encontrado ou expirado.")
    return item


@router.get("/api/share/quiz/{share_id}/scores")
def get_quiz_scores(share_id: str, request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    result = admin_store.get_quiz_ranking(share_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Jogo não encontrado ou expirado.")
    return result


@router.post("/api/share/quiz/{share_id}/scores")
def post_quiz_score(share_id: str, req: QuizScoreRequest, request: Request):
    """Regista o resultado de um jogador e devolve a posição e o top 10."""
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    enforce_rate_limit(request, "quiz-score", 60, 3600)
    try:
        result = admin_store.add_quiz_score(
            share_id, name=req.name, score=req.score, correct=req.correct, total=req.total
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if result is None:
        raise HTTPException(status_code=404, detail="Jogo não encontrado ou expirado.")
    return result


@router.get("/api/share/transcript/{share_id}")
def get_share_transcript(share_id: str, request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    item = admin_store.get_shared_transcript(share_id)
    if not item:
        raise HTTPException(status_code=404, detail="Partilha não encontrada ou expirada.")
    return item


@router.get("/share/{share_id}")
def share_html_page(share_id: str):
    """HTML com Open Graph para crawlers sociais (proxied from /s/:id)."""
    from fastapi.responses import HTMLResponse
    import html as html_lib

    item = admin_store.get_shared_transcript(share_id)
    if not item:
        raise HTTPException(status_code=404, detail="Partilha não encontrada ou expirada.")
    title = html_lib.escape((item.get("title") or "Transcrição")[:120])
    text = item.get("text") or ""
    desc = html_lib.escape(re.sub(r"\s+", " ", text)[:160])
    body = html_lib.escape(text)
    page_url = f"https://www.ouviescrevi.pt/s/{html_lib.escape(share_id)}"
    og_image = "https://www.ouviescrevi.pt/og/partilha.png"
    html = f"""<!DOCTYPE html>
<html lang="pt">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | Ouviescrevi</title>
  <meta name="description" content="{desc}">
  <meta name="robots" content="noindex, follow">
  <link rel="canonical" href="{page_url}">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{desc}">
  <meta property="og:url" content="{page_url}">
  <meta property="og:image" content="{og_image}">
  <meta property="og:type" content="article">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{title}">
  <meta name="twitter:description" content="{desc}">
  <link rel="stylesheet" href="https://www.ouviescrevi.pt/css/ouviescrevi.css">
</head>
<body>
  <main style="max-width:720px;margin:2rem auto;padding:0 1.25rem 3rem;font-family:system-ui,sans-serif">
    <p style="color:#64748b">Partilha pública · Ouviescrevi</p>
    <h1>{title}</h1>
    <article style="white-space:pre-wrap;line-height:1.55">{body}</article>
    <p style="margin-top:2rem"><a href="https://www.ouviescrevi.pt/index.html">Transcrever o teu áudio grátis</a></p>
    <p style="font-size:0.875rem;color:#64748b">Feito com Ouviescrevi</p>
  </main>
</body>
</html>"""
    return HTMLResponse(html)
