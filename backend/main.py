# main.py
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime, date
from dotenv import load_dotenv

import os
import tempfile
import uuid
import subprocess
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
import json
import textwrap
import shutil
import time
import math
import re
import threading

import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from openai import OpenAI

from security import (
    RateLimiter,
    client_ip,
    origin_is_allowed,
    parse_csv_env,
    safe_http_get,
    validate_public_http_url,
)
from cms import get_all_content, get_seo_overrides
from analytics import record_visit, get_visit_stats
import admin_store
from admin_routes import router as admin_router
from log_buffer import attach_memory_handler

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging robusto (ficheiro rotativo + console + envio opcional p/ Vercel) ─
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", os.path.abspath("./logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "ouviescrevi.log")

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for k in ("rid", "path", "method", "status", "ms"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)

class VercelHTTPHandler(logging.Handler):
    """Envia logs para uma Function no Vercel (aparecem no dashboard)."""
    def __init__(self, url: str, token: str | None = None, level=logging.warning):
        super().__init__(level)
        self.url = url
        self.token = token
    def emit(self, record: logging.LogRecord):
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            data = {
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            requests.post(self.url, json=data, headers=headers, timeout=2)
        except Exception:
            pass

logger = logging.getLogger("ouviescrevi")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not logger.handlers:
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
    logger.addHandler(ch)

    VERCEL_LOG_URL = os.getenv("VERCEL_LOG_URL")
    VERCEL_LOG_TOKEN = os.getenv("VERCEL_LOG_TOKEN")
    if VERCEL_LOG_URL:
        vh = VercelHTTPHandler(VERCEL_LOG_URL, VERCEL_LOG_TOKEN, level=logging.WARNING)
        vh.setFormatter(JSONFormatter())
        logger.addHandler(vh)

attach_memory_handler(logger, logging.Formatter(LOG_FORMAT, DATE_FMT))

for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
    logging.getLogger(name).setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if not logging.getLogger(name).handlers:
        logging.getLogger(name).handlers = logger.handlers
    logging.getLogger(name).propagate = False

# DB bootstrap
from database import criar_base
criar_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY no .env")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    raise RuntimeError("Falta ADMIN_TOKEN no .env")

API_TOKEN = os.getenv("API_TOKEN") or ADMIN_TOKEN
BACKOFFICE_PASSWORD = os.getenv("BACKOFFICE_PASSWORD")
if not BACKOFFICE_PASSWORD:
    raise RuntimeError("Falta BACKOFFICE_PASSWORD no .env")
admin_store.ensure_default_admin(BACKOFFICE_PASSWORD)

APP_ENV = os.getenv("APP_ENV", "development").lower()
ENABLE_DEBUG_ENDPOINTS = os.getenv("ENABLE_DEBUG_ENDPOINTS", "true" if APP_ENV == "development" else "false").lower() in ("1", "true", "yes")
ALLOWED_ORIGINS = parse_csv_env(os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,https://ouviescrevi.pt,https://www.ouviescrevi.pt",
))
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", "https://api.ouviescrevi.pt").rstrip("/")

RATE_LIMITER = RateLimiter()
RATE_LIMIT_TRANSCRIBE = int(os.getenv("RATE_LIMIT_TRANSCRIBE", "20"))
RATE_LIMIT_TRANSCRIBE_WINDOW = int(os.getenv("RATE_LIMIT_TRANSCRIBE_WINDOW", "3600"))
RATE_LIMIT_VIDEO_SUBS = int(os.getenv("RATE_LIMIT_VIDEO_SUBS", "10"))
RATE_LIMIT_VIDEO_SUBS_WINDOW = int(os.getenv("RATE_LIMIT_VIDEO_SUBS_WINDOW", "3600"))
RATE_LIMIT_AI = int(os.getenv("RATE_LIMIT_AI", "60"))
RATE_LIMIT_AI_WINDOW = int(os.getenv("RATE_LIMIT_AI_WINDOW", "3600"))
RATE_LIMIT_TRACK = int(os.getenv("RATE_LIMIT_TRACK", "300"))
RATE_LIMIT_TRACK_WINDOW = int(os.getenv("RATE_LIMIT_TRACK_WINDOW", "3600"))

# Timeouts e parâmetros
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "110"))  # por chunk
WHISPER_LANGUAGE = (os.getenv("WHISPER_LANGUAGE", "") or "").strip().lower() or None
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
WHISPER_PROMPT_OVERRIDE = (os.getenv("WHISPER_PROMPT") or "").strip() or None
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "60"))
TOTAL_TRANSCRIBE_TIMEOUT = int(os.getenv("TOTAL_TRANSCRIBE_TIMEOUT", "900"))  # watchdog global
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "600"))  # 10 min
SUBS_TIMEOUT = int(os.getenv("SUBS_TIMEOUT", "900"))  # tempo p/ queimar legendas
SUBS_BURN_PRESET = os.getenv("SUBS_BURN_PRESET", "ultrafast")
SUBS_BURN_CRF = int(os.getenv("SUBS_BURN_CRF", "23"))
SUBS_BURN_MAX_WIDTH = int(os.getenv("SUBS_BURN_MAX_WIDTH", "1280"))
SUBS_BURN_SEC_PER_SEC = float(os.getenv("SUBS_BURN_SEC_PER_SEC", "1.0"))
SUBS_TRANSCRIBE_SEC_PER_SEC = float(os.getenv("SUBS_TRANSCRIBE_SEC_PER_SEC", "0.25"))
VIDEO_CLEANUP_MAX_AGE_HOURS = float(os.getenv("VIDEO_CLEANUP_MAX_AGE_HOURS", "36"))
UPLOAD_CLEANUP_MAX_AGE_HOURS = float(os.getenv("UPLOAD_CLEANUP_MAX_AGE_HOURS", "2"))

# Modelos
SUM_MODEL = os.getenv("SUM_MODEL", "gpt-4o-mini")
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4o-mini")
COR_MODEL = os.getenv("COR_MODEL", "gpt-4o-mini")
EML_MODEL = os.getenv("EML_MODEL", "gpt-4o-mini")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY, timeout=WHISPER_TIMEOUT)

# Descobrir ffmpeg
try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None

FFMPEG = shutil.which("ffmpeg") or (get_ffmpeg_exe() if get_ffmpeg_exe else None)
if not FFMPEG:
    raise RuntimeError("ffmpeg não encontrado. Instala o binário do sistema ou mantém 'imageio-ffmpeg' no requirements.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Token", "X-Site-Session"],
)

# estáticos
STATIC_DIR = os.path.abspath("static")
VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def _on_startup() -> None:
    threading.Thread(target=_run_startup_cleanup, daemon=True).start()


@app.middleware("http")
async def admin_error_logger(request: Request, call_next):
    try:
        response = await call_next(request)
        if response.status_code >= 400 and request.url.path.startswith("/api/"):
            admin_store.log_api_error(
                request.url.path,
                response.status_code,
                "HTTP error",
                client_ip(request),
            )
        return response
    except HTTPException as exc:
        if request.url.path.startswith("/api/"):
            admin_store.log_api_error(request.url.path, exc.status_code, str(exc.detail), client_ip(request))
        raise
    except Exception as exc:
        admin_store.log_api_error(request.url.path, 500, str(exc)[:500], client_ip(request))
        raise

# ──────────────────────────────────────────────────────────────────────────────
# Helpers logging/IO
# ──────────────────────────────────────────────────────────────────────────────
def _fmt_mb(nbytes: int) -> float:
    try:
        return round(nbytes / (1024 * 1024), 2)
    except Exception:
        return 0.0

async def _stream_upload_to_disk(upload_file: UploadFile, dest_path: str, rid: str, tag: str, log_every_mb: int = 10, max_bytes: int | None = None) -> int:
    """Lê o UploadFile em chunks para disco, devolvendo bytes escritos e logando progresso."""
    bytes_written = 0
    CHUNK = 1024 * 1024
    logged_next = log_every_mb * 1024 * 1024
    try:
        await upload_file.seek(0)
        with open(dest_path, "wb") as out:
            while True:
                chunk = await upload_file.read(CHUNK)
                if not chunk:
                    break
                if max_bytes and (bytes_written + len(chunk)) > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Ficheiro > {MAX_FILE_SIZE_MB}MB. Reduz o tamanho e tenta novamente.",
                    )
                out.write(chunk)
                bytes_written += len(chunk)
                if bytes_written >= logged_next:
                    logger.info("[%s] [%s] upload parcial: %0.2f MB", rid, tag, _fmt_mb(bytes_written))
                    logged_next += log_every_mb * 1024 * 1024
        return bytes_written
    except HTTPException:
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except OSError:
            pass
        raise
    except Exception:
        logger.exception("[%s] [%s] Falha ao gravar upload (parcial=%0.2f MB)", rid, tag, _fmt_mb(bytes_written))
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except OSError:
            pass
        raise

def _max_upload_bytes() -> int:
    return MAX_FILE_SIZE_MB * 1024 * 1024

def _reject_oversized_upload(request: Request) -> None:
    cl = request.headers.get("content-length")
    if not cl:
        return
    try:
        if int(cl) > _max_upload_bytes() + (5 * 1024 * 1024):
            raise HTTPException(
                status_code=413,
                detail=f"Ficheiro > {MAX_FILE_SIZE_MB}MB. Reduz o tamanho e tenta novamente.",
            )
    except ValueError:
        return

@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.monotonic()
    extra = {"rid": rid, "path": request.url.path, "method": request.method}
    try:
        ua = request.headers.get("user-agent", "-")
        cl = request.headers.get("content-length", "-")
        ct = request.headers.get("content-type", "-")
        ref = request.headers.get("referer", "-")
        client_ip = getattr(request.client, "host", "-") if request.client else "-"
        logger.info("→ %s %s from=%s len=%s ct=%s ua=%s ref=%s", request.method, request.url.path, client_ip, cl, ct, ua, ref, extra=extra)

        response = await call_next(request)

        ms = round((time.monotonic() - start) * 1000, 1)
        extra |= {"status": response.status_code, "ms": ms}
        logger.info("← %s %s %s %sms", request.method, request.url.path, response.status_code, ms, extra=extra)
        return response
    except Exception:
        ms = round((time.monotonic() - start) * 1000, 1)
        extra |= {"status": 500, "ms": ms}
        logger.exception("✖ %s %s 500 %sms", request.method, request.url.path, ms, extra=extra)
        raise

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — autenticação e limites
# ──────────────────────────────────────────────────────────────────────────────
def extract_api_token(request: Request, body_token: str | None = None) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if body_token:
        return body_token.strip()
    return (request.headers.get("x-api-token") or "").strip()


def require_api_token(request: Request, body_token: str | None = None) -> None:
    token = extract_api_token(request, body_token)
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")


def require_admin_token(request: Request) -> None:
    token = extract_api_token(request)
    if token == ADMIN_TOKEN:
        request.state.admin_session = {"username": "env", "role": "admin"}
        request.state.admin_user = "env"
        return
    session = admin_store.resolve_session(token)
    if session:
        request.state.admin_session = session
        request.state.admin_user = session["username"]
        return
    raise HTTPException(status_code=403, detail="Acesso administrativo negado.")


def require_token(token: str):
    """Compatibilidade com rotas que recebem token no corpo JSON."""
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")


SITE_USER_ROLE = "user"
STAFF_ROLES = frozenset(admin_store.ROLE_LEVEL.keys())


def resolve_site_actor(request: Request) -> dict:
    site_token = (request.headers.get("x-site-session") or "").strip()
    if not site_token:
        return {"type": "anonymous"}
    if site_token == ADMIN_TOKEN:
        return {"type": "admin", "username": "env", "role": "admin"}
    session = admin_store.resolve_session(site_token)
    if not session:
        return {"type": "anonymous"}
    role = session["role"]
    username = session["username"]
    if role == SITE_USER_ROLE:
        return {"type": "user", "email": username, "username": username}
    if role in STAFF_ROLES:
        return {"type": "admin", "username": username, "role": role}
    return {"type": "anonymous"}


def should_notify_activity(request: Request) -> bool:
    return resolve_site_actor(request)["type"] != "admin"


def activity_actor_label(request: Request) -> str:
    actor = resolve_site_actor(request)
    if actor["type"] == "user":
        return actor.get("email") or actor.get("username") or "utilizador"
    if actor["type"] == "admin":
        return f"admin:{actor.get('username')}"
    return "anónimo"


def _transcription_has_content(transcription: str, formatted: str) -> bool:
    texto = (transcription or "").strip()
    if texto:
        return True
    formatado = (formatted or "").strip()
    if not formatado:
        return False
    if re.search(r"\[Falha no segmento\]", formatado, re.I):
        return False
    return True


def enforce_transcribe_quota(request: Request) -> dict:
    actor = resolve_site_actor(request)
    status = admin_store.transcribe_quota_status(request, actor)
    if not status.get("unlimited") and status.get("remaining", 1) <= 0:
        raise HTTPException(status_code=429, detail=status.get("message", "Limite diário atingido."))
    return status


def record_transcribe_success(
    request: Request | None = None,
    *,
    actor: dict | None = None,
    usage_key: str | None = None,
    filename: str | None = None,
    language: str | None = None,
    size_bytes: int | None = None,
    duration_sec: float | None = None,
    transcription: str = "",
    formatted: str = "",
) -> None:
    if actor is None and request is not None:
        actor = resolve_site_actor(request)
    actor = actor or {"type": "anonymous"}
    if usage_key is None and request is not None:
        usage_key, _tier = admin_store.usage_key_for_request(request, actor)
    try:
        if usage_key:
            admin_store.increment_daily_transcribe(usage_key)
    except Exception as exc:
        logger.warning("Falha ao registar quota diária: %s", exc)
    if actor.get("type") != "user" or not (transcription or formatted).strip():
        return
    try:
        admin_store.save_user_transcription(
            actor["email"],
            filename=filename,
            language=language,
            size_bytes=size_bytes,
            duration_sec=duration_sec,
            transcription=transcription,
            formatted=formatted,
        )
    except Exception as exc:
        logger.warning("Falha ao guardar histórico do utilizador: %s", exc)


def maybe_notify_activity(
    request: Request | None,
    mensagem: str,
    assunto: str = "Nova atividade no Ouviescrevi",
    *,
    actor_label: str | None = None,
    notify: bool | None = None,
) -> None:
    if notify is None:
        notify = should_notify_activity(request) if request else True
    if not notify:
        logger.debug("Notificação omitida (sessão admin): %s", assunto)
        return
    from email_notify import activity_notifications_enabled

    if not activity_notifications_enabled():
        logger.debug("Notificação de atividade desativada na config: %s", assunto)
        return
    label = actor_label
    if label is None and request:
        label = activity_actor_label(request)
    if not label:
        label = "anónimo"
    mensagem = f"{mensagem}\n\nConta: {label}"
    logger.info("A agendar notificação por email: %s → %s", assunto, label)

    def _send() -> None:
        from email_notify import send_notification_email

        ok, err = send_notification_email(mensagem, assunto, kind="activity", actor=label)
        if not ok:
            logger.warning("Falha ao enviar notificação: %s — %s", assunto, err)

    threading.Thread(target=_send, daemon=True).start()


def _run_in_background(fn) -> None:
    """Executa em thread; síncrono quando TEST_SYNC_NOTIFICATIONS=1 (pytest)."""
    if os.getenv("TEST_SYNC_NOTIFICATIONS") == "1":
        try:
            fn()
        except Exception:
            logger.exception("Notificação em background falhou")
        return
    threading.Thread(target=fn, daemon=True).start()


def enforce_rate_limit(request: Request, bucket: str, limit: int, window: int) -> None:
    RATE_LIMITER.check(client_ip(request), bucket, limit, window)


def require_debug_enabled() -> None:
    if not ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Não encontrado.")


def admin_guard(request: Request) -> None:
    require_admin_token(request)


app.include_router(admin_router, dependencies=[Depends(admin_guard)])


def get_manutencao() -> bool:
    return admin_store.get_maintenance()["manutencao"]


def get_maintenance_payload() -> dict:
    return admin_store.get_maintenance()


def require_not_maintenance() -> None:
    maint = admin_store.get_maintenance()
    if maint["manutencao"] and maint.get("block_transcribe_only", True):
        raise HTTPException(status_code=503, detail="Serviço temporariamente em manutenção. Tenta mais tarde.")

def enviar_email_assunto(mensagem: str, assunto: str = "Nova atividade no Ouviescrevi"):
    from email_notify import send_notification_email

    send_notification_email(mensagem, assunto)

def _seg_get(seg, key, default=None):
    try:
        return getattr(seg, key)
    except Exception:
        try:
            return seg.get(key, default)
        except Exception:
            return default

def _format_time(seconds: float):
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 0.0
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"

def format_segments_with_offset(segments, offset_seconds: int = 0):
    formatted = []
    for s in segments or []:
        start = _seg_get(s, "start", 0)
        text = (_seg_get(s, "text", "") or "").strip()
        if text:
            formatted.append(f"{_format_time(start + offset_seconds)} {text}")
    return "\n\n".join(formatted).strip()


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for c in text if "\u3040" <= c <= "\u9fff" or "\u30a0" <= c <= "\u30ff")
    return cjk / max(len(text), 1)


def _normalize_block(text: str) -> str:
    text = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", text.strip(), flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip().lower()


def filter_whisper_segments(segments, language: str | None = None):
    """Remove segmentos com sinais típicos de alucinação do Whisper (silêncio/ruído)."""
    raw = list(segments or [])
    non_empty = [s for s in raw if (_seg_get(s, "text", "") or "").strip()]
    if not non_empty:
        return []

    def _apply(
        items,
        *,
        no_speech_max: float,
        logprob_min: float,
        compression_max: float,
    ):
        filtered = []
        dropped = 0
        for s in items:
            text = (_seg_get(s, "text", "") or "").strip()
            if not text:
                continue
            no_speech = float(_seg_get(s, "no_speech_prob", 0) or 0)
            avg_logprob = float(_seg_get(s, "avg_logprob", 0) or 0)
            compression = float(_seg_get(s, "compression_ratio", 1) or 1)
            if no_speech > no_speech_max:
                dropped += 1
                continue
            if avg_logprob < logprob_min:
                dropped += 1
                continue
            if compression > compression_max:
                dropped += 1
                continue
            if language == "pt" and _cjk_ratio(text) > 0.2:
                dropped += 1
                continue
            filtered.append(s)
        return filtered, dropped

    filtered, dropped = _apply(non_empty, no_speech_max=0.5, logprob_min=-1.0, compression_max=2.2)
    if len(non_empty) >= 5 and len(filtered) < max(2, int(len(non_empty) * 0.2)):
        relaxed, dropped_relaxed = _apply(
            non_empty, no_speech_max=0.75, logprob_min=-1.4, compression_max=2.8
        )
        if len(relaxed) > len(filtered):
            logger.warning(
                "Whisper: filtros strict removeram %d/%d; relaxed mantém %d",
                dropped, len(non_empty), len(relaxed),
            )
            filtered = relaxed
            dropped = dropped_relaxed
    elif dropped:
        logger.info("Whisper: descartados %d segmentos (ruído/alucinação)", dropped)

    if len(non_empty) >= 3 and len(filtered) < max(2, int(len(non_empty) * 0.35)):
        logger.warning(
            "Whisper: mantendo %d/%d segmentos com filtro mínimo (muito conteúdo descartado)",
            len(non_empty), len(non_empty),
        )
        return non_empty
    return filtered


def dedupe_consecutive_blocks(text: str) -> str:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out, prev_norm = [], None
    for block in blocks:
        norm = _normalize_block(block)
        if norm and norm == prev_norm:
            continue
        prev_norm = norm or prev_norm
        out.append(block)
    return "\n\n".join(out)


def collapse_repeated_phrases(text: str, min_chars: int = 18, max_keep: int = 1) -> str:
    """Frases longas repetidas 3+ vezes (padrão de alucinação) ficam só uma vez."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    counts: dict[str, int] = {}
    for block in blocks:
        norm = _normalize_block(block)
        if len(norm) >= min_chars:
            counts[norm] = counts.get(norm, 0) + 1
    noisy = {n for n, c in counts.items() if c >= 3}
    if not noisy:
        return text
    seen: dict[str, int] = {}
    out = []
    for block in blocks:
        norm = _normalize_block(block)
        if norm in noisy:
            seen[norm] = seen.get(norm, 0) + 1
            if seen[norm] > max_keep:
                continue
        out.append(block)
    return "\n\n".join(out)


def remove_cjk_blocks(text: str) -> str:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    kept = [b for b in blocks if _cjk_ratio(_normalize_block(b)) <= 0.2]
    return "\n\n".join(kept)


def clean_transcription_text(text: str, language: str | None = None) -> str:
    if not text:
        return ""
    text = dedupe_consecutive_blocks(text)
    text = collapse_repeated_phrases(text)
    if language == "pt":
        text = remove_cjk_blocks(text)
    return text.strip()


def resolve_whisper_language(form_language: str | None) -> str | None:
    lang = (form_language or WHISPER_LANGUAGE or "").strip().lower()
    if not lang or lang in ("auto", "detect"):
        return None
    return lang


def whisper_prompt_for_language(language: str | None) -> str | None:
    if WHISPER_PROMPT_OVERRIDE:
        return WHISPER_PROMPT_OVERRIDE
    if language == "pt":
        return "Transcrição em português de Portugal de uma reunião de trabalho ou conversa."
    if language == "en":
        return "English speech transcription of a conversation or presentation."
    return None


def process_whisper_result(result, language: str | None, offset_seconds: int = 0):
    raw_segs = getattr(result, "segments", []) or []
    segs = filter_whisper_segments(raw_segs, language)
    text = " ".join((_seg_get(s, "text", "") or "").strip() for s in segs).strip()
    formatted = format_segments_with_offset(segs, offset_seconds)
    text = clean_transcription_text(text, language)
    formatted = clean_transcription_text(formatted, language)
    return text, formatted, segs

def safe_run_ffmpeg(cmd: list, desc: str, timeout: int = FFMPEG_TIMEOUT):
    t0 = time.monotonic()
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)
        dur = time.monotonic() - t0
        logger.info("FFmpeg OK (%s) em %.2fs", desc, dur)
        return cp
    except subprocess.TimeoutExpired:
        dur = time.monotonic() - t0
        logger.error("FFmpeg TIMEOUT (%s) após %.2fs", desc, dur)
        raise
    except subprocess.CalledProcessError as e:
        dur = time.monotonic() - t0
        err = (e.stderr or b"").decode(errors="ignore")
        logger.error("FFmpeg ERRO (%s) em %.2fs: %s", desc, dur, err[:1000])
        raise


def _fmt_elapsed(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def safe_run_ffmpeg_with_heartbeat(
    cmd: list,
    desc: str,
    timeout: int,
    job_id: str,
    *,
    progress_start: int = 85,
    progress_end: int = 96,
    status_message: str = "A incorporar legendas no vídeo",
) -> None:
    """Executa FFmpeg com atualizações periódicas ao job (evita UI parada)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    t0 = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(3.0):
            elapsed = time.monotonic() - t0
            span = max(progress_end - progress_start, 1)
            pct = progress_start + min(span - 1, int((elapsed / max(timeout, 30)) * span))
            _video_job_set(
                job_id,
                message=f"{status_message}… ({_fmt_elapsed(elapsed)} — pode demorar vários minutos)",
                progress=pct,
            )

    hb = threading.Thread(target=heartbeat, daemon=True)
    hb.start()
    try:
        _, stderr = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            err = (stderr or b"").decode(errors="ignore")
            raise subprocess.CalledProcessError(proc.returncode, cmd, b"", stderr)
        logger.info("FFmpeg OK (%s) em %.2fs", desc, time.monotonic() - t0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        logger.error("FFmpeg TIMEOUT (%s) após %.2fs", desc, time.monotonic() - t0)
        raise
    finally:
        stop.set()


def probe_media_duration_sec(path: str) -> float | None:
    if not path or not os.path.isfile(path) or not FFMPEG:
        return None
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
        if not m:
            return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        return None


def probe_video_dimensions(path: str) -> tuple[int | None, int | None]:
    if not path or not os.path.isfile(path) or not FFMPEG:
        return None, None
    try:
        proc = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        m = re.search(r"Stream #\d+:\d+.*Video:.*? (\d+)x(\d+)", proc.stderr or "")
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _parse_trim_form(
    trim_start: str | None,
    trim_end: str | None,
) -> tuple[float | None, float | None]:
    def _to_float(value: str | None) -> float | None:
        if value is None or str(value).strip() == "":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    start = _to_float(trim_start)
    end = _to_float(trim_end)
    if start is None and end is None:
        return None, None
    if start is not None and start < 0:
        start = 0.0
    if start is not None and end is not None and end <= start + 0.5:
        return None, None
    return start, end


def trim_media_file(
    src: str,
    dst: str,
    start_sec: float | None,
    end_sec: float | None,
    *,
    desc: str = "trim",
    audio_only: bool = False,
) -> None:
    """Corta um ficheiro de media com ffmpeg (trecho start_sec → end_sec)."""
    if start_sec is None and end_sec is None:
        raise ValueError("trim sem intervalo")
    start = max(0.0, start_sec or 0.0)
    if end_sec is not None and end_sec <= start + 0.5:
        raise ValueError("fim do trecho inválido")
    cmd = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        src,
    ]
    if end_sec is not None:
        cmd.extend(["-t", f"{(end_sec - start):.3f}"])
    if audio_only:
        cmd.extend(["-vn", "-sn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-movflags", "+faststart"])
    cmd.append(dst)
    safe_run_ffmpeg(cmd, desc=desc, timeout=FFMPEG_TIMEOUT)


def apply_upload_trim(
    src_path: str,
    start_sec: float | None,
    end_sec: float | None,
    *,
    rid: str,
    audio_only: bool = False,
) -> tuple[str, int]:
    """Aplica corte opcional; devolve (path_a_usar, size_bytes)."""
    if start_sec is None and end_sec is None:
        return src_path, os.path.getsize(src_path)
    ext = ".wav" if audio_only else (os.path.splitext(src_path)[1] or ".mp4")
    dst = os.path.join(tempfile.gettempdir(), f"trim_{uuid.uuid4()}{ext}")
    trim_media_file(src_path, dst, start_sec, end_sec, desc="trim-upload", audio_only=audio_only)
    try:
        os.remove(src_path)
    except OSError:
        pass
    size = os.path.getsize(dst)
    logger.info("[%s] Trecho cortado: %.1fs → %.1fs (%d bytes)", rid, start_sec or 0, end_sec or -1, size)
    return dst, size


def probe_video_width(path: str) -> int | None:
    w, _ = probe_video_dimensions(path)
    return w


def _hex_to_ass_colour(hex_color: str, *, alpha: int = 0) -> str:
    h = (hex_color or "#ffffff").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        h = "ffffff"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _parse_style_json(style: str | None) -> dict:
    if not style:
        return {}
    try:
        data = json.loads(style)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _style_json_to_ass_force_style(
    style: dict,
    video_width: int | None,
    video_height: int | None,
) -> str:
    font_size = int(style.get("fontSize") or 40)
    vw = video_width or 1280
    vh = video_height or 720
    preview_ref_w = min(854, vw)
    ass_font = max(12, round(font_size * vw / preview_ref_w))

    color = _hex_to_ass_colour(style.get("color") or "#ffffff")
    outline_w = int(style.get("outline") if style.get("outline") is not None else 2)
    shadow_kind = style.get("shadow") or "soft"
    shadow_depth = {"none": 0, "soft": 2, "strong": 4}.get(shadow_kind, 2)

    bg = bool(style.get("bg", True))
    bg_opacity = float(style.get("bgOpacity") if style.get("bgOpacity") is not None else 0.35)
    bg_alpha = min(255, max(0, int((1.0 - bg_opacity) * 255)))
    back_colour = _hex_to_ass_colour("#000000", alpha=bg_alpha)
    outline_colour = _hex_to_ass_colour("#000000")
    border_style = (4 if outline_w > 0 else 3) if bg else 1

    align_h = style.get("align") or "center"
    position = style.get("position") or "bottom"
    if position == "top":
        alignment = {"left": 7, "center": 8, "right": 9}.get(align_h, 8)
    elif position == "custom":
        # custom ≈ meio vertical; alinhamento horizontal mantém-se
        alignment = {"left": 4, "center": 5, "right": 6}.get(align_h, 5)
    else:
        alignment = {"left": 1, "center": 2, "right": 3}.get(align_h, 2)

    margin_v = int(style.get("marginV") if style.get("marginV") is not None else 48)
    # padding da UI → margem lateral aproximada
    padding = int(style.get("padding") if style.get("padding") is not None else 12)
    margin_lr = max(10, min(vw // 3, padding * 3))

    # maxWidthPct → margens laterais para limitar largura da caixa
    max_width_pct = style.get("maxWidthPct")
    if max_width_pct is not None:
        try:
            pct = max(20, min(100, int(max_width_pct)))
            side = int(vw * (100 - pct) / 200)
            margin_lr = max(margin_lr, side)
        except (TypeError, ValueError):
            pass

    if position == "custom":
        # centrar verticalmente com MarginV relativo à metade da altura
        margin_v = max(margin_v, int(vh * 0.08))

    return ",".join(
        [
            "FontName=DejaVu Sans",
            f"FontSize={ass_font}",
            f"PrimaryColour={color}",
            f"OutlineColour={outline_colour}",
            f"BackColour={back_colour}",
            f"BorderStyle={border_style}",
            f"Outline={outline_w}",
            f"Shadow={shadow_depth}",
            f"Alignment={alignment}",
            f"MarginV={margin_v}",
            f"MarginL={margin_lr}",
            f"MarginR={margin_lr}",
        ]
    )


def _entries_to_transcription(
    entries: list[tuple[float, float, str]],
    whisper_lang: str | None,
) -> tuple[str, str]:
    plain: list[str] = []
    lines: list[str] = []
    for start, _end, text in entries:
        t = (text or "").strip()
        if not t:
            continue
        plain.append(t)
        lines.append(f"{_format_time(start)} {t}")
    transcription = clean_transcription_text(" ".join(plain), whisper_lang)
    formatted = clean_transcription_text("\n\n".join(lines), whisper_lang)
    return transcription, formatted


def _estimate_transcribe_seconds(duration_sec: float | None, size_bytes: int | None = None) -> int:
    if duration_sec and duration_sec > 0:
        return max(15, int(duration_sec * SUBS_TRANSCRIBE_SEC_PER_SEC) + 10)
    if size_bytes and size_bytes > 0:
        mb = size_bytes / (1024 * 1024)
        return max(20, int(mb * 2) + 15)
    return 45


def _estimate_burn_seconds(duration_sec: float | None) -> int:
    if not duration_sec or duration_sec <= 0:
        return 120
    return max(30, int(duration_sec * SUBS_BURN_SEC_PER_SEC) + 20)


def _cleanup_old_video_files() -> int:
    if not os.path.isdir(VIDEO_DIR):
        return 0
    max_age = VIDEO_CLEANUP_MAX_AGE_HOURS * 3600
    now = time.time()
    removed = 0
    for name in os.listdir(VIDEO_DIR):
        path = os.path.join(VIDEO_DIR, name)
        try:
            if not os.path.isfile(path):
                continue
            if now - os.path.getmtime(path) > max_age:
                os.remove(path)
                removed += 1
        except Exception:
            pass
    if removed:
        logger.info("Limpeza static/videos: %d ficheiro(s) removido(s)", removed)
    return removed


def _cleanup_temp_uploads() -> int:
    """Remove ficheiros temporários órfãos (uploads de transcrição/legendas)."""
    tmp = tempfile.gettempdir()
    max_age = UPLOAD_CLEANUP_MAX_AGE_HOURS * 3600
    now = time.time()
    removed = 0
    prefixes = ("input_", "audio_", "subs_", "vid_", "split_")
    try:
        for name in os.listdir(tmp):
            path = os.path.join(tmp, name)
            try:
                if name.startswith("split_") and os.path.isdir(path):
                    if now - os.path.getmtime(path) > max_age:
                        shutil.rmtree(path, ignore_errors=True)
                        removed += 1
                    continue
                if not os.path.isfile(path):
                    continue
                if not name.startswith(prefixes):
                    continue
                if now - os.path.getmtime(path) > max_age:
                    os.remove(path)
                    removed += 1
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Limpeza de temporários falhou: %s", exc)
    if removed:
        logger.info(
            "Limpeza uploads temporários: %d item(ns) (>%.1fh)",
            removed,
            UPLOAD_CLEANUP_MAX_AGE_HOURS,
        )
    return removed


def _run_startup_cleanup() -> None:
    _cleanup_old_video_files()
    _cleanup_temp_uploads()


def _build_burn_subtitles_cmd(
    video_path: str,
    srt_path: str,
    out_path: str,
    *,
    force_style: str | None = None,
) -> list[str]:
    style = force_style or (
        "FontName=DejaVu Sans,FontSize=24,Outline=1,BorderStyle=1,Shadow=0,MarginV=24"
    )
    subs = f"subtitles={_escape_subtitles_path(srt_path)}:force_style='{style}'"
    width = probe_video_width(video_path)
    if SUBS_BURN_MAX_WIDTH > 0 and width and width > SUBS_BURN_MAX_WIDTH:
        vf = f"scale={SUBS_BURN_MAX_WIDTH}:-2,{subs}"
    else:
        vf = subs
    return [
        FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-threads", "0",
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", SUBS_BURN_PRESET, "-crf", str(SUBS_BURN_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        out_path,
    ]


_video_sub_jobs: dict[str, dict] = {}
_video_sub_jobs_lock = threading.Lock()


def _video_job_set(job_id: str, **kwargs) -> None:
    with _video_sub_jobs_lock:
        job = _video_sub_jobs.setdefault(job_id, {"job_log": []})
        new_msg = kwargs.get("message")
        if new_msg and new_msg != job.get("message"):
            job["stage_started_at"] = time.monotonic()
            job.setdefault("job_log", []).append(
                {"t": datetime.utcnow().strftime("%H:%M:%S"), "msg": new_msg}
            )
            job["job_log"] = job["job_log"][-40:]
        job.update(kwargs)
        job["updated_at"] = time.monotonic()


def _video_job_get(job_id: str) -> dict | None:
    with _video_sub_jobs_lock:
        job = _video_sub_jobs.get(job_id)
        if not job:
            return None
        out = dict(job)
    now = time.monotonic()
    if out.get("stage_started_at"):
        out["stage_elapsed_sec"] = int(now - out["stage_started_at"])
    if out.get("created_at"):
        out["total_elapsed_sec"] = int(now - out["created_at"])
    return out


def export_video_sub_jobs() -> list[dict]:
    with _video_sub_jobs_lock:
        rows = []
        for jid, job in _video_sub_jobs.items():
            row = {k: v for k, v in job.items() if k != "updated_at"}
            row["job_id"] = jid
            rows.append(row)
        return sorted(rows, key=lambda r: r.get("created_at", 0), reverse=True)


def _prune_video_jobs() -> None:
    cutoff = time.monotonic() - 3600
    with _video_sub_jobs_lock:
        stale = [jid for jid, j in _video_sub_jobs.items() if j.get("updated_at", 0) < cutoff]
        for jid in stale:
            _video_sub_jobs.pop(jid, None)


_transcribe_jobs: dict[str, dict] = {}
_transcribe_jobs_lock = threading.Lock()


def _transcribe_job_set(job_id: str, **kwargs) -> None:
    with _transcribe_jobs_lock:
        job = _transcribe_jobs.setdefault(job_id, {"job_log": []})
        new_msg = kwargs.get("message")
        if new_msg and new_msg != job.get("message"):
            job["stage_started_at"] = time.monotonic()
            job.setdefault("job_log", []).append(
                {"t": datetime.utcnow().strftime("%H:%M:%S"), "msg": new_msg}
            )
            job["job_log"] = job["job_log"][-40:]
        job.update(kwargs)
        job["updated_at"] = time.monotonic()


def _transcribe_job_get(job_id: str) -> dict | None:
    with _transcribe_jobs_lock:
        job = _transcribe_jobs.get(job_id)
        if not job:
            return None
        out = dict(job)
    now = time.monotonic()
    if out.get("stage_started_at"):
        out["stage_elapsed_sec"] = int(now - out["stage_started_at"])
    if out.get("created_at"):
        out["total_elapsed_sec"] = int(now - out["created_at"])
    return out


def _prune_transcribe_jobs() -> None:
    cutoff = time.monotonic() - 3600
    with _transcribe_jobs_lock:
        stale = [jid for jid, j in _transcribe_jobs.items() if j.get("updated_at", 0) < cutoff]
        for jid in stale:
            _transcribe_jobs.pop(jid, None)


def split_audio(input_path, output_dir, segment_duration=SEGMENT_DURATION):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        FFMPEG,
        "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        os.path.join(output_dir, "segment_%03d.wav"),
        "-y",
    ]
    safe_run_ffmpeg(cmd, desc="segmentacao", timeout=max(30, min(FFMPEG_TIMEOUT, segment_duration + 30)))
    segments = sorted(os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav"))
    return segments

def registar_transcricao(
    nome_ficheiro: str,
    *,
    language: str | None = None,
    size_bytes: int | None = None,
    duration_sec: float | None = None,
    processing_sec: float | None = None,
    status: str = "ok",
    error_message: str | None = None,
    ui_locale: str | None = None,
    page_path: str | None = None,
):
    admin_store.record_transcription(
        nome_ficheiro,
        language=language,
        size_bytes=size_bytes,
        duration_sec=duration_sec,
        processing_sec=processing_sec,
        status=status,
        error_message=error_message,
        ui_locale=ui_locale,
        page_path=page_path,
    )
    try:
        stats = admin_store.estimate_costs()
        visits = get_visit_stats()
        admin_store.maybe_send_alerts(
            stats["transcricoes_hoje"],
            visits["visitas_hoje"],
            enviar_email_assunto,
        )
    except Exception:
        pass

def transcrever_parte_c_com_retries(
    file_path: str,
    retries: int = 3,
    sleep_base: float = 1.0,
    timeout: int = WHISPER_TIMEOUT,
    language: str | None = None,
):
    last_err = None
    lang = resolve_whisper_language(language)
    prompt = whisper_prompt_for_language(lang)
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            kwargs = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": WHISPER_TEMPERATURE,
            }
            if lang:
                kwargs["language"] = lang
            if prompt:
                kwargs["prompt"] = prompt
            with open(file_path, "rb") as audio:
                result = client.with_options(timeout=timeout).audio.transcriptions.create(
                    file=audio,
                    **kwargs,
                )
            dur = time.monotonic() - t0
            logger.info("Whisper OK (%s) tentativa %d em %.2fs lang=%s", os.path.basename(file_path), attempt, dur, lang or "auto")
            return result
        except Exception as e:
            dur = time.monotonic() - t0
            last_err = e
            logger.warning("Whisper FALHA (%s) tentativa %d/%d em %.2fs: %s",
                           os.path.basename(file_path), attempt, retries, dur, str(e)[:300])
            time.sleep(sleep_base * (2 ** (attempt - 1)))
    raise last_err

# ── Helpers específicos p/ SRT ────────────────────────────────────────────────
def _srt_timestamp(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    if s >= 60:
        m += 1
        s -= 60
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _escape_subtitles_path(p: str) -> str:
    # Escapar caracteres que interferem com o parser do filtro
    return os.path.abspath(p).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

def _write_srt(entries: list[tuple[float, float, str]], out_path: str):
    lines: list[str] = []
    idx = 1
    for start, end, text in entries:
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{_srt_timestamp(start)} --> {_srt_timestamp(end)}")
        lines.append(text.strip())
        lines.append("")
        idx += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ──────────────────────────────────────────────────────────────────────────────
# Rotas — configuração e admin
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/api/billing/status")
def billing_status_public():
    from billing import billing_config

    return billing_config()


class BillingCheckoutRequest(BaseModel):
    success_url: str | None = None
    cancel_url: str | None = None


@app.post("/api/billing/checkout")
def billing_checkout(request: Request, body: BillingCheckoutRequest):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para subscrever o plano Pro.")
    from billing import create_checkout_session

    origin = request.headers.get("origin") or PUBLIC_API_BASE.replace("api.", "www.").rstrip("/")
    if "api." in origin:
        origin = origin.replace("api.", "www.", 1)
    # Prefer locale-aware success URL from client; fallback to PT.
    success = body.success_url or f"{origin}/precos.html?ok=1"
    cancel = body.cancel_url or f"{origin}/precos.html?cancel=1"
    try:
        session = create_checkout_session(actor["email"], success_url=success, cancel_url=cancel)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return session


@app.post("/api/billing/portal")
def billing_portal(request: Request):
    """Stripe Customer Portal — gerir/cancelar subscrição Pro."""
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para gerir a subscrição.")
    from billing import create_portal_session

    origin = request.headers.get("origin") or PUBLIC_API_BASE.replace("api.", "www.").rstrip("/")
    if "api." in origin:
        origin = origin.replace("api.", "www.", 1)
    return_url = f"{origin}/precos.html"
    try:
        session = create_portal_session(actor["email"], return_url=return_url)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return session


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request):
    from billing import handle_stripe_webhook

    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        return handle_stripe_webhook(payload, sig)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Stripe webhook erro: %s", exc)
        raise HTTPException(status_code=400, detail="Webhook inválido.") from exc


class ExportDocxRequest(BaseModel):
    text: str
    title: str | None = None


@app.post("/api/export/docx")
def export_docx_pro(request: Request, body: ExportDocxRequest):
    from billing import billing_enabled, build_docx_bytes, is_pro_user, pricing_hidden
    from fastapi.responses import Response

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto em falta.")
    actor = resolve_site_actor(request)
    if billing_enabled():
        if actor["type"] != "user" or not is_pro_user(actor.get("email") or ""):
            detail = "Exportação DOCX disponível no plano Pro. Vê ouviescrevi.pt/precos.html"
            if pricing_hidden():
                detail = "Exportação DOCX em breve."
            raise HTTPException(status_code=403, detail=detail)
    else:
        detail = "Plano Pro em breve — exportação DOCX ainda não disponível."
        if pricing_hidden():
            detail = "Exportação DOCX em breve."
        raise HTTPException(status_code=503, detail=detail)
    try:
        data = build_docx_bytes(text, title=body.title or "Transcrição Ouviescrevi")
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": 'attachment; filename="transcricao.docx"'},
    )


@app.get("/api/frontend-config")
def frontend_config(request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    return {"apiBase": PUBLIC_API_BASE, "token": API_TOKEN, "maxFileSizeMb": MAX_FILE_SIZE_MB, "pricingHidden": __import__("billing").pricing_hidden()}


class SiteRegisterRequest(BaseModel):
    email: str
    password: str
    name: str | None = None
    marketing_opt_in: bool = False


class SiteLoginRequest(BaseModel):
    email: str
    password: str
    admin: bool = False


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    password: str


class ShareTranscriptRequest(BaseModel):
    text: str
    title: str | None = None
    locale: str = "pt"


@app.post("/api/auth/register")
def site_register(req: SiteRegisterRequest, request: Request):
    enforce_rate_limit(request, "auth", 15, 3600)
    try:
        user = admin_store.register_site_user(
            req.email,
            req.password,
            req.name,
            marketing_opt_in=bool(req.marketing_opt_in),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = admin_store.create_session(user["email"], SITE_USER_ROLE, hours=720)

    def _welcome() -> None:
        from email_notify import send_welcome_email

        ok, err = send_welcome_email(user["email"], user.get("name"))
        if not ok:
            logger.warning("Falha email boas-vindas %s: %s", user["email"], err)

    _run_in_background(_welcome)
    return {
        "sessionToken": token,
        "email": user["email"],
        "name": user.get("name"),
        "role": SITE_USER_ROLE,
        "marketing_opt_in": bool(user.get("marketing_opt_in")),
    }


@app.post("/api/auth/forgot-password")
def site_forgot_password(req: ForgotPasswordRequest, request: Request):
    enforce_rate_limit(request, "auth_forgot", 10, 3600)
    email = (req.email or "").strip().lower()
    token = admin_store.create_password_reset_token(email) if email else None
    if token:
        origin = (request.headers.get("origin") or "https://www.ouviescrevi.pt").rstrip("/")
        reset_url = f"{origin}/index.html?reset={token}#reset"

        def _send() -> None:
            from email_notify import send_password_reset_email

            ok, err = send_password_reset_email(email, reset_url)
            if not ok:
                logger.warning("Falha email reset %s: %s", email, err)

        _run_in_background(_send)
    return {
        "ok": True,
        "message": "Se o email existir, enviámos um link de reposição.",
    }


@app.post("/api/auth/reset-password")
def site_reset_password(req: ResetPasswordRequest, request: Request):
    enforce_rate_limit(request, "auth_reset", 20, 3600)
    try:
        ok = admin_store.reset_password_with_token(req.token, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=400, detail="Link inválido ou expirado.")
    return {"ok": True}


@app.post("/api/share/transcript")
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
    # Prefer pretty /s/{id} (proxied to API HTML with OG tags via Cloudflare Pages)
    url = f"{origin}/s/{created['id']}"
    return {"ok": True, "id": created["id"], "url": url, "expires_at": created["expires_at"]}


@app.get("/api/share/transcript/{share_id}")
def get_share_transcript(share_id: str, request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    item = admin_store.get_shared_transcript(share_id)
    if not item:
        raise HTTPException(status_code=404, detail="Partilha não encontrada ou expirada.")
    return item


@app.get("/share/{share_id}")
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


@app.post("/api/auth/login")
def site_login(req: SiteLoginRequest, request: Request):
    enforce_rate_limit(request, "auth", 30, 3600)
    ident = (req.email or "").strip()
    password = req.password or ""
    if not ident or not password:
        raise HTTPException(status_code=400, detail="Email e palavra-passe são obrigatórios.")
    if req.admin:
        user = admin_store.authenticate_user(ident, password)
        if not user and password == BACKOFFICE_PASSWORD and ident in ("", "admin"):
            token = admin_store.create_session("admin", "admin")
            admin_store.log_audit_login("admin")
            return {
                "sessionToken": token,
                "username": "admin",
                "role": "admin",
                "isStaff": True,
            }
        if not user:
            raise HTTPException(status_code=403, detail="Credenciais de administrador inválidas.")
        token = admin_store.create_session(user["username"], user["role"])
        admin_store.log_audit_login(user["username"])
        return {
            "sessionToken": token,
            "username": user["username"],
            "role": user["role"],
            "isStaff": True,
        }
    user = admin_store.authenticate_site_user(ident, password)
    if not user:
        raise HTTPException(status_code=403, detail="Email ou palavra-passe incorretos.")
    token = admin_store.create_session(user["email"], SITE_USER_ROLE, hours=720)
    return {
        "sessionToken": token,
        "email": user["email"],
        "name": user.get("name"),
        "role": SITE_USER_ROLE,
        "isStaff": False,
    }


@app.post("/api/auth/login")
def site_login(req: SiteLoginRequest, request: Request):
    enforce_rate_limit(request, "auth", 30, 3600)
    ident = (req.email or "").strip()
    password = req.password or ""
    if not ident or not password:
        raise HTTPException(status_code=400, detail="Email e palavra-passe são obrigatórios.")
    if req.admin:
        user = admin_store.authenticate_user(ident, password)
        if not user and password == BACKOFFICE_PASSWORD and ident in ("", "admin"):
            token = admin_store.create_session("admin", "admin")
            admin_store.log_audit_login("admin")
            return {
                "sessionToken": token,
                "username": "admin",
                "role": "admin",
                "isStaff": True,
            }
        if not user:
            raise HTTPException(status_code=403, detail="Credenciais de administrador inválidas.")
        token = admin_store.create_session(user["username"], user["role"])
        admin_store.log_audit_login(user["username"])
        return {
            "sessionToken": token,
            "username": user["username"],
            "role": user["role"],
            "isStaff": True,
        }
    user = admin_store.authenticate_site_user(ident, password)
    if not user:
        raise HTTPException(status_code=403, detail="Email ou palavra-passe incorretos.")
    token = admin_store.create_session(user["email"], SITE_USER_ROLE, hours=720)
    return {
        "sessionToken": token,
        "email": user["email"],
        "name": user.get("name"),
        "role": SITE_USER_ROLE,
        "isStaff": False,
    }


@app.get("/api/auth/me")
def site_me(request: Request):
    actor = resolve_site_actor(request)
    if actor["type"] == "anonymous":
        quota = admin_store.transcribe_quota_status(request, actor)
        return {"loggedIn": False, "quota": quota}
    out = {"loggedIn": True, "type": actor["type"]}
    out["quota"] = admin_store.transcribe_quota_status(request, actor)
    if actor["type"] == "user":
        out["email"] = actor.get("email")
        out["isStaff"] = False
        from billing import get_user_plan

        out["plan"] = get_user_plan(actor.get("email") or "")
    else:
        out["username"] = actor.get("username")
        out["role"] = actor.get("role")
        out["isStaff"] = True
    return out


@app.get("/api/usage")
def api_usage(request: Request):
    actor = resolve_site_actor(request)
    return admin_store.transcribe_quota_status(request, actor)


@app.get("/api/auth/history")
def user_history(request: Request, limit: int = 30, offset: int = 0):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para ver o histórico.")
    items = admin_store.list_user_transcriptions(actor["email"], limit=limit, offset=offset)
    return {"items": items}


@app.get("/api/auth/history/{item_id}")
def user_history_item(request: Request, item_id: int):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para ver o histórico.")
    row = admin_store.get_user_transcription(actor["email"], item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")
    return row


@app.delete("/api/auth/history/{item_id}")
def user_history_delete(request: Request, item_id: int):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para apagar do histórico.")
    if not admin_store.delete_user_transcription(actor["email"], item_id):
        raise HTTPException(status_code=404, detail="Transcrição não encontrada.")
    return {"ok": True}


@app.get("/api/auth/corrections")
def user_corrections(request: Request, limit: int = 30, offset: int = 0):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para ver o histórico.")
    items = admin_store.list_user_corrections(actor["email"], limit=limit, offset=offset)
    return {"items": items}


@app.get("/api/auth/corrections/{item_id}")
def user_correction_item(request: Request, item_id: int):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para ver o histórico.")
    row = admin_store.get_user_correction(actor["email"], item_id)
    if not row:
        raise HTTPException(status_code=404, detail="Correção não encontrada.")
    return row


@app.delete("/api/auth/corrections/{item_id}")
def user_correction_delete(request: Request, item_id: int):
    actor = resolve_site_actor(request)
    if actor["type"] != "user":
        raise HTTPException(status_code=403, detail="Inicia sessão para apagar do histórico.")
    if not admin_store.delete_user_correction(actor["email"], item_id):
        raise HTTPException(status_code=404, detail="Correção não encontrada.")
    return {"ok": True}


class AdminLoginRequest(BaseModel):
    password: str
    username: str | None = None


@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest):
    if req.username:
        user = admin_store.authenticate_user(req.username, req.password)
        if not user:
            raise HTTPException(status_code=403, detail="Credenciais inválidas.")
        token = admin_store.create_session(user["username"], user["role"])
        admin_store.log_audit_login(user["username"])
        return {"ok": True, "adminToken": token, "role": user["role"], "username": user["username"]}
    if req.password == BACKOFFICE_PASSWORD:
        token = admin_store.create_session("admin", "admin")
        admin_store.log_audit_login("admin")
        return {"ok": True, "adminToken": token, "role": "admin", "username": "admin"}
    raise HTTPException(status_code=403, detail="Credenciais inválidas.")


class TrackVisitRequest(BaseModel):
    path: str = "/"
    referrer: str | None = None
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None


@app.post("/api/track-visit")
def track_visit(request: Request, body: TrackVisitRequest):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    enforce_rate_limit(request, "track", RATE_LIMIT_TRACK, RATE_LIMIT_TRACK_WINDOW)
    path = (body.path or "/").strip()[:500] or "/"
    record_visit(
        path,
        client_ip(request),
        referrer=body.referrer or request.headers.get("referer"),
        user_agent=request.headers.get("user-agent"),
        utm_source=body.utm_source,
        utm_medium=body.utm_medium,
        utm_campaign=body.utm_campaign,
    )
    return {"ok": True}


class SuggestionRequest(BaseModel):
    nome: str | None = None
    mensagem: str
    lang: str = "pt"


@app.post("/api/suggestions")
def public_suggestion(request: Request, body: SuggestionRequest):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    if not (body.mensagem or "").strip():
        raise HTTPException(status_code=400, detail="Mensagem vazia.")
    sid = admin_store.add_suggestion(body.nome, body.mensagem.strip(), body.lang or "pt")
    referer = request.headers.get("referer") or request.headers.get("Referer")
    msg = body.mensagem.strip()
    lang = body.lang or "pt"
    nome = body.nome

    def _notify() -> None:
        from email_notify import send_suggestion_notification

        ok, err = send_suggestion_notification(sid, nome, msg, lang, referer)
        if not ok:
            logger.warning("Falha email sugestão #%s: %s", sid, err)

    _run_in_background(_notify)
    return {"ok": True, "id": sid}


@app.get("/api/site-content")
def site_content(request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    maint = get_maintenance_payload()
    return {
        "content": get_all_content(),
        "manutencao": maint["manutencao"],
        "maintenance_message": maint.get("maintenance_message") or "",
        "block_transcribe_only": maint.get("block_transcribe_only", True),
        "banner": admin_store.get_active_banner(),
        "seo": get_seo_overrides(),
    }


@app.get("/debug")
def debug():
    require_debug_enabled()
    return {"status": "OK", "versao": "1.6"}


def _normalize_ui_locale(raw: str | None) -> str | None:
    loc = (raw or "").strip().lower()[:8]
    if loc in ("pt", "en", "es", "fr", "de"):
        return loc
    return None


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    token: str | None = Form(None),
    language: str | None = Form(None),
    trim_start_sec: str | None = Form(None),
    trim_end_sec: str | None = Form(None),
    ui_locale: str | None = Form(None),
    page_path: str | None = Form(None),
):
    """
    Upload → job_id imediato → processamento em segundo plano.
    Consultar GET /transcribe/jobs/{job_id} até status=completed.
    """
    require_api_token(request, token)
    require_not_maintenance()
    enforce_transcribe_quota(request)
    enforce_rate_limit(request, "transcribe", RATE_LIMIT_TRANSCRIBE, RATE_LIMIT_TRANSCRIBE_WINDOW)
    rid = str(uuid.uuid4())
    whisper_lang = resolve_whisper_language(language)
    logger.info(
        "[%s] Upload recebido (transcribe): nome=%s ct=%s cl=%s lang=%s",
        rid, file.filename, file.content_type, request.headers.get("content-length"), whisper_lang or "auto",
    )
    _reject_oversized_upload(request)

    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4()}{orig_ext}")
    try:
        written = await _stream_upload_to_disk(file, tmp_path, rid, "transcribe", max_bytes=_max_upload_bytes())
        if written == 0:
            logger.error("[%s] Upload vazio (0 bytes) em /transcribe", rid)
            raise HTTPException(status_code=400, detail="Upload vazio.")
        size_mb = _fmt_mb(written)
        logger.info("[%s] Upload guardado em disco: %0.2f MB", rid, size_mb)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] Falha ao gravar upload", rid)
        raise HTTPException(status_code=400, detail=f"Falha ao gravar ficheiro: {e}") from e

    if size_mb > MAX_FILE_SIZE_MB:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=413,
            detail=f"Ficheiro demasiado grande ({size_mb:.0f} MB). O limite é {MAX_FILE_SIZE_MB} MB.",
        )

    trim_start, trim_end = _parse_trim_form(trim_start_sec, trim_end_sec)
    if trim_start is not None or trim_end is not None:
        try:
            tmp_path, written = apply_upload_trim(
                tmp_path, trim_start, trim_end, rid=rid, audio_only=True
            )
            size_mb = _fmt_mb(written)
        except Exception as exc:
            logger.warning("[%s] Falha ao cortar trecho antes de transcrever: %s", rid, exc)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=400,
                detail="Não foi possível cortar o trecho pedido. Tenta outras marcas de início/fim.",
            ) from exc

    actor = resolve_site_actor(request)
    usage_key, _tier = admin_store.usage_key_for_request(request, actor)
    notify_email = should_notify_activity(request)
    actor_label = activity_actor_label(request)
    duration_sec = probe_media_duration_sec(tmp_path)
    estimate_sec = _estimate_transcribe_seconds(duration_sec, written)
    locale_norm = _normalize_ui_locale(ui_locale)
    path_norm = (page_path or "").strip()[:500] or None

    job_id = str(uuid.uuid4())
    _prune_transcribe_jobs()
    _transcribe_job_set(
        job_id,
        status="processing",
        message="Ficheiro recebido — a iniciar transcrição…",
        progress=8,
        rid=rid,
        created_at=time.monotonic(),
        filename=file.filename or "sem_nome",
        duration_sec=duration_sec,
        estimate_transcribe_sec=estimate_sec,
    )
    threading.Thread(
        target=_execute_transcribe_job,
        args=(
            job_id,
            rid,
            tmp_path,
            file.filename or "sem_nome",
            written,
            whisper_lang,
            actor,
            usage_key,
            notify_email,
            actor_label,
            locale_norm,
            path_norm,
        ),
        daemon=True,
    ).start()
    return {
        "job_id": job_id,
        "status": "processing",
        "rid": rid,
        "estimate_transcribe_sec": estimate_sec,
        "duration_sec": duration_sec,
    }


@app.get("/transcribe/jobs/{job_id}")
def transcribe_job_status(job_id: str, request: Request):
    require_api_token(request)
    job = _transcribe_job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabalho não encontrado ou expirado.")
    return job


def _execute_transcribe_job(
    job_id: str,
    rid: str,
    tmp_path: str,
    filename: str,
    written: int,
    whisper_lang: str | None,
    actor: dict,
    usage_key: str,
    notify_email: bool,
    actor_label: str,
    ui_locale: str | None = None,
    page_path: str | None = None,
) -> None:
    t_start = time.monotonic()
    audio_wav_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.wav")
    split_dir = tempfile.mkdtemp(prefix="split_")
    converted_ok = False
    try:
        _transcribe_job_set(job_id, message="A converter áudio…", progress=15)
        try:
            conv = [
                FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_path,
                "-vn", "-sn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_wav_path,
            ]
            safe_run_ffmpeg(conv, desc="conversao-wav", timeout=FFMPEG_TIMEOUT)
            converted_ok = True
        except Exception:
            converted_ok = False
            logger.warning("[%s] Conversão WAV falhou; seguir com original.", rid)

        parts: list[str] = []
        used_source = None
        watchdog_hit = False
        try:
            _transcribe_job_set(job_id, message="A preparar segmentos de áudio…", progress=20)
            source_for_split = audio_wav_path if converted_ok else tmp_path
            parts = split_audio(source_for_split, split_dir)
            used_source = source_for_split
            logger.info("[%s] Segmentos criados: %d", rid, len(parts))
        except Exception as e:
            logger.warning("[%s] Falha ao partir áudio (%s). Vai sem split. Erro: %s", rid, filename, str(e)[:300])
            parts = []

        if not parts:
            used_source = audio_wav_path if converted_ok else tmp_path
            parts = [used_source]

        full_text_chunks, formatted_chunks = [], []
        offset_seconds = 0
        failed_segments = 0
        processed_segments = 0
        quota_exceeded = False
        duration_sec = None
        total_parts = len(parts)

        for idx, part in enumerate(parts):
            if (time.monotonic() - t_start) > TOTAL_TRANSCRIBE_TIMEOUT:
                watchdog_hit = True
                logger.error("[%s] Watchdog TOTAL_TRANSCRIBE_TIMEOUT atingido.", rid)
                break
            pct = 22 + int((idx / max(1, total_parts)) * 68)
            _transcribe_job_set(
                job_id,
                message=f"A transcrever segmento {idx + 1}/{total_parts}…",
                progress=pct,
            )
            try:
                result = transcrever_parte_c_com_retries(
                    part, retries=3, sleep_base=1.0, timeout=WHISPER_TIMEOUT, language=whisper_lang
                )
                text_piece, formatted_piece, kept_segs = process_whisper_result(
                    result, whisper_lang, offset_seconds
                )
                full_text_chunks.append(text_piece)
                formatted_chunks.append(formatted_piece)
                logger.info(
                    "[%s] Chunk %d/%d transcrito. segs=%d len(text)=%d",
                    rid, idx + 1, total_parts, len(kept_segs), len(text_piece),
                )
            except Exception as e:
                failed_segments += 1
                logger.exception("[%s] Erro ao transcrever parte %d (%s)", rid, idx, os.path.basename(part))
                formatted_chunks.append(f"{_format_time(offset_seconds)} [Falha no segmento]")
                err_msg = str(e).lower()
                if "insufficient_quota" in err_msg or "exceeded your current quota" in err_msg:
                    quota_exceeded = True
            finally:
                processed_segments += 1
                if total_parts > 1:
                    offset_seconds += SEGMENT_DURATION

        try:
            dur_src = used_source or (audio_wav_path if converted_ok else tmp_path)
            duration_sec = probe_media_duration_sec(dur_src)
            if duration_sec is None and parts:
                duration_sec = float(len(parts) * SEGMENT_DURATION)
            processing_sec = time.monotonic() - t_start
            registar_transcricao(
                filename,
                language=whisper_lang,
                size_bytes=written,
                duration_sec=duration_sec,
                processing_sec=round(processing_sec, 2),
                status="ok",
                error_message=None if not quota_exceeded else "insufficient_quota",
                ui_locale=ui_locale,
                page_path=page_path,
            )
        except Exception as e:
            logger.warning("[%s] Falha ao registar na DB: %s", rid, e)

        try:
            maybe_notify_activity(
                None,
                f"Nova transcrição recebida: {filename}",
                "Nova transcrição no Ouviescrevi",
                actor_label=actor_label,
                notify=notify_email,
            )
        except Exception as e:
            logger.warning("[%s] Falha ao agendar email de notificação: %s", rid, e)

        transcription_out = clean_transcription_text(
            "\n".join(t for t in full_text_chunks if t).strip(), whisper_lang
        )
        formatted_out = clean_transcription_text(
            "\n\n".join(t for t in formatted_chunks if t).strip(), whisper_lang
        )

        warning = None
        if quota_exceeded:
            warning = "Conta OpenAI sem créditos (insufficient_quota). Adiciona billing em platform.openai.com."
        elif failed_segments > 0:
            warning = f"{failed_segments} de {processed_segments} segmentos falharam (aplicado retry/fallback)."
        if watchdog_hit:
            extra = "Tempo total excedido (parcial devolvido)."
            warning = f"{warning} {extra}" if warning else extra

        if _transcription_has_content(transcription_out, formatted_out):
            record_transcribe_success(
                actor=actor,
                usage_key=usage_key,
                filename=filename,
                language=whisper_lang,
                size_bytes=written,
                duration_sec=duration_sec,
                transcription=transcription_out,
                formatted=formatted_out,
            )

        if not _transcription_has_content(transcription_out, formatted_out) and not warning:
            _transcribe_job_set(
                job_id,
                status="failed",
                progress=100,
                error="Não foi possível obter a transcrição.",
                message="Falha na transcrição.",
            )
            return

        _transcribe_job_set(
            job_id,
            status="completed",
            progress=100,
            message="Transcrição concluída.",
            transcription=transcription_out,
            formatted=formatted_out,
            warning=warning,
            duration_sec=duration_sec,
            processing_ms=int((time.monotonic() - t_start) * 1000),
        )
        logger.info(
            "[%s] FIM transcribe job %s em %.2fs | processed=%d failed=%d",
            rid, job_id, time.monotonic() - t_start, processed_segments, failed_segments,
        )
    except Exception as e:
        logger.exception("[%s] Erro inesperado no job transcribe %s", rid, job_id)
        _transcribe_job_set(
            job_id,
            status="failed",
            progress=100,
            error=str(e),
            message="Erro ao processar o ficheiro.",
        )
    finally:
        for p in (audio_wav_path, tmp_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass
        try:
            for f in os.listdir(split_dir):
                try:
                    os.remove(os.path.join(split_dir, f))
                except OSError:
                    pass
            os.rmdir(split_dir)
        except OSError:
            pass


# ── NOVO: Vídeo com legendas embutidas ───────────────────────────────────────
def _execute_video_subs_job(
    job_id: str,
    rid: str,
    tmp_video: str,
    filename: str,
    whisper_lang: str | None,
    written: int,
    t_start: float,
    style_json: str | None,
    burn_mp4: bool,
    notify_email: bool,
    actor_label: str,
    actor_snapshot: dict | None = None,
    usage_key: str | None = None,
) -> None:
    audio_wav_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4()}.wav")
    split_dir = tempfile.mkdtemp(prefix="subs_split_")
    srt_tmp = None
    try:
        duration_sec = probe_media_duration_sec(tmp_video)
        est_transcribe = _estimate_transcribe_seconds(duration_sec)
        est_burn = _estimate_burn_seconds(duration_sec) if burn_mp4 else 0
        _video_job_set(
            job_id,
            status="processing",
            message="A extrair áudio do vídeo…",
            progress=12,
            duration_sec=duration_sec,
            estimate_transcribe_sec=est_transcribe,
            estimate_burn_sec=est_burn,
            estimate_total_sec=est_transcribe + est_burn,
            burn_mp4=burn_mp4,
        )
        conv = [
            FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video, "-vn", "-sn",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_wav_path,
        ]
        safe_run_ffmpeg(conv, desc="audio p/ subs (wav)", timeout=max(60, FFMPEG_TIMEOUT))

        _video_job_set(job_id, message="A transcrever o áudio…", progress=25)
        parts = split_audio(audio_wav_path, split_dir)
        if not parts:
            parts = [audio_wav_path]
        entries: list[tuple[float, float, str]] = []
        offset_seconds = 0
        failed_segments = 0

        for idx, part in enumerate(parts):
            if (time.monotonic() - t_start) > TOTAL_TRANSCRIBE_TIMEOUT:
                logger.error("[%s] [video-subs] Watchdog TOTAL timeout", rid)
                break
            pct = 25 + int((idx / max(len(parts), 1)) * 40)
            _video_job_set(job_id, message=f"A transcrever segmento {idx + 1}/{len(parts)}…", progress=pct)
            try:
                result = transcrever_parte_c_com_retries(
                    part, retries=3, sleep_base=1.0, timeout=WHISPER_TIMEOUT, language=whisper_lang
                )
                segs = filter_whisper_segments(getattr(result, "segments", []) or [], whisper_lang)
                for s in segs:
                    st = float(_seg_get(s, "start", 0.0)) + offset_seconds
                    en = float(_seg_get(s, "end", st + 0.01)) + offset_seconds
                    tx = (_seg_get(s, "text", "") or "").strip()
                    if tx:
                        entries.append((st, en, tx))
                logger.info("[%s] [video-subs] chunk %d/%d OK c/ %d segmentos", rid, idx + 1, len(parts), len(segs))
            except Exception:
                logger.exception("[%s] [video-subs] Erro a transcrever chunk %d", rid, idx + 1)
                failed_segments += 1
            finally:
                if len(parts) > 1:
                    offset_seconds += SEGMENT_DURATION

        _video_job_set(job_id, message="A gerar ficheiro de legendas…", progress=72)
        base = str(uuid.uuid4())
        srt_tmp = os.path.join(tempfile.gettempdir(), f"{base}.srt")
        _write_srt(entries, srt_tmp)

        srt_out = os.path.join(VIDEO_DIR, f"{base}.srt")
        shutil.copyfile(srt_tmp, srt_out)

        transcription, formatted = _entries_to_transcription(entries, whisper_lang)
        if duration_sec is None and parts:
            duration_sec = float(len(parts) * SEGMENT_DURATION)

        srt_result = {
            "status": "srt_ready" if burn_mp4 else "completed",
            "message": "Legendas SRT prontas."
            + (" A gerar vídeo MP4 em segundo plano…" if burn_mp4 else ""),
            "progress": 78 if burn_mp4 else 100,
            "srt_url": f"/static/videos/{os.path.basename(srt_out)}",
            "transcription": transcription,
            "formatted": formatted,
            "rid": rid,
            "burn_mp4": burn_mp4,
        }
        if failed_segments:
            srt_result["note"] = f"Alguns segmentos falharam ({failed_segments})."
        _video_job_set(job_id, **srt_result)

        if not burn_mp4:
            processing_ms = round((time.monotonic() - t_start) * 1000)
            _video_job_set(job_id, processing_ms=processing_ms)
            if _transcription_has_content(transcription, formatted):
                record_transcribe_success(
                    actor=actor_snapshot,
                    usage_key=usage_key,
                    filename=f"{filename} [legendado]",
                    language=whisper_lang,
                    size_bytes=written,
                    duration_sec=duration_sec,
                    transcription=transcription,
                    formatted=formatted,
                )
            _schedule_video_subs_notify(
                rid, filename, whisper_lang, written, duration_sec, t_start,
                notify_email=notify_email, actor_label=actor_label,
            )
            return

        _video_job_set(
            job_id,
            status="burning",
            message="A incorporar legendas no vídeo…",
            progress=85,
        )
        out_video = os.path.join(VIDEO_DIR, f"{base}.mp4")
        vw, vh = probe_video_dimensions(tmp_video)
        force_style = _style_json_to_ass_force_style(_parse_style_json(style_json), vw, vh)
        burn = _build_burn_subtitles_cmd(tmp_video, srt_tmp, out_video, force_style=force_style)
        warning = None
        out_video_url = None
        try:
            safe_run_ffmpeg_with_heartbeat(
                burn,
                desc="queimar-legendas",
                timeout=SUBS_TIMEOUT,
                job_id=job_id,
                status_message="A incorporar legendas no vídeo",
            )
            logger.info("[%s] [video-subs] Vídeo legendado gerado", rid)
            out_video_url = f"/static/videos/{os.path.basename(out_video)}"
        except Exception:
            warning = "Não foi possível embutir as legendas (FFmpeg/libass). A entregar apenas o .srt."
            logger.warning("[%s] [video-subs] Falha a queimar legendas. Fallback SRT.", rid)

        processing_ms = round((time.monotonic() - t_start) * 1000)
        result = {
            "status": "completed",
            "message": "Legendas prontas.",
            "progress": 100,
            "srt_url": srt_result["srt_url"],
            "transcription": transcription,
            "formatted": formatted,
            "rid": rid,
            "processing_ms": processing_ms,
            "burn_mp4": True,
        }
        if out_video_url:
            result["video_url"] = out_video_url
        if warning:
            result["warning"] = warning
        if failed_segments:
            result["note"] = f"Alguns segmentos falharam ({failed_segments})."
        _video_job_set(job_id, **result)
        if _transcription_has_content(transcription, formatted):
            record_transcribe_success(
                actor=actor_snapshot,
                usage_key=usage_key,
                filename=f"{filename} [legendado]",
                language=whisper_lang,
                size_bytes=written,
                duration_sec=duration_sec,
                transcription=transcription,
                formatted=formatted,
            )
        _schedule_video_subs_notify(
            rid, filename, whisper_lang, written, duration_sec, t_start,
            notify_email=notify_email, actor_label=actor_label,
        )
    except Exception as e:
        logger.exception("[%s] [video-subs] job %s falhou", rid, job_id)
        _video_job_set(job_id, status="failed", error=str(e), message="Falha ao processar o vídeo.")
    finally:
        for p in (audio_wav_path, tmp_video):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        try:
            for f in os.listdir(split_dir):
                try:
                    os.remove(os.path.join(split_dir, f))
                except Exception:
                    pass
            os.rmdir(split_dir)
        except Exception:
            pass
        try:
            if srt_tmp and os.path.exists(srt_tmp):
                os.remove(srt_tmp)
        except Exception:
            pass


def _schedule_video_subs_notify(
    rid: str,
    filename: str,
    whisper_lang: str | None,
    written: int,
    duration_sec: float | None,
    t_start: float,
    *,
    notify_email: bool = True,
    actor_label: str = "anónimo",
) -> None:
    def _post_video_subs_notify() -> None:
        try:
            registar_transcricao(
                filename + " [legendado]",
                language=whisper_lang,
                size_bytes=written,
                duration_sec=duration_sec,
                processing_sec=round(time.monotonic() - t_start, 2),
                status="ok",
            )
        except Exception as e:
            logger.warning("[%s] [video-subs] Falha ao registar DB: %s", rid, e)
        maybe_notify_activity(
            None,
            f"Vídeo legendado gerado: {filename}",
            "Vídeo legendado no Ouviescrevi",
            actor_label=actor_label,
            notify=notify_email,
        )

    threading.Thread(target=_post_video_subs_notify, daemon=True).start()


@app.post("/video-subs")
async def video_subs(
    request: Request,
    file: UploadFile = File(...),
    style: str | None = Form(None),
    burn_mp4: str | None = Form("true"),
    token: str | None = Form(None),
    language: str | None = Form(None),
    trim_start_sec: str | None = Form(None),
    trim_end_sec: str | None = Form(None),
):
    require_api_token(request, token)
    require_not_maintenance()
    enforce_transcribe_quota(request)
    enforce_rate_limit(request, "video-subs", RATE_LIMIT_VIDEO_SUBS, RATE_LIMIT_VIDEO_SUBS_WINDOW)
    whisper_lang = resolve_whisper_language(language)
    want_burn_mp4 = str(burn_mp4 or "true").strip().lower() not in ("0", "false", "no", "off")
    actor = resolve_site_actor(request)
    usage_key, _tier = admin_store.usage_key_for_request(request, actor)
    notify_email = should_notify_activity(request)
    actor_label = activity_actor_label(request)
    threading.Thread(target=_run_startup_cleanup, daemon=True).start()
    """
    Upload de vídeo → resposta imediata com job_id → processamento em segundo plano.
    Consultar GET /video-subs/jobs/{job_id} até status=completed.
    """
    rid = str(uuid.uuid4())
    t_start = time.monotonic()

    ua = request.headers.get("user-agent", "-")
    cl = request.headers.get("content-length", "-")
    ct = request.headers.get("content-type", "-")
    client_ip = getattr(request.client, "host", "-") if request.client else "-"
    logger.info("[%s] [video-subs] REQUEST from=%s len=%s ct=%s ua=%s", rid, client_ip, cl, ct, ua)
    logger.info(
        "[%s] [video-subs] Upload: %s (%s) style=%s",
        rid, file.filename, file.content_type,
        (style[:120] + "…") if style and len(style) > 120 else style,
    )
    _reject_oversized_upload(request)

    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if orig_ext not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        orig_ext = ".mp4"
    tmp_video = os.path.join(tempfile.gettempdir(), f"vid_{uuid.uuid4()}{orig_ext}")
    try:
        written = await _stream_upload_to_disk(file, tmp_video, rid, "video-subs", max_bytes=_max_upload_bytes())
        if written == 0:
            logger.error("[%s] [video-subs] Upload vazio (0 bytes).", rid)
            raise HTTPException(status_code=400, detail="Upload vazio.")
        size_mb = _fmt_mb(written)
        logger.info("[%s] [video-subs] Guardado: %0.2f MB", rid, size_mb)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] [video-subs] Falha a gravar vídeo", rid)
        raise HTTPException(status_code=400, detail=f"Falha ao gravar vídeo: {e}")

    if size_mb > MAX_FILE_SIZE_MB:
        try:
            os.remove(tmp_video)
        except Exception:
            pass
        raise HTTPException(status_code=413, detail=f"Ficheiro > {MAX_FILE_SIZE_MB}MB. Reduz o tamanho e tenta novamente.")

    trim_start, trim_end = _parse_trim_form(trim_start_sec, trim_end_sec)
    if trim_start is not None or trim_end is not None:
        try:
            tmp_video, written = apply_upload_trim(
                tmp_video, trim_start, trim_end, rid=rid, audio_only=False
            )
            size_mb = _fmt_mb(written)
        except Exception as exc:
            logger.warning("[%s] [video-subs] Falha ao cortar trecho: %s", rid, exc)
            raise HTTPException(
                status_code=400,
                detail="Não foi possível cortar o trecho pedido. Ajusta início e fim.",
            ) from exc

    job_id = str(uuid.uuid4())
    _prune_video_jobs()
    _video_job_set(
        job_id,
        status="processing",
        message="Ficheiro recebido — a iniciar processamento…",
        progress=5,
        rid=rid,
        created_at=time.monotonic(),
        filename=file.filename or "sem_nome",
    )
    threading.Thread(
        target=_execute_video_subs_job,
        args=(
            job_id, rid, tmp_video, file.filename or "sem_nome",
            whisper_lang, written, t_start, style, want_burn_mp4,
            notify_email, actor_label, dict(actor), usage_key,
        ),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "processing", "rid": rid}


@app.get("/video-subs/jobs/{job_id}")
def video_subs_job_status(job_id: str, request: Request):
    require_api_token(request)
    job = _video_job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Tarefa não encontrada ou expirada.")
    return {k: v for k, v in job.items() if k != "updated_at"}

# ── IA payloads ───────────────────────────────────────────────────────────────
class DiarizeRequest(BaseModel):
    text: str
    token: str = ""
    names: list[str] | None = None
    language: str = "pt"


@app.post("/api/diarize")
async def diarize_speakers(req: DiarizeRequest, request: Request):
    """Atribui locutores ao texto com timestamps via GPT (com fallback heurístico)."""
    require_api_token(request, req.token or None)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Texto em falta.")
    names = [n.strip() for n in (req.names or []) if n and str(n).strip()]
    if len(names) < 2:
        names = ["João", "Maria"] if (req.language or "pt").startswith("pt") else ["Speaker 1", "Speaker 2"]
    names = names[:6]

    def _heuristic() -> str:
        lines_out = []
        idx = 0
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^\[(\d{2}:\d{2})\]\s*(.*)$", line)
            if m:
                name = names[idx % len(names)]
                idx += 1
                rest = m.group(2)
                # evitar duplicar nome se já existir
                if re.match(rf"^{re.escape(name)}\s*:", rest):
                    lines_out.append(line)
                else:
                    lines_out.append(f"[{m.group(1)}] {name}: {rest}".rstrip())
            else:
                lines_out.append(line)
        return "\n".join(lines_out).strip()

    lang = (req.language or "pt").lower()
    sys = (
        "You assign speakers to a timestamped transcript. "
        "Keep every line that starts with [MM:SS]. "
        f"Use only these speaker names in order of turns: {', '.join(names)}. "
        "Format each line as: [MM:SS] Name: text. "
        "Group consecutive lines from the same speaker when the dialogue clearly continues. "
        "Do not invent timestamps. Return only the labeled transcript."
    )
    if lang.startswith("pt"):
        sys = (
            "Atribui locutores a uma transcrição com timestamps. "
            "Mantém todas as linhas que começam com [MM:SS]. "
            f"Usa apenas estes nomes: {', '.join(names)}. "
            "Formato: [MM:SS] Nome: texto. "
            "Agrupa falas consecutivas do mesmo locutor quando fizer sentido. "
            "Não inventes timestamps. Devolve só a transcrição etiquetada."
        )
    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": text[:12000]},
            ],
            temperature=0.2,
            max_tokens=min(4000, max(400, len(text) // 2 + 200)),
        )
        out = (resp.choices[0].message.content or "").strip()
        if out and "[" in out:
            return {"text": out, "method": "llm", "names": names}
    except Exception as exc:
        logger.warning("Diarização LLM falhou: %s", exc)
    return {"text": _heuristic(), "method": "heuristic", "names": names}


class SummarizeRequest(BaseModel):
    text: str
    token: str = ""
    mode: str = "normal"
    lang: str = "pt"

@app.post("/summarize")
async def summarize(req: SummarizeRequest, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    if req.lang == "en":
        if req.mode == "minuta":
            prompt = ("Based on the transcript below, create clear bullet-point minutes including:\n"
                      "- Topics discussed\n- Decisions\n- Owners (if mentioned)\n- Action items\n\n"
                      f"Transcript:\n{req.text}")
            sys = "You summarize transcripts into meeting minutes."
        else:
            prompt = f"Summarize clearly and concisely:\n\n{req.text}"
            sys = "You summarize transcripts."
    else:
        if req.mode == "minuta":
            prompt = ("A partir da transcrição abaixo, cria uma minuta em tópicos com:\n"
                      "- Tópicos discutidos\n- Decisões\n- Responsáveis (se houver)\n- Ações\n\n"
                      f"Transcrição:\n{req.text}")
            sys = "Resumes transcrições em minutas."
        else:
            prompt = f"Resume de forma clara e concisa:\n\n{req.text}"
            sys = "És um assistente que resume transcrições de áudio."
    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600,
        )
        maybe_notify_activity(request, "Resumo com IA gerado", "Resumo criado no Ouviescrevi")
        return {"summary": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    text = data.get("text") or ""
    language = (data.get("language") or "").lower()
    token = data.get("token") or ""
    require_token(token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    idiomas = {"inglês": "English", "espanhol": "Spanish", "francês": "French", "alemão": "German", "italiano": "Italian", "português": "Portuguese"}
    if language not in idiomas:
        raise HTTPException(status_code=400, detail=f"Idioma não suportado: {language}")
    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": f"Traduz o texto para {idiomas[language]}."}, {"role": "user", "content": text}],
            temperature=0.3,
        )
        maybe_notify_activity(request, "Tradução com IA feita", "Tradução realizada no Ouviescrevi")
        return {"translation": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ClassifyRequest(BaseModel):
    text: str
    token: str

@app.post("/classify")
async def classify_content(req: ClassifyRequest, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    prompt = ("Classifica o texto como uma das opções:\n"
              "- Entrevista\n- Aula\n- Podcast\n- Reunião\n- Apresentação\n- Testemunho\n- Conversa informal\n\n"
              f"Texto:\n{req.text}\n\nResponde só com uma etiqueta.")
    try:
        resp = client.chat.completions.create(model=CLS_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=20)
        maybe_notify_activity(request, "Classificação com IA feita", "Classificação feita no Ouviescrevi")
        return {"type": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct")
async def correct_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    token = data.get("token", "")
    mode = data.get("mode", "normal")
    lang = data.get("lang", "pt")
    require_token(token)
    enforce_rate_limit(req, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Texto em falta.")
    prompt_sets = {
        "pt": {
            "normal": "Corrige ortografia e gramática em português europeu, mantendo o sentido e o tom. Devolve apenas o texto corrigido, sem explicações.",
            "spelling": "Corrige apenas ortografia, acentuação e pontuação em português europeu. Não reformules frases nem alteres o estilo. Devolve apenas o texto corrigido.",
            "formal": "Corrige o texto em português europeu e adapta-o a um tom mais formal e profissional, mantendo o sentido. Devolve apenas o texto corrigido.",
            "simple": "Corrige o texto em português europeu e simplifica a linguagem para ser mais clara, mantendo o sentido. Devolve apenas o texto corrigido.",
        },
        "en": {
            "normal": "Correct spelling and grammar in English, keeping meaning and tone. Return only the corrected text, no explanations.",
            "spelling": "Correct only spelling and punctuation in English. Do not rephrase or change style. Return only the corrected text.",
            "formal": "Correct the English text and adapt it to a more formal professional tone, keeping the meaning. Return only the corrected text.",
            "simple": "Correct the English text and simplify the language for clarity, keeping the meaning. Return only the corrected text.",
        },
        "es": {
            "normal": "Corrige ortografía y gramática en español, manteniendo el sentido y el tono. Devuelve solo el texto corregido, sin explicaciones.",
            "spelling": "Corrige solo ortografía y puntuación en español. No reformules frases. Devuelve solo el texto corregido.",
            "formal": "Corrige el texto en español y adáptalo a un tono más formal, manteniendo el sentido. Devuelve solo el texto corregido.",
            "simple": "Corrige el texto en español y simplifica el lenguaje, manteniendo el sentido. Devuelve solo el texto corregido.",
        },
        "fr": {
            "normal": "Corrige l'orthographe et la grammaire en français, en gardant le sens et le ton. Renvoie uniquement le texte corrigé, sans explications.",
            "spelling": "Corrige uniquement l'orthographe et la ponctuation en français. Ne reformule pas. Renvoie uniquement le texte corrigé.",
            "formal": "Corrige le texte en français et adapte-le à un ton plus formel, en gardant le sens. Renvoie uniquement le texte corrigé.",
            "simple": "Corrige le texte en français et simplifie le langage, en gardant le sens. Renvoie uniquement le texte corrigé.",
        },
        "de": {
            "normal": "Korrigiere Rechtschreibung und Grammatik auf Deutsch, behalte Sinn und Ton bei. Gib nur den korrigierten Text zurück, ohne Erklärungen.",
            "spelling": "Korrigiere nur Rechtschreibung und Zeichensetzung auf Deutsch. Formuliere nicht um. Gib nur den korrigierten Text zurück.",
            "formal": "Korrigiere den deutschen Text und passe ihn an einen formelleren Ton an, behalte den Sinn bei. Gib nur den korrigierten Text zurück.",
            "simple": "Korrigiere den deutschen Text und vereinfache die Sprache, behalte den Sinn bei. Gib nur den korrigierten Text zurück.",
        },
    }
    prompts = prompt_sets.get(lang, prompt_sets["pt"])
    system = prompts.get(mode, prompts["normal"])
    text_len = len(text)
    max_out = min(4096, max(96, int(text_len * 1.15) + 32))
    if mode == "spelling":
        max_out = min(max_out, max(96, int(text_len * 1.08) + 24))
    try:
        resp = client.chat.completions.create(
            model=COR_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text}],
            temperature=0.2,
            max_tokens=max_out,
        )
        corrected = resp.choices[0].message.content.strip()
        maybe_notify_activity(req, "Correção com IA feita", "Texto corrigido no Ouviescrevi")
        history_id = None
        actor = resolve_site_actor(req)
        if actor["type"] == "user":
            history_id = admin_store.save_user_correction(
                actor["email"],
                original_text=text,
                corrected_text=corrected,
                mode=mode,
            )
        return {"corrected": corrected, "history_id": history_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EmailRequest(BaseModel):
    text: str
    token: str
    tone: str = "formal"

@app.post("/generate-email")
async def generate_email(req: EmailRequest, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    prompt = (f"Gera um email em tom {req.tone} a partir do seguinte resumo/transcrição de uma reunião:\n\n{req.text}\n\n"
              "O email deve ser claro, direto e adequado para enviar após a reunião.")
    try:
        resp = client.chat.completions.create(
            model=EML_MODEL,
            messages=[{"role": "system", "content": "Escreves emails profissionais a partir de resumos/transcrições."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=700,
        )
        maybe_notify_activity(request, "Email com IA gerado", "Email gerado no Ouviescrevi")
        return {"email": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def get_status():
    return get_maintenance_payload()


@app.post("/api/status")
async def update_status(request: Request):
    require_admin_token(request)
    data = await request.json()
    return admin_store.set_maintenance(
        bool(data.get("manutencao", False)),
        data.get("maintenance_message"),
        data.get("block_transcribe_only"),
        getattr(request.state, "admin_user", "admin"),
    )

@app.get("/transcricoes-hoje")
def contar_transcricoes_hoje(request: Request):
    require_admin_token(request)
    from database import get_connection
    conn = get_connection()
    try:
        cur = conn.cursor()
        hoje = date.today().isoformat()
        cur.execute("SELECT COUNT(*) FROM transcricoes WHERE substr(data,1,10) = ?", (hoje,))
        total = cur.fetchone()[0]
        return {"total": total}
    finally:
        conn.close()

@app.get("/api/logs")
def get_logs(request: Request, q: str | None = None, status: str | None = None, limit: int = 100):
    require_admin_token(request)
    return {"logs": admin_store.list_transcriptions(q=q, status=status, limit=limit)}

class QuestionRequest(BaseModel):
    text: str
    token: str
    lang: str = "pt"
    num_questions: int = 3

_QUESTION_LANG_SPECS: dict[str, tuple[str, str]] = {
    "pt": (
        "Crias perguntas de estudo em português.",
        "Gera {n} perguntas de escolha múltipla com base no texto. "
        "Para cada pergunta: enunciado, quatro opções (A–D), resposta correta e breve explicação. "
        "Escreve tudo em português, mesmo que o texto de origem esteja noutro idioma.\n\nTexto:\n{text}",
    ),
    "en": (
        "You create study questions in English.",
        "Generate {n} multiple-choice questions based on the text. "
        "For each: question, four options (A–D), correct answer, short explanation. "
        "Write everything in English, even if the source text is in another language.\n\nText:\n{text}",
    ),
    "es": (
        "Creas preguntas de estudio en español.",
        "Genera {n} preguntas de opción múltiple basadas en el texto. "
        "Para cada una: enunciado, cuatro opciones (A–D), respuesta correcta y breve explicación. "
        "Escribe todo en español, aunque el texto de origen esté en otro idioma.\n\nTexto:\n{text}",
    ),
    "fr": (
        "Tu crées des questions d'étude en français.",
        "Génère {n} questions à choix multiples à partir du texte. "
        "Pour chacune : énoncé, quatre options (A–D), bonne réponse et brève explication. "
        "Écris tout en français, même si le texte source est dans une autre langue.\n\nTexte :\n{text}",
    ),
    "de": (
        "Du erstellst Lernfragen auf Deutsch.",
        "Erstelle {n} Multiple-Choice-Fragen basierend auf dem Text. "
        "Für jede: Frage, vier Optionen (A–D), richtige Antwort und kurze Erklärung. "
        "Schreibe alles auf Deutsch, auch wenn der Quelltext in einer anderen Sprache ist.\n\nText:\n{text}",
    ),
}

@app.post("/generate-questions")
async def generate_questions(req: QuestionRequest, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    lang = (req.lang or "pt").lower()
    if lang not in _QUESTION_LANG_SPECS:
        lang = "pt"
    sys, prompt_tpl = _QUESTION_LANG_SPECS[lang]
    prompt = prompt_tpl.format(n=req.num_questions, text=req.text)
    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=900,
        )
        maybe_notify_activity(request, "Perguntas de estudo geradas", "Perguntas geradas no Ouviescrevi")
        return {"questions": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

_AULA_PRONTA_MAX_CHARS = 14_000

class AulaProntaRequest(BaseModel):
    text: str
    token: str = ""
    lang: str = "pt"
    num_questions: int = 10

def _parse_llm_json(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

@app.post("/generate-aula-pronta")
async def generate_aula_pronta(req: AulaProntaRequest, request: Request):
    """Pacote de estudo: resumos, glossário, pontos-chave e perguntas."""
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)

    source = (req.text or "").strip()
    if len(source) < 80:
        raise HTTPException(status_code=400, detail="Texto demasiado curto para gerar o pacote (mín. ~80 caracteres).")

    lang = (req.lang or "pt").lower()
    if lang not in ("pt", "en", "es", "fr", "de"):
        lang = "pt"
    n_q = max(5, min(15, int(req.num_questions or 10)))

    truncated = False
    if len(source) > _AULA_PRONTA_MAX_CHARS:
        source = source[:_AULA_PRONTA_MAX_CHARS]
        truncated = True

    lang_names = {
        "pt": "português de Portugal",
        "en": "English",
        "es": "español",
        "fr": "français",
        "de": "Deutsch",
    }
    lang_label = lang_names[lang]

    sys = (
        "És um assistente pedagógico. Respondes APENAS com JSON válido, sem markdown, "
        "seguindo exatamente o esquema pedido."
    )
    prompt = f"""Com base no texto de uma aula/transcrição abaixo, cria um pacote de estudo completo em {lang_label}.

Devolve um único objeto JSON com estas chaves:
- "title": título curto da aula (string)
- "short_summary": resumo em 3–5 frases (string)
- "study_summary": resumo para estudar, em parágrafos ou tópicos com \\n (string)
- "key_points": lista de 5–8 ideias-chave (array de strings)
- "glossary": lista de 5–12 termos importantes, cada um com "term" e "definition" (array de objetos)
- "questions": exatamente {n_q} perguntas de escolha múltipla; cada uma com:
    "prompt" (string), "options" (objeto com chaves A,B,C,D), "answer" (letra A-D), "explanation" (string breve)

Regras:
- Tudo no idioma pedido ({lang_label}), mesmo que o texto fonte seja outro idioma.
- Perguntas adequadas a revisão escolar; opções plausíveis.
- Sem texto fora do JSON.

Texto:
{source}"""

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=3200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        pack = _parse_llm_json(raw)
        if not isinstance(pack.get("questions"), list) or not pack.get("short_summary"):
            raise ValueError("Resposta incompleta do modelo")
        maybe_notify_activity(
            request,
            "Pacote Aula Pronta gerado",
            "Material de estudo criado no Ouviescrevi",
        )
        return {
            "pack": pack,
            "num_questions": len(pack.get("questions") or []),
            "truncated": truncated,
        }
    except json.JSONDecodeError as e:
        logger.exception("Aula pronta JSON inválido")
        raise HTTPException(status_code=500, detail="Resposta inválida do modelo. Tenta novamente.") from e
    except Exception as e:
        logger.exception("Erro em /generate-aula-pronta")
        raise HTTPException(status_code=500, detail=str(e)) from e

_CHAPTERS_MAX_CHARS = 16_000
_TS_BLOCK_RE = re.compile(r"^\[(\d{2}):(\d{2})\]\s*(.*)$", re.DOTALL)

class ChaptersRequest(BaseModel):
    text: str
    token: str = ""
    lang: str = "pt"
    max_chapters: int = 12

def _parse_timestamped_blocks(text: str) -> tuple[list[dict], bool]:
    blocks: list[dict] = []
    has_ts = False
    for chunk in re.split(r"\n\s*\n", (text or "").strip()):
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.split("\n")
        first = lines[0].strip()
        m = _TS_BLOCK_RE.match(first)
        if m:
            has_ts = True
            mins, secs = int(m.group(1)), int(m.group(2))
            body = m.group(3).strip()
            if len(lines) > 1:
                body = (body + "\n" + "\n".join(lines[1:])).strip()
            blocks.append({
                "start_sec": mins * 60 + secs,
                "start": f"{mins:02d}:{secs:02d}",
                "text": body,
            })
        else:
            blocks.append({"start_sec": None, "start": None, "text": chunk})
    return blocks, has_ts

def _build_chapter_timeline(blocks: list[dict], limit: int = 90) -> str:
    if not blocks:
        return ""
    if len(blocks) <= limit:
        sampled = blocks
    else:
        step = len(blocks) / limit
        sampled = [blocks[int(i * step)] for i in range(limit)]
    lines = []
    for b in sampled:
        label = b.get("start") or "—"
        snippet = re.sub(r"\s+", " ", (b.get("text") or ""))[:220]
        lines.append(f"{label} | {snippet}")
    return "\n".join(lines)

def _youtube_timestamp(mm_ss: str) -> str:
    if not mm_ss or ":" not in mm_ss:
        return "0:00"
    parts = mm_ss.strip().split(":")
    if len(parts) != 2:
        return mm_ss
    try:
        m, s = int(parts[0]), int(parts[1])
    except ValueError:
        return mm_ss
    if m >= 60:
        h, rem = divmod(m, 60)
        return f"{h}:{rem:02d}:{s:02d}"
    return f"{m}:{s:02d}"

@app.post("/generate-chapters")
async def generate_chapters(req: ChaptersRequest, request: Request):
    """Capítulos com timestamps a partir de transcrição formatada ou texto longo."""
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)

    source = (req.text or "").strip()
    if len(source) < 120:
        raise HTTPException(status_code=400, detail="Texto demasiado curto (mín. ~120 caracteres).")

    lang = (req.lang or "pt").lower()
    if lang not in ("pt", "en", "es", "fr", "de"):
        lang = "pt"
    max_ch = max(4, min(20, int(req.max_chapters or 12)))

    truncated = False
    if len(source) > _CHAPTERS_MAX_CHARS:
        source = source[:_CHAPTERS_MAX_CHARS]
        truncated = True

    blocks, has_ts = _parse_timestamped_blocks(source)
    timeline = _build_chapter_timeline(blocks)

    lang_names = {
        "pt": "português de Portugal",
        "en": "English",
        "es": "español",
        "fr": "français",
        "de": "Deutsch",
    }
    lang_label = lang_names[lang]

    if has_ts:
        ts_rule = (
            "O texto inclui timestamps no formato [MM:SS]. Para cada capítulo, usa o campo "
            '"start" com o timestamp EXATO de um dos blocos (formato MM:SS, sem colchetes). '
            "Os capítulos devem seguir a ordem cronológica."
        )
    else:
        ts_rule = (
            'Não há timestamps no texto. Usa "start": null em todos os capítulos e ordena por relevância lógica.'
        )

    sys = (
        "És um editor de podcasts e vídeos educativos. Respondes APENAS com JSON válido, sem markdown."
    )
    prompt = f"""Analisa a transcrição/aula abaixo e divide em {max_ch} capítulos claros em {lang_label}.

Devolve um objeto JSON:
- "title": título geral sugerido (string)
- "has_timestamps": {str(has_ts).lower()}
- "chapters": lista de capítulos, cada um com:
    "title" (string curta),
    "start" (string MM:SS ou null),
    "summary" (1-2 frases)

{ts_rule}
- Títulos informativos (não genéricos como "Parte 1").
- Cobre todo o conteúdo sem grandes lacunas.

Linha do tempo (amostra):
{timeline or source[:4000]}

Texto completo:
{source}"""

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.45,
            max_tokens=2200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_llm_json(raw)
        chapters = data.get("chapters") or []
        if not isinstance(chapters, list) or not chapters:
            raise ValueError("Sem capítulos")

        normalized = []
        for i, ch in enumerate(chapters):
            if not isinstance(ch, dict):
                continue
            start = ch.get("start")
            if start is not None and start != "":
                start = str(start).strip().lstrip("[").rstrip("]")
            else:
                start = None
            normalized.append({
                "index": i + 1,
                "title": str(ch.get("title") or f"Capítulo {i + 1}").strip(),
                "start": start,
                "youtube_start": _youtube_timestamp(start) if start else None,
                "summary": str(ch.get("summary") or "").strip(),
            })

        if not normalized:
            raise ValueError("Capítulos vazios")

        maybe_notify_activity(
            request,
            "Capítulos gerados",
            "Capítulos com timestamps criados no Ouviescrevi",
        )
        return {
            "title": data.get("title") or "",
            "chapters": normalized,
            "has_timestamps": has_ts,
            "truncated": truncated,
        }
    except json.JSONDecodeError as e:
        logger.exception("Capítulos JSON inválido")
        raise HTTPException(status_code=500, detail="Resposta inválida do modelo. Tenta novamente.") from e
    except Exception as e:
        logger.exception("Erro em /generate-chapters")
        raise HTTPException(status_code=500, detail=str(e)) from e

_FLASHCARDS_MAX_CHARS = 14_000
_YOUTUBE_DESC_MAX_CHARS = 16_000


class FlashcardsRequest(BaseModel):
    text: str
    token: str = ""
    lang: str = "pt"
    num_cards: int = 15


class YoutubeDescriptionRequest(BaseModel):
    text: str
    token: str = ""
    lang: str = "pt"
    title_hint: str = ""
    chapters_text: str = ""


@app.post("/generate-flashcards")
async def generate_flashcards(req: FlashcardsRequest, request: Request):
    """Flashcards (frente/verso) a partir de texto ou transcrição."""
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)

    source = (req.text or "").strip()
    if len(source) < 80:
        raise HTTPException(status_code=400, detail="Texto demasiado curto (mín. ~80 caracteres).")

    lang = (req.lang or "pt").lower()
    if lang not in ("pt", "en", "es", "fr", "de"):
        lang = "pt"
    n_cards = max(5, min(30, int(req.num_cards or 15)))

    truncated = False
    if len(source) > _FLASHCARDS_MAX_CHARS:
        source = source[:_FLASHCARDS_MAX_CHARS]
        truncated = True

    lang_names = {
        "pt": "português de Portugal",
        "en": "English",
        "es": "español",
        "fr": "français",
        "de": "Deutsch",
    }
    lang_label = lang_names[lang]

    sys = (
        "És um assistente pedagógico. Respondes APENAS com JSON válido, sem markdown."
    )
    prompt = f"""Cria exatamente {n_cards} flashcards de estudo em {lang_label} a partir do texto abaixo.

Devolve um objeto JSON:
- "title": título curto do conjunto (string)
- "cards": lista de exatamente {n_cards} cartões, cada um com:
    "front" (pergunta ou termo, string curta),
    "back" (resposta ou definição, string clara)

Regras:
- Cartões variados: conceitos, definições, factos-chave e perguntas rápidas.
- Frente concisa; verso completo mas sem parágrafos enormes.
- Tudo em {lang_label}, mesmo que o texto fonte seja outro idioma.
- Sem texto fora do JSON.

Texto:
{source}"""

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2800,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_llm_json(raw)
        cards = data.get("cards") or []
        if not isinstance(cards, list) or not cards:
            raise ValueError("Sem cartões")
        normalized = []
        for i, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            front = str(card.get("front") or "").strip()
            back = str(card.get("back") or "").strip()
            if front and back:
                normalized.append({"index": i + 1, "front": front, "back": back})
        if not normalized:
            raise ValueError("Cartões vazios")
        maybe_notify_activity(
            request,
            "Flashcards gerados",
            "Conjunto de flashcards criado no Ouviescrevi",
        )
        return {
            "title": data.get("title") or "",
            "cards": normalized,
            "num_cards": len(normalized),
            "truncated": truncated,
        }
    except json.JSONDecodeError as e:
        logger.exception("Flashcards JSON inválido")
        raise HTTPException(status_code=500, detail="Resposta inválida do modelo. Tenta novamente.") from e
    except Exception as e:
        logger.exception("Erro em /generate-flashcards")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/generate-youtube-description")
async def generate_youtube_description(req: YoutubeDescriptionRequest, request: Request):
    """Título, descrição e tags para YouTube a partir de transcrição/resumo."""
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)

    source = (req.text or "").strip()
    if len(source) < 120:
        raise HTTPException(status_code=400, detail="Texto demasiado curto (mín. ~120 caracteres).")

    lang = (req.lang or "pt").lower()
    if lang not in ("pt", "en", "es", "fr", "de"):
        lang = "pt"

    truncated = False
    if len(source) > _YOUTUBE_DESC_MAX_CHARS:
        source = source[:_YOUTUBE_DESC_MAX_CHARS]
        truncated = True

    chapters_block = (req.chapters_text or "").strip()
    title_hint = (req.title_hint or "").strip()

    lang_names = {
        "pt": "português de Portugal",
        "en": "English",
        "es": "español",
        "fr": "français",
        "de": "Deutsch",
    }
    lang_label = lang_names[lang]

    chapters_note = ""
    if chapters_block:
        chapters_note = (
            "\n\nCapítulos já definidos (inclui estes na descrição, um por linha, formato 0:00 Título):\n"
            + chapters_block[:3000]
        )

    hint_note = f'\nSugestão de título do criador: "{title_hint}"\n' if title_hint else ""

    sys = (
        "És um especialista em SEO para YouTube e podcasts. Respondes APENAS com JSON válido, sem markdown."
    )
    prompt = f"""Cria metadados para um vídeo YouTube em {lang_label} com base no conteúdo abaixo.
{hint_note}{chapters_note}

Devolve um objeto JSON:
- "titles": lista de 3 títulos alternativos (strings, máx. ~70 caracteres, apelativos e claros)
- "description": descrição completa para YouTube (string com parágrafos usando \\n):
    * 2-3 frases de gancho no início
    * bullet points ou lista do que o espectador aprende (3-6 itens)
    * bloco "Capítulos:" com timestamps se fornecidos ou inferidos do texto (formato M:SS ou H:MM:SS + título, um por linha)
    * linha final: "Gerado com Ouviescrevi — transcrição e IA grátis"
- "tags": lista de 8-15 tags/palavras-chave (strings curtas, sem #)

Regras:
- Tudo em {lang_label}.
- Descrição pronta a colar no YouTube (sem markdown).
- Tags relevantes para descoberta, separadas conceptualmente.

Conteúdo:
{source}"""

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=2800,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = _parse_llm_json(raw)
        titles = data.get("titles") or []
        description = str(data.get("description") or "").strip()
        tags = data.get("tags") or []
        if not description:
            raise ValueError("Descrição vazia")
        if not isinstance(titles, list):
            titles = []
        if not isinstance(tags, list):
            tags = []
        titles = [str(t).strip() for t in titles if str(t).strip()][:5]
        tags = [str(t).strip() for t in tags if str(t).strip()][:20]
        maybe_notify_activity(
            request,
            "Descrição YouTube gerada",
            "Metadados YouTube criados no Ouviescrevi",
        )
        return {
            "titles": titles,
            "description": description,
            "tags": tags,
            "tags_csv": ", ".join(tags),
            "truncated": truncated,
        }
    except json.JSONDecodeError as e:
        logger.exception("YouTube description JSON inválido")
        raise HTTPException(status_code=500, detail="Resposta inválida do modelo. Tenta novamente.") from e
    except Exception as e:
        logger.exception("Erro em /generate-youtube-description")
        raise HTTPException(status_code=500, detail=str(e)) from e

def extract_url_article_text(html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "noscript", "iframe", "svg", "form"]):
        tag.decompose()

    def paragraphs_from(node):
        if not node:
            return []
        return [
            p.get_text(" ", strip=True)
            for p in node.find_all("p")
            if len(p.get_text(strip=True)) >= 15
        ]

    candidates = []
    for root in (
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
    ):
        if root:
            candidates.append(paragraphs_from(root))

    for sel in (
        ".article-body",
        ".article-content",
        ".content-body",
        ".post-content",
        ".entry-content",
        "#article-body",
        "[class*='article']",
    ):
        el = soup.select_one(sel)
        if el:
            candidates.append(paragraphs_from(el))

    best = max(candidates, key=len, default=[])
    if len(best) < 2:
        best = paragraphs_from(soup.body or soup)

    full_text = " ".join(best)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "")
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                if not isinstance(item, dict):
                    continue
                body = item.get("articleBody") or item.get("description")
                if body and isinstance(body, str) and len(body) > len(full_text):
                    full_text = body.strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if len(full_text) < 300:
        extras = []
        for meta in soup.find_all("meta"):
            name = (meta.get("name") or meta.get("property") or "").lower()
            if name in ("description", "og:description", "twitter:description"):
                content = (meta.get("content") or "").strip()
                if len(content) > 40:
                    extras.append(content)
        if extras:
            full_text = (full_text + " " + " ".join(extras)).strip()

    if len(full_text) < 150:
        root = soup.body or soup
        raw = root.get_text(" ", strip=True)
        raw = re.sub(r"\s+", " ", raw)
        if len(raw) > len(full_text):
            full_text = raw[:12000]

    return full_text.strip()

@app.post("/summarize-url")
async def summarize_url(req: Request):
    data = await req.json()
    url = data.get("url")
    token = data.get("token")
    mode = data.get("mode", "normal")
    lang = data.get("lang", "pt")
    require_token(token)
    enforce_rate_limit(req, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    if not url:
        raise HTTPException(status_code=400, detail="URL em falta.")
    validate_public_http_url(url)
    try:
        r = safe_http_get(url, timeout=12)
        full_text = extract_url_article_text(r.content)
        if not full_text or len(full_text) < 80:
            if len(r.content) < 8000:
                raise HTTPException(
                    status_code=400,
                    detail="O site pode estar a bloquear leitura automática ou o artigo não está acessível.",
                )
            raise HTTPException(status_code=400, detail="Não foi possível extrair conteúdo útil da URL.")
        chunks = textwrap.wrap(full_text, 3000)
        summaries = []
        for i, chunk in enumerate(chunks, start=1):
            if lang == "en":
                if mode == "minuta":
                    prompt = "Generate bullet point minutes from this section:\n\n" + chunk
                    sys = "You summarize online articles."
                else:
                    prompt = "Summarize clearly and concisely:\n\n" + chunk
                    sys = "You summarize online articles."
            else:
                if mode == "minuta":
                    prompt = "Gera uma minuta em tópicos a partir desta secção:\n\n" + chunk
                    sys = "És um assistente que resume artigos online."
                else:
                    prompt = "Resume de forma clara e concisa:\n\n" + chunk
                    sys = "És um assistente que resume artigos online."
            resp = client.chat.completions.create(
                model=SUM_MODEL,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=800,
            )
            summaries.append(f"🧩 Parte {i}:\n{resp.choices[0].message.content.strip()}")
        final_summary = "\n\n".join(summaries)
        maybe_notify_activity(req, f"Resumo gerado por URL:\n{url}", "Resumo por URL no Ouviescrevi")
        return {"summary": final_summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar URL: {e}")

# ── Notificações: WhatsApp share ──────────────────────────────────────────────
class WhatsAppNotify(BaseModel):
    page: str | None = None
    note: str | None = None
    token: str

@app.post("/notify-whatsapp-share")
@app.post("/notify/whatsapp")
async def notify_whatsapp(req: WhatsAppNotify, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    try:
        msg = f"Partilha no WhatsApp usada.\nPágina: {req.page or '-'}\nNota: {req.note or '-'}\nData: {datetime.now().isoformat()}"
        maybe_notify_activity(request, msg, "WhatsApp share usado no Ouviescrevi")
        logger.info("WhatsApp share notificado: page=%s note=%s", req.page, req.note)
        return {"ok": True}
    except Exception as e:
        logger.exception("Erro ao notificar WhatsApp share")
        raise HTTPException(status_code=500, detail=str(e))

# ── Vídeo (router) existente ─────────────────────────────────────────────────
router = APIRouter()

class VideoRequest(BaseModel):
    text: str
    image_url: str = "https://placehold.co/720x1280?text=Ouviescrevi"
    voice_lang: str = "pt"
    token: str = ""

@router.post("/generate-video")
async def generate_video(req: VideoRequest, request: Request):
    require_api_token(request, req.token or None)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    try:
        audio_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        gTTS(text=req.text, lang=req.voice_lang).save(audio_tmp)
        img_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
        validate_public_http_url(req.image_url)
        rr = safe_http_get(req.image_url, timeout=10, max_bytes=5_000_000)
        ct = rr.headers.get("Content-Type", "")
        if rr.status_code == 200 and "image" in ct:
            with open(img_tmp, "wb") as f:
                f.write(rr.content)
        else:
            raise HTTPException(status_code=400, detail="Erro ao obter imagem.")
        out_name = f"{uuid.uuid4()}.mp4"
        out_path = os.path.join(VIDEO_DIR, out_name)
        cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-loop", "1", "-i", img_tmp, "-i", audio_tmp,
               "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
               "-shortest", "-y", out_path]
        safe_run_ffmpeg(cmd, desc="gerar-video", timeout=max(60, FFMPEG_TIMEOUT))
        for p in (audio_tmp, img_tmp):
            try: os.remove(p)
            except: pass
        return {"success": True, "video_url": f"/static/videos/{out_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erro ao gerar vídeo")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

# ── util ──────────────────────────────────────────────────────────────────────
def _route_entries() -> list[dict]:
    entries = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path:
            continue
        methods = getattr(route, "methods", None) or set()
        entries.append({
            "path": path,
            "name": getattr(route, "name", None),
            "method": sorted(methods)[0] if methods else None,
        })
    return entries


@app.get("/")
def root():
    if ENABLE_DEBUG_ENDPOINTS:
        return {"routes": _route_entries()}
    return {"status": "ok", "service": "ouviescrevi-api"}


@app.get("/rotas")
def rotas():
    require_debug_enabled()
    return [entry["path"] for entry in _route_entries()]

@app.get("/test-email")
def test_email(request: Request):
    require_debug_enabled()
    require_admin_token(request)
    from email_notify import send_notification_email

    ok, err = send_notification_email(
        "Teste de envio do Ouviescrevi.\n\nSe recebeste isto, as notificações estão a funcionar.",
        "Teste de notificação Ouviescrevi",
    )
    if not ok:
        raise HTTPException(
            status_code=502,
            detail=err or "Falha ao enviar. No Render usa RESEND_API_KEY (SMTP costuma estar bloqueado).",
        )
    return {"status": "ok", "sent": True}
