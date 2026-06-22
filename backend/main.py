# main.py
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter, Form
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

# Timeouts e parâmetros
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "110"))  # por chunk
WHISPER_LANGUAGE = (os.getenv("WHISPER_LANGUAGE", "pt") or "").strip().lower() or None
WHISPER_TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0"))
WHISPER_PROMPT = (os.getenv(
    "WHISPER_PROMPT",
    "Transcrição em português de Portugal de uma reunião de trabalho ou conversa.",
) or "").strip() or None
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "60"))
TOTAL_TRANSCRIBE_TIMEOUT = int(os.getenv("TOTAL_TRANSCRIBE_TIMEOUT", "900"))  # watchdog global
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "600"))  # 10 min
SUBS_TIMEOUT = int(os.getenv("SUBS_TIMEOUT", "900"))  # tempo p/ queimar legendas

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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Token"],
)

# estáticos
STATIC_DIR = os.path.abspath("static")
VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Acesso administrativo negado.")


def require_token(token: str):
    """Compatibilidade com rotas que recebem token no corpo JSON."""
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")


def enforce_rate_limit(request: Request, bucket: str, limit: int, window: int) -> None:
    RATE_LIMITER.check(client_ip(request), bucket, limit, window)


def require_debug_enabled() -> None:
    if not ENABLE_DEBUG_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Não encontrado.")

def enviar_email_assunto(mensagem: str, assunto: str = "Nova atividade no Ouviescrevi"):
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg.set_content(mensagem)
        msg["Subject"] = assunto
        msg["From"] = os.getenv("SMTP_FROM", "notificacoes@ouviescrevi.pt")
        msg["To"] = os.getenv("SMTP_TO", "ouviescrevi@gmail.com")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))
        if not (smtp_user and smtp_password):
            logger.warning("SMTP_USER/SMTP_PASSWORD não configurados; a notificação não será enviada.")
            return
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
    except Exception as e:
        logger.error("Erro ao enviar email: %s", e)

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
    filtered = []
    dropped = 0
    for s in segments or []:
        text = (_seg_get(s, "text", "") or "").strip()
        if not text:
            continue
        no_speech = float(_seg_get(s, "no_speech_prob", 0) or 0)
        avg_logprob = float(_seg_get(s, "avg_logprob", 0) or 0)
        compression = float(_seg_get(s, "compression_ratio", 1) or 1)
        if no_speech > 0.5:
            dropped += 1
            continue
        if avg_logprob < -1.0:
            dropped += 1
            continue
        if compression > 2.2:
            dropped += 1
            continue
        if language == "pt" and _cjk_ratio(text) > 0.2:
            dropped += 1
            continue
        filtered.append(s)
    if dropped:
        logger.info("Whisper: descartados %d segmentos (ruído/alucinação)", dropped)
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

def registar_transcricao(nome_ficheiro: str):
    conn = sqlite3.connect("ouviescrevi.db")
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO transcricoes (ficheiro, data) VALUES (?, ?)", (nome_ficheiro, datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()

def transcrever_parte_c_com_retries(
    file_path: str,
    retries: int = 3,
    sleep_base: float = 1.0,
    timeout: int = WHISPER_TIMEOUT,
    language: str | None = None,
):
    last_err = None
    lang = resolve_whisper_language(language)
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
            if WHISPER_PROMPT:
                kwargs["prompt"] = WHISPER_PROMPT
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
@app.get("/api/frontend-config")
def frontend_config(request: Request):
    if not origin_is_allowed(request, ALLOWED_ORIGINS):
        raise HTTPException(status_code=403, detail="Origem não autorizada.")
    return {"apiBase": PUBLIC_API_BASE, "token": API_TOKEN, "maxFileSizeMb": MAX_FILE_SIZE_MB}


class AdminLoginRequest(BaseModel):
    password: str


@app.post("/api/admin/login")
def admin_login(req: AdminLoginRequest):
    if req.password != BACKOFFICE_PASSWORD:
        raise HTTPException(status_code=403, detail="Credenciais inválidas.")
    return {"ok": True, "adminToken": ADMIN_TOKEN}


@app.get("/debug")
def debug():
    require_debug_enabled()
    return {"status": "OK", "versao": "1.6"}


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    token: str | None = Form(None),
    language: str | None = Form(None),
):
    require_api_token(request, token)
    enforce_rate_limit(request, "transcribe", RATE_LIMIT_TRANSCRIBE, RATE_LIMIT_TRANSCRIBE_WINDOW)
    rid = str(uuid.uuid4())
    t_start = time.monotonic()
    whisper_lang = resolve_whisper_language(language)
    logger.info("[%s] Upload recebido (transcribe): nome=%s ct=%s cl=%s", rid, file.filename, file.content_type, request.headers.get("content-length"))
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
        return {"transcription": "", "formatted": "", "warning": f"Falha ao gravar ficheiro: {e}"}

    if size_mb > MAX_FILE_SIZE_MB:
        try: os.remove(tmp_path)
        except: pass
        logger.warning("[%s] Ficheiro > %dMB (%0.2f MB).", rid, MAX_FILE_SIZE_MB, size_mb)
        return {"transcription": "", "formatted": "", "warning": f"Ficheiro demasiado grande ({size_mb:.0f} MB). O limite é {MAX_FILE_SIZE_MB} MB."}

    audio_wav_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.wav")
    converted_ok = False
    try:
        conv = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", tmp_path, "-vn", "-sn",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_wav_path]
        safe_run_ffmpeg(conv, desc="conversao-wav", timeout=FFMPEG_TIMEOUT)
        converted_ok = True
    except Exception:
        converted_ok = False
        logger.warning("[%s] Conversão WAV falhou; seguir com original.", rid)

    split_dir = tempfile.mkdtemp(prefix="split_")
    parts = []
    used_source = None
    watchdog_hit = False

    try:
        try:
            source_for_split = audio_wav_path if converted_ok else tmp_path
            parts = split_audio(source_for_split, split_dir)
            used_source = source_for_split
            logger.info("[%s] Segmentos criados: %d", rid, len(parts))
        except Exception as e:
            logger.warning("[%s] Falha ao partir áudio (%s). Vai sem split. Erro: %s", rid, file.filename, str(e)[:300])
            parts = []

        if not parts:
            used_source = audio_wav_path if converted_ok else tmp_path
            parts = [used_source]

        full_text_chunks, formatted_chunks = [], []
        offset_seconds = 0
        failed_segments = 0
        processed_segments = 0
        quota_exceeded = False

        for idx, part in enumerate(parts):
            if (time.monotonic() - t_start) > TOTAL_TRANSCRIBE_TIMEOUT:
                watchdog_hit = True
                logger.error("[%s] Watchdog TOTAL_TRANSCRIBE_TIMEOUT atingido aos %.2fs. Interrompendo.",
                             rid, time.monotonic() - t_start)
                break
            try:
                result = transcrever_parte_c_com_retries(
                    part, retries=3, sleep_base=1.0, timeout=WHISPER_TIMEOUT, language=whisper_lang
                )
                text_piece, formatted_piece, _ = process_whisper_result(result, whisper_lang, offset_seconds)
                full_text_chunks.append(text_piece)
                formatted_chunks.append(formatted_piece)
                logger.info("[%s] Chunk %d/%d transcrito. len(text)=%d", rid, idx + 1, len(parts), len(text_piece))
            except Exception as e:
                failed_segments += 1
                logger.exception("[%s] Erro ao transcrever parte %d (%s)", rid, idx, os.path.basename(part))
                formatted_chunks.append(f"{_format_time(offset_seconds)} [Falha no segmento]")
                err_msg = str(e).lower()
                if "insufficient_quota" in err_msg or "exceeded your current quota" in err_msg:
                    quota_exceeded = True
            finally:
                processed_segments += 1
                if len(parts) > 1:
                    offset_seconds += SEGMENT_DURATION

        try:
            registar_transcricao(file.filename or "sem_nome")
        except Exception as e:
            logger.warning("[%s] Falha ao registar na DB: %s", rid, e)

        try:
            enviar_email_assunto(f"Nova transcrição recebida: {file.filename}", "Nova transcrição no Ouviescrevi")
        except Exception as e:
            logger.warning("[%s] Falha ao enviar email de notificação: %s", rid, e)

        transcription_out = clean_transcription_text("\n".join(t for t in full_text_chunks if t).strip(), whisper_lang)
        formatted_out = clean_transcription_text("\n\n".join(t for t in formatted_chunks if t).strip(), whisper_lang)

        payload = {"transcription": transcription_out, "formatted": formatted_out}
        if quota_exceeded:
            payload["warning"] = "Conta OpenAI sem créditos (insufficient_quota). Adiciona billing em platform.openai.com."
        elif failed_segments > 0:
            payload["warning"] = f"{failed_segments} de {processed_segments} segmentos falharam (aplicado retry/fallback)."
        if watchdog_hit:
            payload["warning"] = (payload.get("warning", "") + " " if payload.get("warning") else "") + "Tempo total excedido (parcial devolvido)."

        dur_total = time.monotonic() - t_start
        logger.info("[%s] FIM transcribe em %.2fs | processed=%d failed=%d watchdog=%s", rid, dur_total, processed_segments, failed_segments, watchdog_hit)
        return payload

    except Exception as e:
        logger.exception("[%s] Erro inesperado no processamento", rid)
        return {"transcription": "", "formatted": "", "warning": f"Erro ao processar: {e}"}
    finally:
        for p in (audio_wav_path, tmp_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
        try:
            for f in os.listdir(split_dir):
                try:
                    os.remove(os.path.join(split_dir, f))
                except:
                    pass
            os.rmdir(split_dir)
        except:
            pass

# ── NOVO: Vídeo com legendas embutidas ───────────────────────────────────────
@app.post("/video-subs")
async def video_subs(
    request: Request,
    file: UploadFile = File(...),
    style: str | None = Form(None),
    token: str | None = Form(None),
    language: str | None = Form(None),
):
    require_api_token(request, token)
    enforce_rate_limit(request, "video-subs", RATE_LIMIT_VIDEO_SUBS, RATE_LIMIT_VIDEO_SUBS_WINDOW)
    whisper_lang = resolve_whisper_language(language)
    """
    Upload de vídeo → transcreve (Whisper) → gera SRT → queima legendas no vídeo.
    Resposta: { video_url, srt_url, warning?, rid?, processing_ms? }
    """
    rid = str(uuid.uuid4())
    t_start = time.monotonic()

    # Log headers úteis assim que entra no handler
    ua = request.headers.get("user-agent", "-")
    cl = request.headers.get("content-length", "-")
    ct = request.headers.get("content-type", "-")
    client_ip = getattr(request.client, "host", "-") if request.client else "-"
    logger.info("[%s] [video-subs] REQUEST from=%s len=%s ct=%s ua=%s", rid, client_ip, cl, ct, ua)

    logger.info("[%s] [video-subs] Upload: %s (%s) style=%s", rid, file.filename, file.content_type, (style[:120] + "…") if style and len(style) > 120 else style)
    _reject_oversized_upload(request)

    # Gravar upload (com contagem de bytes e logs parciais)
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
        try: os.remove(tmp_video)
        except: pass
        raise HTTPException(status_code=413, detail=f"Ficheiro > {MAX_FILE_SIZE_MB}MB. Reduz o tamanho e tenta novamente.")

    # Extrair/normalizar áudio para transcrição
    audio_wav_path = os.path.join(tempfile.gettempdir(), f"subs_{uuid.uuid4()}.wav")
    try:
        conv = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", tmp_video, "-vn", "-sn",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_wav_path]
        safe_run_ffmpeg(conv, desc="audio p/ subs (wav)", timeout=max(60, FFMPEG_TIMEOUT))
    except Exception:
        logger.exception("[%s] [video-subs] Falha a extrair áudio", rid)
        raise HTTPException(status_code=500, detail="Falha ao extrair áudio com FFmpeg.")

    # Particionar áudio e transcrever (como no /transcribe)
    split_dir = tempfile.mkdtemp(prefix="subs_split_")
    try:
        parts = split_audio(audio_wav_path, split_dir)
        if not parts:
            parts = [audio_wav_path]
        entries: list[tuple[float, float, str]] = []  # [(start,end,text)]
        offset_seconds = 0
        processed_segments = 0
        failed_segments = 0
        watchdog_hit = False

        for idx, part in enumerate(parts):
            if (time.monotonic() - t_start) > TOTAL_TRANSCRIBE_TIMEOUT:
                watchdog_hit = True
                logger.error("[%s] [video-subs] Watchdog TOTAL timeout", rid)
                break
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
                processed_segments += 1
                logger.info("[%s] [video-subs] chunk %d/%d OK c/ %d segmentos", rid, idx+1, len(parts), len(segs))
            except Exception:
                logger.exception("[%s] [video-subs] Erro a transcrever chunk %d", rid, idx+1)
                failed_segments += 1
            finally:
                if len(parts) > 1:
                    offset_seconds += SEGMENT_DURATION

        # Gerar SRT
        base = str(uuid.uuid4())
        srt_tmp = os.path.join(tempfile.gettempdir(), f"{base}.srt")
        _write_srt(entries, srt_tmp)

        # Copiar SRT para estáticos (download)
        srt_out = os.path.join(VIDEO_DIR, f"{base}.srt")
        try:
            shutil.copyfile(srt_tmp, srt_out)
        except Exception:
            logger.exception("[%s] [video-subs] Falha a copiar SRT p/ static", rid)
            raise HTTPException(status_code=500, detail="Falha ao preparar SRT para download.")

        # Queimar legendas no vídeo
        out_video = os.path.join(VIDEO_DIR, f"{base}.mp4")
        vf = f"subtitles={_escape_subtitles_path(srt_tmp)}:force_style='FontName=DejaVu Sans,FontSize=24,Outline=1,BorderStyle=1,Shadow=0,MarginV=24'"
        burn = [
            FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", tmp_video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "copy",
            out_video
        ]
        warning = None
        try:
            safe_run_ffmpeg(burn, desc="queimar-legendas", timeout=SUBS_TIMEOUT)
            logger.info("[%s] [video-subs] Vídeo legendado gerado", rid)
        except Exception:
            # Fallback: entregar só o SRT
            warning = "Não foi possível embutir as legendas (FFmpeg/libass). A entregar apenas o .srt."
            logger.warning("[%s] [video-subs] Falha a queimar legendas. Fallback SRT.", rid)
            out_video = None

        # Registar e notificar
        try:
            registar_transcricao((file.filename or "sem_nome") + " [legendado]")
        except Exception as e:
            logger.warning("[%s] [video-subs] Falha ao registar DB: %s", rid, e)
        try:
            enviar_email_assunto(f"Vídeo legendado gerado: {file.filename}", "Vídeo legendado no Ouviescrevi")
        except Exception:
            pass

        processing_ms = round((time.monotonic() - t_start) * 1000)
        resp = {
            "srt_url": f"/static/videos/{os.path.basename(srt_out)}",
            "rid": rid,
            "processing_ms": processing_ms
        }
        if out_video:
            resp["video_url"] = f"/static/videos/{os.path.basename(out_video)}"
        if warning:
            resp["warning"] = warning
        if failed_segments:
            resp["note"] = f"Alguns segmentos falharam ({failed_segments})."

        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[%s] [video-subs] Erro inesperado", rid)
        raise HTTPException(status_code=500, detail=f"Erro ao processar vídeo: {e}")
    finally:
        # limpeza
        for p in (audio_wav_path, tmp_video, ):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except:
                pass
        try:
            for f in os.listdir(split_dir):
                try:
                    os.remove(os.path.join(split_dir, f))
                except:
                    pass
            os.rmdir(split_dir)
        except:
            pass
        try:
            if 'srt_tmp' in locals() and os.path.exists(srt_tmp):
                os.remove(srt_tmp)
        except:
            pass

# ── IA payloads ───────────────────────────────────────────────────────────────
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
        try:
            enviar_email_assunto("Resumo com IA gerado", "Resumo criado no Ouviescrevi")
        except Exception:
            pass
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
        try:
            enviar_email_assunto("Tradução com IA feita", "Tradução realizada no Ouviescrevi")
        except Exception:
            pass
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
        try:
            enviar_email_assunto("Classificação com IA feita", "Classificação feita no Ouviescrevi")
        except Exception:
            pass
        return {"type": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/correct")
async def correct_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    token = data.get("token", "")
    require_token(token)
    enforce_rate_limit(req, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    try:
        resp = client.chat.completions.create(
            model=COR_MODEL,
            messages=[{"role": "system", "content": "Corrige ortografia e gramática mantendo o sentido e o tom."},
                      {"role": "user", "content": text}],
            temperature=0.2,
        )
        try:
            enviar_email_assunto("Correção com IA feita", "Texto corrigido no Ouviescrevi")
        except Exception:
            pass
        return {"corrected": resp.choices[0].message.content.strip()}
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
        try:
            enviar_email_assunto("Email com IA gerado", "Email gerado no Ouviescrevi")
        except Exception:
            pass
        return {"email": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def get_status():
    conn = sqlite3.connect("ouviescrevi.db")
    try:
        cur = conn.cursor()
        cur.execute("SELECT manutencao FROM status WHERE id = 1")
        row = cur.fetchone()
        return {"manutencao": bool(row[0]) if row else False}
    finally:
        conn.close()

@app.post("/api/status")
async def update_status(request: Request):
    require_admin_token(request)
    data = await request.json()
    manutencao = bool(data.get("manutencao", False))
    conn = sqlite3.connect("ouviescrevi.db")
    try:
        cur = conn.cursor()
        cur.execute("UPDATE status SET manutencao = ? WHERE id = 1", (manutencao,))
        if cur.rowcount == 0:
            cur.execute("INSERT INTO status (id, manutencao) VALUES (1, ?)", (manutencao,))
        conn.commit()
        return {"message": "Estado atualizado com sucesso", "manutencao": manutencao}
    finally:
        conn.close()

@app.get("/transcricoes-hoje")
def contar_transcricoes_hoje(request: Request):
    require_admin_token(request)
    conn = sqlite3.connect("ouviescrevi.db")
    try:
        cur = conn.cursor()
        hoje = date.today().isoformat()
        cur.execute("SELECT COUNT(*) FROM transcricoes WHERE substr(data,1,10) = ?", (hoje,))
        total = cur.fetchone()[0]
        return {"total": total}
    finally:
        conn.close()

@app.get("/api/logs")
def get_logs(request: Request):
    require_admin_token(request)
    try:
        conn = sqlite3.connect("ouviescrevi.db")
        cur = conn.cursor()
        cur.execute("SELECT ficheiro, data FROM transcricoes ORDER BY data DESC")
        rows = cur.fetchall()
        return [{"ficheiro": r[0], "data": r[1]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler logs: {e}")
    finally:
        try:
            conn.close()
        except:
            pass

class QuestionRequest(BaseModel):
    text: str
    token: str
    lang: str = "pt"
    num_questions: int = 3

@app.post("/generate-questions")
async def generate_questions(req: QuestionRequest, request: Request):
    require_token(req.token)
    enforce_rate_limit(request, "ai", RATE_LIMIT_AI, RATE_LIMIT_AI_WINDOW)
    if req.lang == "en":
        prompt = (f"Generate {req.num_questions} multiple-choice questions based on the text. "
                  "For each: question, four options (A–D), correct answer, short explanation.\n\n"
                  f"Text:\n{req.text}")
        sys = "You create study questions."
    else:
        prompt = (f"Gera {req.num_questions} perguntas de escolha múltipla com base no texto. "
                  "Para cada pergunta: enunciado, quatro opções (A–D), resposta correta e breve explicação.\n\n"
                  f"Texto:\n{req.text}")
        sys = "Cria perguntas de estudo."
    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=900,
        )
        try:
            enviar_email_assunto("Perguntas de estudo geradas", "Perguntas geradas no Ouviescrevi")
        except Exception:
            pass
        return {"questions": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        headers = {"User-Agent": "OuviescreviBot/1.0 (+https://ouviescrevi.pt)"}
        r = safe_http_get(url, timeout=12)
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = soup.find_all("p")
        full_text = " ".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text()) > 40)
        if not full_text:
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
        try:
            enviar_email_assunto(f"Resumo gerado por URL:\n{url}", "Resumo por URL no Ouviescrevi")
        except Exception:
            pass
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
        enviar_email_assunto(msg, "WhatsApp share usado no Ouviescrevi")
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
    token: str

@router.post("/generate-video")
async def generate_video(req: VideoRequest, request: Request):
    require_token(req.token)
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
    enviar_email_assunto("Teste de envio", "Teste SMTP Ouviescrevi")
    return {"status": "ok"}
