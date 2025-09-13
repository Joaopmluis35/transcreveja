# main.py
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, APIRouter
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

import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from openai import OpenAI

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
        # extras comuns se existirem
        for k in ("rid", "path", "method", "status", "ms"):
            if hasattr(record, k):
                payload[k] = getattr(record, k)
        return json.dumps(payload, ensure_ascii=False)

class VercelHTTPHandler(logging.Handler):
    """Envia logs para uma Function no Vercel (aparecem no dashboard)."""
    def __init__(self, url: str, token: str | None = None, level=logging.WARNING):
        super().__init__(level)
        self.url = url
        self.token = token

    def emit(self, record: logging.LogRecord):
        try:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            # Usa JSON “compacto” (a Function no Vercel só faz console.log)
            data = {
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
            requests.post(self.url, json=data, headers=headers, timeout=2)
        except Exception:
            # Nunca deixar logging quebrar a app
            pass

logger = logging.getLogger("ouviescrevi")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

if not logger.handlers:
    # ficheiro rotativo em UTF-8
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
    logger.addHandler(fh)

    # consola (Render/uvicorn)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
    logger.addHandler(ch)

    # opcional: envio p/ Vercel
    VERCEL_LOG_URL = os.getenv("VERCEL_LOG_URL")
    VERCEL_LOG_TOKEN = os.getenv("VERCEL_LOG_TOKEN")
    if VERCEL_LOG_URL:
        vh = VercelHTTPHandler(VERCEL_LOG_URL, VERCEL_LOG_TOKEN, level=logging.WARNING)
        # usa formatter simples ou JSON (ambos ok; no Vercel verás o texto do message)
        vh.setFormatter(JSONFormatter())
        logger.addHandler(vh)

# Capta também logs do uvicorn/fastapi no mesmo ficheiro/console
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

# Timeouts e parâmetros
WHISPER_TIMEOUT = int(os.getenv("WHISPER_TIMEOUT", "110"))  # por chunk
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "60"))
TOTAL_TRANSCRIBE_TIMEOUT = int(os.getenv("TOTAL_TRANSCRIBE_TIMEOUT", "170"))  # watchdog global
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "300"))
SEGMENT_DURATION = int(os.getenv("SEGMENT_DURATION", "600"))  # 10 min

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
    allow_origins=["*"],  # em prod restringe aos teus domínios
    allow_methods=["*"],
    allow_headers=["*"],
)

# garantir diretório estático
STATIC_DIR = os.path.abspath("static")
VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# Middleware: log de cada request com latência
# ──────────────────────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.monotonic()
    extra = {"rid": rid, "path": request.url.path, "method": request.method}
    logging.getLogger("ouviescrevi").info(f"→ {request.method} {request.url.path}", extra=extra)
    try:
        response = await call_next(request)
        ms = round((time.monotonic() - start) * 1000, 1)
        extra |= {"status": response.status_code, "ms": ms}
        logging.getLogger("ouviescrevi").info(f"← {request.method} {request.url.path} {response.status_code} {ms}ms", extra=extra)
        return response
    except Exception:
        ms = round((time.monotonic() - start) * 1000, 1)
        extra |= {"status": 500, "ms": ms}
        logging.getLogger("ouviescrevi").exception(f"✖ {request.method} {request.url.path} 500 {ms}ms", extra=extra)
        raise

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def require_token(token: str):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido.")

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
        seconds = int(seconds)
    except Exception:
        seconds = 0
    m, s = divmod(seconds, 60)
    return f"[{m:02d}:{s:02d}]"

def format_segments_with_offset(segments, offset_seconds: int = 0):
    formatted = []
    for s in segments or []:
        start = _seg_get(s, "start", 0)
        text = (_seg_get(s, "text", "") or "").strip()
        if text:
            formatted.append(f"{_format_time(start + offset_seconds)} {text}")
    return "\n\n".join(formatted).strip()

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

def transcrever_parte_c_com_retries(file_path: str, retries: int = 3, sleep_base: float = 1.0, timeout: int = WHISPER_TIMEOUT):
    last_err = None
    for attempt in range(1, retries + 1):
        t0 = time.monotonic()
        try:
            with open(file_path, "rb") as audio:
                result = client.with_options(timeout=timeout).audio.transcriptions.create(
                    model="whisper-1", file=audio, response_format="verbose_json"
                )
            dur = time.monotonic() - t0
            logger.info("Whisper OK (%s) tentativa %d em %.2fs", os.path.basename(file_path), attempt, dur)
            return result
        except Exception as e:
            dur = time.monotonic() - t0
            last_err = e
            logger.warning("Whisper FALHA (%s) tentativa %d/%d em %.2fs: %s",
                           os.path.basename(file_path), attempt, retries, dur, str(e)[:300])
            time.sleep(sleep_base * (2 ** (attempt - 1)))
    raise last_err

# ──────────────────────────────────────────────────────────────────────────────
# Rotas
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/debug")
def debug():
    return {"status": "OK", "versao": "1.3"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    rid = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info("[%s] Upload recebido: %s (%s)", rid, file.filename, file.content_type)

    # grava o upload
    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4()}{orig_ext}")
    try:
        with open(tmp_path, "wb") as out:
            await file.seek(0)
            shutil.copyfileobj(file.file, out)
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        logger.info("[%s] Upload guardado em disco: %.2f MB", rid, size_mb)
    except Exception as e:
        logger.exception("[%s] Falha ao gravar upload", rid)
        return {"transcription": "", "formatted": "", "warning": f"Falha ao gravar ficheiro: {e}"}

    if size_mb > MAX_FILE_SIZE_MB:
        try:
            os.remove(tmp_path)
        except:
            pass
        logger.warning("[%s] Ficheiro > %dMB (%.2f MB).", rid, MAX_FILE_SIZE_MB, size_mb)
        return {"transcription": "", "formatted": "", "warning": f"Ficheiro > {MAX_FILE_SIZE_MB}MB. Reduz o tamanho e tenta novamente."}

    # converter para wav 16k mono
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

        for idx, part in enumerate(parts):
            if (time.monotonic() - t_start) > TOTAL_TRANSCRIBE_TIMEOUT:
                watchdog_hit = True
                logger.error("[%s] Watchdog TOTAL_TRANSCRIBE_TIMEOUT atingido aos %.2fs. Interrompendo.",
                             rid, time.monotonic() - t_start)
                break

            try:
                result = transcrever_parte_c_com_retries(part, retries=3, sleep_base=1.0, timeout=WHISPER_TIMEOUT)
                text_piece = getattr(result, "text", "") or ""
                segs = getattr(result, "segments", []) or []
                full_text_chunks.append(text_piece)
                formatted_chunks.append(format_segments_with_offset(segs, offset_seconds))
                logger.info("[%s] Chunk %d/%d transcrito. len(text)=%d", rid, idx + 1, len(parts), len(text_piece))
            except Exception:
                failed_segments += 1
                logger.exception("[%s] Erro ao transcrever parte %d (%s)", rid, idx, os.path.basename(part))
                formatted_chunks.append(f"{_format_time(offset_seconds)} [Falha no segmento]")
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

        transcription_out = "\n".join(t for t in full_text_chunks if t).strip()
        formatted_out = "\n\n".join(t for t in formatted_chunks if t).strip()

        payload = {"transcription": transcription_out, "formatted": formatted_out}
        if failed_segments > 0:
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

# ── IA payloads ───────────────────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    text: str
    token: str = ""
    mode: str = "normal"
    lang: str = "pt"

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    require_token(req.token)
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
async def classify_content(req: ClassifyRequest):
    require_token(req.token)
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
async def generate_email(req: EmailRequest):
    require_token(req.token)
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
def contar_transcricoes_hoje():
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
def get_logs():
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
async def generate_questions(req: QuestionRequest):
    require_token(req.token)
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
    try:
        headers = {"User-Agent": "OuviescreviBot/1.0 (+https://ouviescrevi.pt)"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
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

@app.post("/notify/whatsapp")
async def notify_whatsapp(req: WhatsAppNotify):
    require_token(req.token)
    try:
        msg = f"Partilha no WhatsApp usada.\nPágina: {req.page or '-'}\nNota: {req.note or '-'}\nData: {datetime.now().isoformat()}"
        enviar_email_assunto(msg, "WhatsApp share usado no Ouviescrevi")
        logger.info("WhatsApp share notificado: page=%s note=%s", req.page, req.note)
        return {"ok": True}
    except Exception as e:
        logger.exception("Erro ao notificar WhatsApp share")
        raise HTTPException(status_code=500, detail=str(e))

# ── Vídeo (router) ────────────────────────────────────────────────────────────
router = APIRouter()

class VideoRequest(BaseModel):
    text: str
    image_url: str = "https://placehold.co/720x1280?text=Ouviescrevi"
    voice_lang: str = "pt"
    token: str

@router.post("/generate-video")
async def generate_video(req: VideoRequest):
    require_token(req.token)
    try:
        audio_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        gTTS(text=req.text, lang=req.voice_lang).save(audio_tmp)
        img_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
        rr = requests.get(req.image_url, timeout=10)
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
@app.get("/")
def root():
    routes = []
    for route in app.routes:
        info = {"path": route.path, "name": route.name}
        info["method"] = list(getattr(route, "methods", []) or [None])[0]
        routes.append(info)
    return {"routes": routes}

@app.get("/rotas")
def rotas():
    return [route.path for route in app.routes]

@app.get("/test-email")
def test_email():
    enviar_email_assunto("Teste de envio", "Teste SMTP Ouviescrevi")
    return {"status": "ok"}
