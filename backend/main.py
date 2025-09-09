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
import textwrap
import shutil

import requests
from bs4 import BeautifulSoup
from gtts import gTTS

from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    filename="ouviescrevi.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# DB bootstrap
from database import criar_base
criar_base()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY no .env")

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    # obrigamos a definir; evita fallback público
    raise RuntimeError("Falta ADMIN_TOKEN no .env")

client = OpenAI(api_key=OPENAI_API_KEY)

MAX_FILE_SIZE_MB = 25
SEGMENT_DURATION = 600  # 10 min
SUM_MODEL = os.getenv("SUM_MODEL", "gpt-4o-mini")
CLS_MODEL = os.getenv("CLS_MODEL", "gpt-4o-mini")
COR_MODEL = os.getenv("COR_MODEL", "gpt-4o-mini")
EML_MODEL = os.getenv("EML_MODEL", "gpt-4o-mini")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta para os teus domínios em produção
    allow_methods=["*"],
    allow_headers=["*"],
)

# garantir diretório estático
STATIC_DIR = os.path.abspath("static")
VIDEO_DIR = os.path.join(STATIC_DIR, "videos")
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
            logging.warning("SMTP_USER/SMTP_PASSWORD não configurados; a notificação não será enviada.")
            return

        with smtplib.SMTP_SSL(smtp_host, smtp_port) as smtp:
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
    except Exception as e:
        logging.error("Erro ao enviar email: %s", e)

def _seg_get(seg, key, default=None):
    # suporta objeto (atributo) e dict
    try:
        return getattr(seg, key)
    except Exception:
        try:
            return seg.get(key, default)
        except Exception:
            return default

def format_segments(segments):
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"[{m:02d}:{s:02d}]"

    formatted = []
    for s in segments or []:
        start = _seg_get(s, "start", 0)
        text = _seg_get(s, "text", "").strip()
        formatted.append(f"{format_time(start)} {text}")
    return "\n\n".join(formatted).strip()

def split_audio(input_path, output_dir, segment_duration=SEGMENT_DURATION):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        os.path.join(output_dir, "segment_%03d.wav"),
        "-y",
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".wav")
    )

def registar_transcricao(nome_ficheiro: str):
    conn = sqlite3.connect("ouviescrevi.db")
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO transcricoes (ficheiro, data) VALUES (?, ?)",
            (nome_ficheiro, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()

# ──────────────────────────────────────────────────────────────────────────────
# Rotas
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/debug")
def debug():
    return {"status": "OK", "versao": "1.0"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logging.info("Upload recebido: %s", file.filename)
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Ficheiro > {MAX_FILE_SIZE_MB}MB")

    # guardar original
    orig_ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    tmp_path = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4()}{orig_ext}")
    with open(tmp_path, "wb") as tmp:
        tmp.write(contents)

    # converter para wav 16k mono
    audio_path = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.wav")
    try:
        conv = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", audio_path
        ]
        subprocess.run(conv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg error: %s", e.stderr.decode(errors="ignore"))
        raise HTTPException(status_code=400, detail="Erro ao converter áudio (ffmpeg).")
    finally:
        try: os.remove(tmp_path)
        except: pass

    split_dir = tempfile.mkdtemp(prefix="split_")
    try:
        parts = split_audio(audio_path, split_dir)
        full_text, formatted_text = [], []

        for part in parts:
            with open(part, "rb") as audio:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="verbose_json",
                )
            full_text.append(getattr(result, "text", "") or "")
            formatted_text.append(format_segments(getattr(result, "segments", [])))

        registar_transcricao(file.filename or "sem_nome")
        enviar_email_assunto(
            f"Nova transcrição recebida: {file.filename}",
            "Nova transcrição no Ouviescrevi",
        )

        return {
            "transcription": "\n".join(t for t in full_text if t).strip(),
            "formatted": "\n\n".join(t for t in formatted_text if t).strip(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Erro ao processar ficheiro")
        raise HTTPException(status_code=500, detail=f"Erro ao processar ficheiro: {e}")
    finally:
        try: os.remove(audio_path)
        except: pass
        try:
            for f in os.listdir(split_dir):
                try: os.remove(os.path.join(split_dir, f))
                except: pass
            os.rmdir(split_dir)
        except: pass

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
            prompt = (
                "Based on the transcript below, create clear bullet-point minutes including:"
                "\n- Topics discussed\n- Decisions\n- Owners (if mentioned)\n- Action items\n\n"
                f"Transcript:\n{req.text}"
            )
            sys = "You summarize transcripts into meeting minutes."
        else:
            prompt = f"Summarize clearly and concisely:\n\n{req.text}"
            sys = "You summarize transcripts."
    else:
        if req.mode == "minuta":
            prompt = (
                "A partir da transcrição abaixo, cria uma minuta em tópicos com:"
                "\n- Tópicos discutidos\n- Decisões\n- Responsáveis (se houver)\n- Ações\n\n"
                f"Transcrição:\n{req.text}"
            )
            sys = "Resumes transcrições em minutas."
        else:
            prompt = f"Resume de forma clara e concisa:\n\n{req.text}"
            sys = "És um assistente que resume transcrições de áudio."

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        enviar_email_assunto("Resumo com IA gerado", "Resumo criado no Ouviescrevi")
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

    idiomas = {
        "inglês": "English",
        "espanhol": "Spanish",
        "francês": "French",
        "alemão": "German",
        "italiano": "Italian",
        "português": "Portuguese",
    }
    if language not in idiomas:
        raise HTTPException(status_code=400, detail=f"Idioma não suportado: {language}")

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[
                {"role": "system", "content": f"Traduz o texto para {idiomas[language]}."},
                {"role": "user", "content": text},
            ],
            temperature=0.3,
        )
        enviar_email_assunto("Tradução com IA feita", "Tradução realizada no Ouviescrevi")
        return {"translation": resp.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ClassifyRequest(BaseModel):
    text: str
    token: str

@app.post("/classify")
async def classify_content(req: ClassifyRequest):
    require_token(req.token)
    prompt = (
        "Classifica o texto como uma das opções:\n"
        "- Entrevista\n- Aula\n- Podcast\n- Reunião\n- Apresentação\n- Testemunho\n- Conversa informal\n\n"
        f"Texto:\n{req.text}\n\n"
        "Responde só com uma etiqueta."
    )
    try:
        resp = client.chat.completions.create(
            model=CLS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=20,
        )
        enviar_email_assunto("Classificação com IA feita", "Classificação feita no Ouviescrevi")
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
            messages=[
                {"role": "system", "content": "Corrige ortografia e gramática mantendo o sentido e o tom."},
                {"role": "user", "content": text},
            ],
            temperature=0.2,
        )
        enviar_email_assunto("Correção com IA feita", "Texto corrigido no Ouviescrevi")
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
    prompt = (
        f"Gera um email em tom {req.tone} a partir do seguinte resumo/transcrição de uma reunião:\n\n"
        f"{req.text}\n\n"
        "O email deve ser claro, direto e adequado para enviar após a reunião."
    )
    try:
        resp = client.chat.completions.create(
            model=EML_MODEL,
            messages=[
                {"role": "system", "content": "Escreves emails profissionais a partir de resumos/transcrições."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        enviar_email_assunto("Email com IA gerado", "Email gerado no Ouviescrevi")
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
        hoje = date.today().isoformat()  # YYYY-MM-DD
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
        try: conn.close()
        except: pass

class QuestionRequest(BaseModel):
    text: str
    token: str
    lang: str = "pt"
    num_questions: int = 3

@app.post("/generate-questions")
async def generate_questions(req: QuestionRequest):
    require_token(req.token)
    if req.lang == "en":
        prompt = (
            f"Generate {req.num_questions} multiple-choice questions based on the text. "
            "For each: question, four options (A–D), correct answer, short explanation.\n\n"
            f"Text:\n{req.text}"
        )
        sys = "You create study questions."
    else:
        prompt = (
            f"Gera {req.num_questions} perguntas de escolha múltipla com base no texto. "
            "Para cada pergunta: enunciado, quatro opções (A–D), resposta correta e breve explicação.\n\n"
            f"Texto:\n{req.text}"
        )
        sys = "Cria perguntas de estudo."

    try:
        resp = client.chat.completions.create(
            model=SUM_MODEL,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=900,
        )
        enviar_email_assunto("Perguntas de estudo geradas", "Perguntas geradas no Ouviescrevi")
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
        enviar_email_assunto(f"Resumo gerado por URL:\n{url}", "Resumo por URL no Ouviescrevi")
        return {"summary": final_summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar URL: {e}")

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
        # 1) TTS
        audio_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        gTTS(text=req.text, lang=req.voice_lang).save(audio_tmp)

        # 2) imagem
        img_tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.jpg")
        rr = requests.get(req.image_url, timeout=10)
        ct = rr.headers.get("Content-Type", "")
        if rr.status_code == 200 and "image" in ct:
            with open(img_tmp, "wb") as f:
                f.write(rr.content)
        else:
            raise HTTPException(status_code=400, detail="Erro ao obter imagem.")

        # 3) FFmpeg → guarda em /static/videos
        out_name = f"{uuid.uuid4()}.mp4"
        out_path = os.path.join(VIDEO_DIR, out_name)
        cmd = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-loop", "1", "-i", img_tmp, "-i", audio_tmp,
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            "-shortest", "-y", out_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # limpar temporários
        for p in (audio_tmp, img_tmp):
            try: os.remove(p)
            except: pass

        return {"success": True, "video_url": f"/static/videos/{out_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Erro ao gerar vídeo")
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
