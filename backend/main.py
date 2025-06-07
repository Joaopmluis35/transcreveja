from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, date
import tempfile
import os
import subprocess
import openai
import uuid
import sqlite3
import json
import os
from dotenv import load_dotenv
load_dotenv()
import smtplib
from email.message import EmailMessage
from fastapi import APIRouter

import os
from dotenv import load_dotenv

load_dotenv()  # já tens isso no topo

def enviar_email_assunto(mensagem: str, assunto: str = "Nova atividade no Ouviescrevi"):
    try:
        msg = EmailMessage()
        msg.set_content(mensagem)
        msg['Subject'] = assunto
        msg['From'] = "notificacoes@ouviescrevi.pt"
        msg['To'] = "ouviescrevi@gmail.com"

        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(msg)
    except Exception as e:
        print(f"[ERRO AO ENVIAR EMAIL] {str(e)}")



from database import criar_base
criar_base()

print("API DO OUVIESCREVI INICIADA")
print("Chave carregada:", bool(os.getenv("OPENAI_API_KEY")))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_FILE_SIZE_MB = 25
SEGMENT_DURATION = 600  # segundos (10 minutos)

def format_segments(segments):
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"[{m:02d}:{s:02d}]"
    formatted_text = ""
    for s in segments:
        timestamp = format_time(s.start)
        formatted_text += f"{timestamp} {s.text.strip()}\n\n"
    return formatted_text.strip()

def split_audio(input_path, output_dir, segment_duration=SEGMENT_DURATION):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "ffmpeg",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",  # garante áudio limpo
        os.path.join(output_dir, "segment_%03d.wav"),
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")])
@app.get("/debug")
def debug():
    return {"status": "OK", "versao": "1.0"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"[{datetime.now()}] Upload recebido: {file.filename}")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return {"error": f"Ficheiro demasiado grande. Limite: {MAX_FILE_SIZE_MB}MB"}

    tmp_path = os.path.join(tempfile.gettempdir(), f"input_{uuid.uuid4()}")
    with open(tmp_path, "wb") as tmp:
        tmp.write(contents)

    audio_path = tmp_path + ".wav"

    try:
        subprocess.run([
            "ffmpeg", "-i", tmp_path,
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", audio_path, "-y"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        return {"error": f"Erro ao converter áudio: {str(e)}"}

    split_dir = tempfile.mkdtemp()
    try:
        parts = split_audio(audio_path, split_dir)
        full_text, formatted_text = "", ""

        for part in parts:
            with open(part, "rb") as audio:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="verbose_json"
                )
            full_text += result.text + "\n"
            formatted_text += format_segments(result.segments) + "\n\n"

        registar_transcricao(file.filename)
        enviar_email_assunto(f"Nova transcrição recebida: {file.filename}", "Nova transcrição no Ouviescrevi")




        return {
            "transcription": full_text.strip(),
            "formatted": formatted_text.strip()
        }
    except Exception as e:
        return {"error": f"Erro ao processar ficheiro: {str(e)}"}
    finally:
        for path in [tmp_path, audio_path]:
            try:
                os.remove(path)
            except: pass
        for f in os.listdir(split_dir):
            try:
                os.remove(os.path.join(split_dir, f))
            except: pass
        try:
            os.rmdir(split_dir)
        except: pass

class SummarizeRequest(BaseModel):
    text: str
    token: str = ""
    mode: str = "normal"
    lang: str = "pt"  # novo campo com valor padrão


@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    if req.token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido ou ausente."}

    if req.lang == "en":
        if req.mode == "minuta":
            prompt = (
                "Based on the following meeting or conversation transcript, generate a clear and organized minutes "
                "in bullet point format. Include:\n"
                "- Topics discussed\n- Decisions made\n- Responsible persons (if mentioned)\n- Action items\n\n"
                f"Transcript:\n{req.text}"
            )
        else:
            prompt = f"Summarize the following transcript in a clear and concise way:\n\n{req.text}"
    else:
        if req.mode == "minuta":
            prompt = (
                "A partir da seguinte transcrição de uma reunião ou conversa, gera uma minuta clara e organizada "
                "em formato de tópicos. Inclui:\n"
                "- Tópicos discutidos\n- Decisões tomadas\n- Responsáveis (se mencionados)\n- Ações a realizar\n\n"
                f"Transcrição:\n{req.text}"
            )
        else:
            prompt = f"Resume de forma clara e concisa a seguinte transcrição:\n\n{req.text}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes transcripts." if req.lang == "en" else "És um assistente que resume transcrições de áudio."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
        enviar_email_assunto("Alguém gerou um resumo com IA", "Resumo criado no Ouviescrevi")

        return {"summary": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    text = data.get("text")
    language = data.get("language")
    token = data.get("token")

    if token != "ouviescrevi2025@resumo":
        raise HTTPException(status_code=403, detail="Token inválido.")

    idiomas_suportados = ["inglês", "espanhol", "francês", "alemão", "italiano", "português"]
    if language.lower() not in idiomas_suportados:
        return {"error": f"Idioma não suportado: {language}"}

    # Mapeamento dos nomes em português para o nome em inglês (esperado pela OpenAI)
    idioma_map = {
        "inglês": "English",
        "espanhol": "Spanish",
        "francês": "French",
        "alemão": "German",
        "italiano": "Italian",
        "português": "Portuguese"
    }
    language_en = idioma_map.get(language.lower(), language)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Traduz o texto para {language_en}."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        enviar_email_assunto("Alguém traduziu um texto com IA", "Tradução realizada no Ouviescrevi")

        return {"translation": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}


class ClassifyRequest(BaseModel):
    text: str
    token: str

@app.post("/classify")
async def classify_content(request: ClassifyRequest):
    if request.token != "ouviescrevi2025@resumo":
        return {"error": "Token inválido"}

    prompt = (
        "Classifica o tipo de conteúdo abaixo como uma das seguintes opções:\n"
        "- Entrevista\n- Aula\n- Podcast\n- Reunião\n- Apresentação\n- Testemunho\n- Conversa informal\n\n"
        f"Texto:\n{request.text}\n\nResponde só com o tipo mais provável."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=20
        )
        enviar_email_assunto("Alguém classificou um tipo de conteúdo com IA", "Classificação feita no Ouviescrevi")

        return {"type": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.post("/correct")
async def correct_text(req: Request):
    data = await req.json()
    text = data.get("text", "")
    token = data.get("token")

    if token != "ouviescrevi2025@resumo":
        return {"error": "Token inválido"}

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Corrige ortografia e gramática do texto. Mantém o conteúdo original."},
                {"role": "user", "content": text}
            ]
        )
        enviar_email_assunto("Alguém corrigiu um texto com IA", "Texto corrigido no Ouviescrevi")

        return {"corrected": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}


class EmailRequest(BaseModel):
    text: str
    token: str
    tone: str = "formal"

@app.post("/generate-email")
async def generate_email(req: EmailRequest):
    if req.token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido"}

    prompt = (
        f"Gera um email em tom {req.tone} a partir da seguinte transcrição ou resumo de uma reunião:\n\n"
        f"{req.text}\n\n"
        "O email deve ser claro, direto e adequado para enviar após a reunião."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "És um assistente que escreve e-mails profissionais a partir de resumos ou transcrições."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        enviar_email_assunto("Alguém gerou um email com IA", "Email gerado no Ouviescrevi")

        return {"email": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/status")
def get_status():
    conn = sqlite3.connect("ouviescrevi.db")
    cursor = conn.cursor()
    cursor.execute("SELECT manutencao FROM status WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    return {"manutencao": bool(row[0])}

@app.post("/api/status")
async def update_status(request: Request):
    data = await request.json()
    manutencao = bool(data.get("manutencao", False))

    conn = sqlite3.connect("ouviescrevi.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE status SET manutencao = ? WHERE id = 1", (manutencao,))
    conn.commit()
    conn.close()

    return {"message": "Estado atualizado com sucesso", "manutencao": manutencao}





LOG_FILE = "log_transcricoes.json"

def registar_transcricao(nome_ficheiro):
    conn = sqlite3.connect("ouviescrevi.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transcricoes (ficheiro, data) VALUES (?, ?)",
        (nome_ficheiro, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

        
 
 
from datetime import datetime, date
from fastapi import Request

# Simulador simples (substitui por base de dados real se tiveres)
import sqlite3




@app.get("/transcricoes-hoje")
def contar_transcricoes_hoje():
    conn = sqlite3.connect("ouviescrevi.db")
    cursor = conn.cursor()
    hoje = date.today().isoformat()
    cursor.execute("SELECT COUNT(*) FROM transcricoes WHERE DATE(data) = ?", (hoje,))
    total = cursor.fetchone()[0]
    conn.close()
    return {"total": total}

@app.get("/api/logs")
def get_logs():
    try:
        conn = sqlite3.connect("ouviescrevi.db")
        cursor = conn.cursor()
        cursor.execute("SELECT ficheiro, data FROM transcricoes ORDER BY data DESC")
        rows = cursor.fetchall()
        conn.close()
        return [{"ficheiro": r[0], "data": r[1]} for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler logs: {e}")


class QuestionRequest(BaseModel):
    text: str
    token: str
    lang: str = "pt"
    num_questions: int = 3

@app.post("/generate-questions")
async def generate_questions(req: QuestionRequest):
    if req.token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido"}

    if req.lang == "en":
        prompt = (
            f"Generate {req.num_questions} multiple-choice questions based on the text below. For each question, provide:\n"
            f"- The question itself\n- Four options (A to D)\n- The correct answer\n- A short explanation\n\n"
            f"Text:\n{req.text}"
        )
        system_message = "You are an assistant that creates study questions based on provided content."
    else:
        prompt = (
            f"Gera {req.num_questions} perguntas de escolha múltipla com base no texto abaixo. Para cada pergunta inclui:\n"
            f"- A pergunta\n- Quatro opções (A a D)\n- A resposta correta\n- Uma breve explicação\n\n"
            f"Texto:\n{req.text}"
        )
        system_message = "És um assistente que cria perguntas de estudo com base no conteúdo fornecido."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        enviar_email_assunto("Alguém gerou perguntas de estudo com IA", "Perguntas geradas no Ouviescrevi")

        return {"questions": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "method": list(route.methods)[0] if route.methods else None,
            "name": route.name
        })
    return {"routes": routes}
    
@app.get("/test-email")
def test_email():
    try:
        enviar_email_assunto("Teste de envio", "Teste SMTP Ouviescrevi")
        return {"status": "Enviado com sucesso"}
    except Exception as e:
        return {"error": str(e)}


from bs4 import BeautifulSoup
import requests

from bs4 import BeautifulSoup
import requests
import textwrap

@app.post("/summarize-url")
async def summarize_url(req: Request):
    data = await req.json()
    url = data.get("url")
    token = data.get("token")
    mode = data.get("mode", "normal")
    lang = data.get("lang", "pt")

    if token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido."}

    try:
        # Extrai texto da página
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        full_text = " ".join([p.get_text() for p in paragraphs if len(p.get_text()) > 40])

        if not full_text:
            return {"error": "Não foi possível extrair conteúdo significativo da URL."}

        # Divide o texto em blocos seguros (~3000 caracteres ≈ ~1000 tokens)
        chunks = textwrap.wrap(full_text, 3000)
        all_summaries = []

        for i, chunk in enumerate(chunks):
            if lang == "en":
                if mode == "minuta":
                    prompt = (
                        "Generate bullet point meeting minutes from this article section:\n\n" + chunk
                    )
                else:
                    prompt = f"Summarize this section clearly and concisely:\n\n{chunk}"
                system_message = "You are an assistant that summarizes online articles."
            else:
                if mode == "minuta":
                    prompt = (
                        "Gera uma minuta em tópicos com base nesta parte do artigo:\n\n" + chunk
                    )
                else:
                    prompt = f"Resume esta secção de forma clara e concisa:\n\n{chunk}"
                system_message = "És um assistente que resume artigos online."

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )

            summary = response.choices[0].message.content.strip()
            all_summaries.append(f"🧩 Parte {i+1}:\n{summary}")

        final_summary = "\n\n".join(all_summaries)

        # ✅ Enviar email de notificação
        enviar_email_assunto(f"Resumo gerado por URL:\n{url}", "Resumo por URL no Ouviescrevi")

        return {"summary": final_summary}

    except Exception as e:
        return {"error": f"Erro ao processar URL: {str(e)}"}


# Router
router = APIRouter()

class VideoRequest(BaseModel):
    text: str
    image_url: str = "https://placehold.co/720x1280?text=Ouviescrevi"
    voice_lang: str = "pt"

@router.post("/generate-video")
async def generate_video(req: VideoRequest):
    try:
        tts = gTTS(text=req.text, lang=req.voice_lang)
        audio_path = "/tmp/audio.mp3"
        tts.save(audio_path)

        image_path = "/tmp/image.jpg"
        response = requests.get(req.image_url)
        if response.status_code == 200 and 'image' in response.headers.get("Content-Type", ""):
            with open(image_path, "wb") as f:
                f.write(response.content)
        else:
            return {"success": False, "error": "Erro ao baixar a imagem."}

        output_path = "/tmp/video.mp4"
        command = f"ffmpeg -loop 1 -i {image_path} -i {audio_path} -c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -shortest -y {output_path}"
        subprocess.call(command, shell=True)

        return {
            "success": True,
            "video_url": f"https://api.ouviescrevi.pt/static/{os.path.basename(output_path)}"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Registrar router no final
app.include_router(router)


@app.get("/rotas")
def rotas():
    return [route.path for route in app.routes]
