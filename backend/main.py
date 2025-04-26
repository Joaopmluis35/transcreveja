from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import subprocess
import openai
from datetime import datetime

print("✅ API DO OUVIESCREVI INICIADA")
print("🔑 Chave carregada:", bool(os.getenv("OPENAI_API_KEY")))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_segments(segments):
    def format_time(seconds):
        m, s = divmod(int(seconds), 60)
        return f"[{m:02d}:{s:02d}]"
    formatted_text = ""
    for s in segments:
        timestamp = format_time(s.start)
        formatted_text += f"{timestamp} {s.text.strip()}\n\n"
    return formatted_text.strip()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"📥 [{datetime.now()}] Upload recebido: {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    audio_path = tmp_path + ".wav"

    try:
        # Usa ffmpeg diretamente (melhor suporte para formatos como webm, mp4 etc.)
        command = [
            "ffmpeg",
            "-i", tmp_path,
            "-ar", "16000",  # amostragem
            "-ac", "1",      # mono
            "-c:a", "pcm_s16le",  # formato WAV compatível
            audio_path,
            "-y"
        ]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception as e:
        print(f"❌ Erro ao converter áudio ({file.filename}): {e}")
        return {"error": f"Erro ao converter áudio: {str(e)}"}

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
    except Exception as e:
        print(f"❌ Erro ao transcrever áudio ({file.filename}): {e}")
        return {"error": f"Erro ao transcrever: {str(e)}"}

    os.remove(tmp_path)
    os.remove(audio_path)

    print(f"✅ [{datetime.now()}] Transcrição concluída com sucesso: {file.filename}")

    formatted = format_segments(transcript.segments)
    return {
        "transcription": transcript.text,
        "formatted": formatted
    }
from pydantic import BaseModel
from fastapi import Request

class SummarizeRequest(BaseModel):
    text: str
    token: str = ""
    mode: str = "normal"


@app.post("/summarize")
async def summarize(req: SummarizeRequest, request: Request):
    if req.token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido ou ausente."}

    if req.mode == "minuta":
        prompt = (
            "A partir da seguinte transcrição de uma reunião ou conversa, gera uma minuta clara e organizada "
            "em formato de tópicos. Inclui:\n"
            "- Tópicos discutidos\n"
            "- Decisões tomadas\n"
            "- Responsáveis (se mencionados)\n"
            "- Ações a realizar\n\n"
            f"Transcrição:\n{req.text}"
        )
    else:
        prompt = f"Resume de forma clara e concisa a seguinte transcrição:\n\n{req.text}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": "És um assistente que resume transcrições de áudio." },
                { "role": "user", "content": prompt }
            ],
            temperature=0.5,
            max_tokens=400
        )
        summary = response.choices[0].message.content.strip()
        return { "summary": summary }

    except Exception as e:
        print("❌ Erro ao gerar resumo:", e)
        return { "error": str(e) }


from fastapi import HTTPException  # certifica-te que está importado

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    text = data.get("text")
    language = data.get("language")
    token = data.get("token")

    if token != "ouviescrevi2025@resumo":
        raise HTTPException(status_code=403, detail="Token inválido.")

    # ✅ Lista de idiomas suportados (em minúsculas)
    idiomas_suportados = ["inglês", "espanhol", "francês", "alemão", "italiano", "português"]

    if language.lower() not in idiomas_suportados:
        return { "error": f"Idioma não suportado: {language}" }

    # ✅ Prompt para tradução
    prompt = f"Traduz o seguinte texto para {language}:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                { "role": "system", "content": f"Traduz o texto para {language}." },
                { "role": "user", "content": text }
            ],
            temperature=0.3
        )

        translated = response.choices[0].message.content.strip()
        return { "translation": translated }

    except Exception as e:
        print("❌ Erro ao traduzir:", e)
        return { "error": str(e) }

from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import os



# Modelo da requisição
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
        f"Texto:\n{request.text}\n\n"
        "Responde só com o tipo mais provável."
    )

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=20
        )

        tipo = response.choices[0].message.content.strip()
        return {"type": tipo}

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
        corrected = response.choices[0].message.content
        return {"corrected": corrected.strip()}
    except Exception as e:
        print("❌ Erro ao corrigir:", e)
        return {"error": str(e)}
