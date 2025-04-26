from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import subprocess
import openai
from datetime import datetime

print("\u2705 API DO OUVIESCREVI INICIADA")
print("\ud83d\udd11 Chave carregada:", bool(os.getenv("OPENAI_API_KEY")))

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
        "-c", "copy",
        os.path.join(output_dir, "segment_%03d.wav"),
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")])

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"\ud83d\udcc5 [{datetime.now()}] Upload recebido: {file.filename}")

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return {"error": f"Ficheiro demasiado grande. Limite: {MAX_FILE_SIZE_MB}MB"}

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

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

        return {
            "transcription": full_text.strip(),
            "formatted": formatted_text.strip()
        }
    except Exception as e:
        return {"error": f"Erro ao processar ficheiro: {str(e)}"}
    finally:
        os.remove(tmp_path)
        os.remove(audio_path)
        for f in os.listdir(split_dir):
            os.remove(os.path.join(split_dir, f))
        os.rmdir(split_dir)

class SummarizeRequest(BaseModel):
    text: str
    token: str = ""
    mode: str = "normal"

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    if req.token != os.getenv("ADMIN_TOKEN", "ouviescrevi2025@resumo"):
        return {"error": "Token inválido ou ausente."}

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
                {"role": "system", "content": "És um assistente que resume transcrições de áudio."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=400
        )
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

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Traduz o texto para {language}."},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
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
        return {"corrected": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}


class EmailRequest(BaseModel):
    text: str
    token: str
    tone: str = "formal"  # "formal" ou "informal"

@app.post("/generate-email")
async def generate_email(req: EmailRequest):
    if req.token != "ouviescrevi2025@resumo":
        return {"error": "Token inválido."}

    prompt = (
        f"Escreve um email em tom {req.tone}, com base na seguinte transcrição de conversa ou reunião:\n\n"
        f"{req.text}\n\n"
        f"O email deve ser claro, coeso e conter os pontos essenciais discutidos, podendo incluir agradecimentos ou ações a seguir, se apropriado."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "És um assistente que transforma transcrições em emails bem escritos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return {"email": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}
