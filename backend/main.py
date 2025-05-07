from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import subprocess
import openai
from datetime import datetime
import uuid
from pydub.utils import mediainfo

print("API DO OUVIESCREVI INICIADA")
print("Chave carregada:", bool(os.getenv("OPENAI_API_KEY")))


app = FastAPI()
# Armazena datas de transcrições na memória
transcricoes_hoje = []
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

def transcricao_otimizada(contents, filename, client):
    import shutil

    tmp_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, f"{tmp_id}_original")
    wav_path = os.path.join(tmp_dir, f"{tmp_id}_converted.wav")
    trimmed_path = os.path.join(tmp_dir, f"{tmp_id}_trimmed.wav")

    try:
        # Salvar o ficheiro temporário original
        with open(input_path, "wb") as f:
            f.write(contents)

        # Convert to mono 16kHz WAV
        subprocess.run([
            "ffmpeg", "-i", input_path,
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", wav_path, "-y"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Remove silêncios do início/fim
        subprocess.run([
            "ffmpeg", "-i", wav_path,
            "-af", "silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.3:"
                   "stop_periods=1:stop_threshold=-45dB:stop_silence=0.5",
            "-y", trimmed_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Verificar duração
        info = mediainfo(trimmed_path)
        dur = float(info.get("duration", 0))

        full_text = ""
        formatted_text = ""

        # Se for longo, dividir
        if dur > SEGMENT_DURATION:
            split_dir = tempfile.mkdtemp()
            parts = split_audio(trimmed_path, split_dir)
            for part in parts:
                with open(part, "rb") as audio:
                    result = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        response_format="verbose_json"
                    )
                full_text += result.text + "\n"
                formatted_text += format_segments(result.segments) + "\n\n"
            shutil.rmtree(split_dir)
        else:
            with open(trimmed_path, "rb") as audio:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    response_format="verbose_json"
                )
            full_text = result.text
            formatted_text = format_segments(result.segments)

        transcricoes_registro.append(datetime.now())
        registar_transcricao(filename)

        return {
            "transcription": full_text.strip(),
            "formatted": formatted_text.strip()
        }

    except Exception as e:
        return {"error": f"Erro ao processar ficheiro: {str(e)}"}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"[{datetime.now()}] Upload recebido: {file.filename}")
    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return {"error": f"Ficheiro demasiado grande. Limite: {MAX_FILE_SIZE_MB}MB"}

    resultado = transcricao_otimizada(contents, file.filename, client)
    return resultado


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
        return {"email": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": str(e)}
import json

STATUS_FILE = "status.json"

@app.get("/api/status")
async def get_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler status: {e}")

@app.post("/api/status")
async def update_status(request: Request):
    try:
        data = await request.json()
        manutencao = data.get("manutencao", False)

        # Grava no ficheiro status.json
        with open(STATUS_FILE, "w") as f:
            json.dump({"manutencao": manutencao}, f)

        return {"message": "Estado atualizado com sucesso", "manutencao": manutencao}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar status: {e}")


import json

LOG_FILE = "log_transcricoes.json"

def registar_transcricao(nome_ficheiro):
    log = {
        "ficheiro": nome_ficheiro,
        "data": datetime.now().isoformat()
    }
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                historico = json.load(f)
        else:
            historico = []
        historico.append(log)
        with open(LOG_FILE, "w") as f:
            json.dump(historico, f, indent=2)
    except Exception as e:
        print("Erro ao registar transcrição:", e)
        
 
 
from datetime import datetime, date
from fastapi import Request

# Simulador simples (substitui por base de dados real se tiveres)
transcricoes_registro = []

@app.post("/registar-transcricao")
def registar_transcricao_route(req: Request):
    transcricoes_registro.append(datetime.now())
    return {"ok": True}


@app.get("/transcricoes-hoje")
def contar_transcricoes_hoje():
    hoje = date.today()
    total = sum(1 for d in transcricoes_registro if d.date() == hoje)
    return {"total": total}
@app.get("/api/logs")
def get_logs():
    try:
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([], f)

        with open(LOG_FILE, "r") as f:
            return json.load(f)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler logs: {e}")
