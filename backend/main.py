from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from moviepy.editor import AudioFileClip
import openai
from datetime import datetime

print("API KEY DO AMBIENTE:", os.getenv("OPENAI_API_KEY"))  # Só para testar no Render!

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

# ✅ Versão final com logging incluído
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"[{datetime.now()}] Utilizador fez upload: {file.filename}")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    audio_path = tmp_path + ".wav"

    try:
        clip = AudioFileClip(tmp_path)
        clip.write_audiofile(audio_path, codec="pcm_s16le")
    except Exception as e:
        return {"error": f"Erro ao converter áudio: {str(e)}"}

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
    except Exception as e:
        return {"error": f"Erro ao transcrever: {str(e)}"}

    os.remove(tmp_path)
    os.remove(audio_path)

    formatted = format_segments(transcript.segments)

    return {
        "transcription": transcript.text,
        "formatted": formatted
    }
