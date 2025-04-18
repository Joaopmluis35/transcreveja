from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from moviepy.editor import AudioFileClip
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
        clip = AudioFileClip(tmp_path)
        clip.write_audiofile(audio_path, codec="pcm_s16le")
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
