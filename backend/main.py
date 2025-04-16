from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from moviepy.editor import AudioFileClip
import openai

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cliente OpenAI
client = openai.OpenAI(api_key="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")

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

    # Transcrição formatada com timestamps
    formatted = format_segments(transcript.segments)

    return {
        "transcription": transcript.text,
        "formatted": formatted
    }