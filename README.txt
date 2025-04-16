
# TranscreveJá com IA real

## Como usar

### Backend (FastAPI + Whisper)
1. Instala dependências:
   pip install fastapi uvicorn openai python-multipart

2. Define tua chave:
   Abre `main.py` e coloca tua chave OpenAI em: openai.api_key = "AQUI"

3. Corre o backend:
   uvicorn main:app --reload

### Frontend
Abre o ficheiro `index.html` no navegador.

## Teste via terminal (curl)

curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3"
