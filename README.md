# Ouviescrevi / TranscreveJá

Plataforma de transcrição e ferramentas de IA (resumo, tradução, correção, etc.) — backend FastAPI + frontend estático.

## Requisitos

- **Python 3.11+** (testado com 3.14)
- **FFmpeg** — incluído via `imageio-ffmpeg` no pip (não é obrigatório instalar manualmente)
- Chave **OpenAI** para transcrição Whisper e funcionalidades de IA

## Instalação rápida

### 1. Backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configuração

```powershell
copy .env.example .env
# Editar .env — ver variáveis obrigatórias abaixo
```

Ou gerar um `.env` de desenvolvimento:

```powershell
.\.venv\Scripts\python.exe create_env.py
```

**Variáveis obrigatórias em `backend/.env`:**

| Variável | Descrição |
|----------|-----------|
| `OPENAI_API_KEY` | Chave da API OpenAI |
| `ADMIN_TOKEN` | Token para backoffice e rotas admin |
| `API_TOKEN` | Token para o frontend (transcrição / IA) |
| `BACKOFFICE_PASSWORD` | Password do painel `backoffice.html` |
| `ALLOWED_ORIGINS` | Origens CORS, ex.: `http://127.0.0.1:5500,http://localhost:5500` |

Para desenvolvimento local, define também `ENABLE_DEBUG_ENDPOINTS=true` e `APP_ENV=development`.

### 3. Arrancar o backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Teste: http://127.0.0.1:8000/debug (só com `ENABLE_DEBUG_ENDPOINTS=true`)

### 4. Frontend

Não uses `file://`. Serve os ficheiros com um servidor HTTP:

```powershell
cd frontend
py -m http.server 5500
```

Abre: http://127.0.0.1:5500/index.html

O cliente `js/ouviescrevi-api.js` deteta `localhost` e liga automaticamente ao backend em `http://127.0.0.1:8000`.

## Estrutura do projeto

```
backend/          API FastAPI (main.py, security.py)
frontend/         Site estático (index.html = página canónica)
frontend/archive/ Cópias legadas (index2, admin)
frontend/css/     Design system (ouviescrevi.css)
frontend/js/      Cliente API e utilitários UX
```

## Páginas principais

| Página | Função |
|--------|--------|
| `index.html` | Transcrição e ferramentas integradas |
| `backoffice.html` | Admin (manutenção, logs) |
| `resumo.html`, `corretor.html`, … | Ferramentas de IA |
| `gerar-video.html` | Vídeo com voz (gTTS — não precisa de OpenAI) |
| `index2.html`, `admin.html` | Redirecionam para `index.html` |

## Documentação adicional

- `PROJECT_AUDIT.md` — auditoria técnica
- `NEXT_STEPS.md` — plano de trabalho
- `BACKUP_NOTES.md` — backups e rotação de segredos
- `CHANGELOG.md` — histórico de alterações

## Teste rápido (com API key)

```powershell
# Health
curl http://127.0.0.1:8000/api/status

# Resumo (substituir TOKEN)
curl -X POST http://127.0.0.1:8000/summarize `
  -H "Authorization: Bearer TOKEN" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Olá mundo\",\"mode\":\"curto\"}"
```

## Produção

Antes de expor publicamente:

1. Rodar todos os tokens (o token antigo do frontend público está comprometido)
2. `ENABLE_DEBUG_ENDPOINTS=false`
3. `ALLOWED_ORIGINS` com os domínios reais
4. Deploy coordenado backend + frontend

Ver `NEXT_STEPS.md` e `BACKUP_NOTES.md`.
