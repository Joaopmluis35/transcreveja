# CHANGELOG — TranscreveJá / Ouviescrevi

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/).  
O projeto não usa versionamento semântico rigoroso no código; as versões abaixo são **marcos documentais**.

---

## [Unreleased]

### Segurança (Sessão 2 — 22 jun 2026)

**Backend**

- Novo módulo `backend/security.py` — rate limiting por IP, validação de origem, proteção SSRF em URLs externas.
- Autenticação por `API_TOKEN` (Bearer / corpo / `X-API-Token`) em `/transcribe`, `/video-subs` e rotas de IA.
- `ADMIN_TOKEN` separado para backoffice (`/api/logs`, `/transcricoes-hoje`, `POST /api/status`).
- Login server-side `POST /api/admin/login` com `BACKOFFICE_PASSWORD` (obrigatório no `.env`).
- `GET /api/frontend-config` — devolve token só para origens em `ALLOWED_ORIGINS`.
- CORS restrito a `ALLOWED_ORIGINS` (deixa de ser `*`).
- Endpoints de diagnóstico (`/debug`, `/`, `/rotas`, `/test-email`) só com `ENABLE_DEBUG_ENDPOINTS=true`.
- Versão em `/debug` atualizada para **1.6**.
- `backend/.env.example` expandido com variáveis de segurança.

**Frontend**

- Novo cliente `frontend/js/ouviescrevi-api.js` — token e base URL vêm do servidor, não do HTML.
- Removidos token `ouviescrevi2025@resumo` e password `admin123.` de todas as páginas.
- `backoffice.html` — login via API; logs renderizados com `textContent` (mitiga XSS).
- Páginas PT, EN, `admin.html`, `index2.html` e `app.js` migrados para `OuviescreviAPI`.

**Repositório**

- `.gitignore` — `backend/.venv/`, `backend/venv/`, logs e vídeos gerados.

### Documentação

- Adicionado `PROJECT_AUDIT.md` — auditoria estática completa do backend e frontend.
- Adicionado `NEXT_STEPS.md` — plano de trabalho priorizado para relançamento.
- Adicionado `BACKUP_NOTES.md` — procedimentos de backup e rotação de segredos.
- Adicionado `CHANGELOG.md` — este ficheiro.
- Atualizado `PROJECT_AUDIT.md` com resultados da **Sessão 1** (validação local).

### Validação local (Sessão 1 — 22 jun 2026)

- Criado ambiente `backend/.venv` com Python 3.14.6.
- `pip install -r requirements.txt` concluído com sucesso (FastAPI 0.138, OpenAI SDK 2.43, etc.).
- Backend arrancou com `uvicorn` em `http://127.0.0.1:8000`.
- `GET /debug` → `200`, versão `1.5`.
- `GET /api/status` → `200`, `manutencao: false` (SQLite novo).
- `POST /transcribe` → pipeline OK (upload, FFmpeg, segmentação); Whisper falhou com chave placeholder (esperado).
- `POST /summarize` → auth por token OK; OpenAI `401` com chave inválida (esperado).
- FFmpeg operacional via `imageio-ffmpeg` (binário embutido no pip).

### Pendente antes de produção

- Criar `backend/.env` com chaves reais (`OPENAI_API_KEY`, `BACKOFFICE_PASSWORD`, `API_TOKEN`, `ADMIN_TOKEN`).
- **Rodar** o token antigo `ouviescrevi2025@resumo` — estava exposto no frontend público.
- Deploy coordenado backend + frontend; confirmar `ALLOWED_ORIGINS` com domínios reais.
- Testar transcrição ponta a ponta com chave OpenAI válida.
- Remover ou deixar de versionar `backend/venv/` legado no repositório remoto.

---

## [1.5] — Backend (referência em `/debug`)

Marco referenciado pelo endpoint `GET /debug` (`"versao": "1.5"`).

### Funcionalidades conhecidas nesta versão

- Transcrição áudio/vídeo com Whisper (`/transcribe`).
- Legendas SRT + vídeo com legendas queimadas (`/video-subs`).
- Resumo, minuta, tradução, classificação, correção, email e perguntas (`/summarize`, `/translate`, `/classify`, `/correct`, `/generate-email`, `/generate-questions`).
- Resumo por URL (`/summarize-url`).
- Geração de vídeo com gTTS (`/generate-video`).
- Modo manutenção e estatísticas (`/api/status`, `/transcricoes-hoje`, `/api/logs`).
- Notificação de partilha WhatsApp (`/notify-whatsapp-share`).
- Logging rotativo, SQLite, ficheiros estáticos em `/static/videos/`.

### Limitações conhecidas

- Token de admin exposto no frontend.
- Vários endpoints administrativos e de transcrição sem autenticação adequada.
- `style` de legendas no frontend não aplicado no backend.
- Exportação «Word» no frontend não é DOCX real.

---

## Histórico anterior (inferido do repositório)

> Não existe histórico Git detalhado nem tags no repositório no momento da auditoria. As entradas abaixo descrevem evolução observada no código e ficheiros.

### Frontend — evolução das páginas

- **`index.html`** — aplicação principal PT (transcrição, IA, legendas, exportações).
- **`en/index.html`** — versão inglesa.
- **`index2.html`**, **`admin.html`** — variantes/duplicados da interface principal.
- **`backoffice.html`** — painel de manutenção e estatísticas.
- **`conversor.html`** — conversões client-side (Word, PDF, imagem).
- **`resumo.html`**, **`url-resumo.html`**, **`perguntas.html`**, **`corretor.html`**, **`gerar-video.html`** — ferramentas especializadas.
- Páginas SEO por nicho (`jornalistas.html`, `aulas.html`, etc.).
- **`frontend/api/logs.ts`** — função serverless Vercel (separada da API FastAPI).

### Backend — evolução técnica

- Migração de configuração inline para `.env` (`OPENAI_API_KEY`, `ADMIN_TOKEN`).
- Integração OpenAI SDK moderno (`OpenAI` client).
- Segmentação de áudio com FFmpeg e retries Whisper.
- Endpoint `/video-subs` para legendas embutidas.
- Middleware de logging com request ID e envio opcional para Vercel.
- Base SQLite para transcrições e flag de manutenção.

### Artefactos legados

- **`README.txt`** — instruções antigas (chave em `main.py`, dependências mínimas).
- **`backend/venv/`** — ambiente virtual Windows de outra máquina (`Administrador`); não portável.
- **`backend/status.json`** — `manutencao: true`; substituído por SQLite em runtime.
- **`app.js`** — cliente de upload simples apontando para API de produção.

---

## Como atualizar este ficheiro

Ao concluir cada fase de `NEXT_STEPS.md`:

1. Mover itens de `[Unreleased]` para uma nova secção `[X.Y] - AAAA-MM-DD`.
2. Registar apenas mudanças **visíveis** ou **operacionais** (features, fixes, segurança, deploy).
3. Não listar valores de segredos nem conteúdo de `.env`.

### Tipos de entrada

- **Added** — funcionalidade nova.
- **Changed** — alteração de comportamento existente.
- **Deprecated** — funcionalidade marcada para remoção.
- **Removed** — funcionalidade removida.
- **Fixed** — correção de bug.
- **Security** — correções de segurança (sem detalhar exploits).

---

## Referências

- `PROJECT_AUDIT.md`
- `NEXT_STEPS.md`
- `BACKUP_NOTES.md`
- `PROJECT_BRIEF.md`
