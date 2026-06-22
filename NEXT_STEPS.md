# NEXT_STEPS — TranscreveJá / Ouviescrevi

**Última atualização:** 22 de junho de 2026  
**Contexto:** `PROJECT_AUDIT.md`, `PROJECT_BRIEF.md`, validação local da Sessão 1.

Este documento define **o que fazer a seguir**, em ordem, sem alterar código até cada passo ser explicitamente aprovado.

---

## Estado atual (resumo)

| Área | Estado |
|------|--------|
| Backend arranca localmente | **Sim** — com `backend/.venv`, Python 3.14, `pip install -r requirements.txt` |
| FFmpeg | **OK** via `imageio-ffmpeg` |
| Transcrição real (Whisper) | **Pendente** — requer `OPENAI_API_KEY` válida em `backend/.env` |
| Frontend + backend local | **Integrado** — `ouviescrevi-api.js` + `ALLOWED_ORIGINS` (ex.: `:5500`) |
| Segurança P0 no código | **Implementado** (Sessão 2) — ver `CHANGELOG.md` |
| Deploy produção | **Pendente** — rodar tokens, `.env` no servidor, deploy coordenado |
| Produção atual | **Não mexer** até Fase 3 do plano |

---

## Fase 0 — Concluir validação local (tu)

Objetivo: confirmar que a IA funciona de ponta a ponta na tua máquina.

### 0.1 Criar `backend/.env`

```dotenv
OPENAI_API_KEY=sk-...
ADMIN_TOKEN=gera-um-token-longo-e-novo
API_TOKEN=gera-outro-token-para-o-frontend
BACKOFFICE_PASSWORD=escolhe-uma-password-forte
ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
ENABLE_DEBUG_ENDPOINTS=true
APP_ENV=development
```

Copiar o resto de `backend/.env.example` conforme necessário.

### 0.2 Arrancar o backend

```powershell
cd C:\Users\joao_\transcreveja\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### 0.3 Testes manuais

| Teste | Comando / ação | Sucesso esperado |
|-------|----------------|------------------|
| Health | `http://127.0.0.1:8000/debug` | `{"status":"OK","versao":"1.6"}` (só com `ENABLE_DEBUG_ENDPOINTS=true`) |
| Config frontend | `GET /api/frontend-config` com `Origin: http://127.0.0.1:5500` | JSON com `token` e `apiBase` |
| Transcrição | `POST /transcribe` com ficheiro + `Authorization: Bearer <API_TOKEN>` | JSON com `transcription` |
| Resumo | `POST /summarize` com Bearer token | JSON com `summary` |
| Token inválido | Pedido sem token ou token errado | HTTP 403 |

### 0.4 Registar resultado

Anotar em `CHANGELOG.md` (secção *Unreleased*) se os testes passaram ou que erros apareceram.

---

## Fase 1 — Segurança mínima

**Estado: implementado no código (Sessão 2, 22 jun 2026).** Ver `CHANGELOG.md`.

Antes de expor publicamente, ainda é obrigatório:

### 1.1 Rotacionar segredos (tu — servidor)

1. Gerar novo `ADMIN_TOKEN` (32+ caracteres aleatórios).
2. Se o token antigo esteve em produção, assumir que está comprometido.
3. Rever chave OpenAI na [dashboard OpenAI](https://platform.openai.com/api-keys) — criar nova se necessário.
4. Atualizar apenas `backend/.env` e variáveis do servidor de produção (quando autorizado).

### 1.2–1.5 Código — concluído

- Segredos removidos do frontend; `ouviescrevi-api.js` obtém token via `/api/frontend-config`.
- Endpoints caros e admin protegidos no `main.py`; CORS, SSRF e rate limits em `security.py`.
- Backoffice com login server-side e mitigação XSS nos logs.

---

## Fase 2 — Ambiente reproduzível e documentação

### 2.1 Git e ambiente

- Adicionar ao `.gitignore`: `backend/venv/`, `backend/.venv/`, `backend/logs/`, `backend/static/videos/`.
- Remover `backend/venv/` do controlo de versão (quando houver commit autorizado).
- Fixar versões: `pip freeze > requirements-lock.txt` ou pins no `requirements.txt`.

### 2.2 Ficheiros de configuração

- Criar `backend/.env.example` (sem segredos).
- Atualizar ou substituir `README.txt` com instruções corretas.

### 2.3 FFmpeg (opcional, sistema)

O source em `C:\Users\joao_\Downloads\ffmpeg-8.1.2\` **não é necessário** para o projeto — o pip já inclui binário.

Se quiseres FFmpeg global no PATH: instalar [build essentials Windows](https://www.gyan.dev/ffmpeg/builds/) (pasta `bin` com `ffmpeg.exe`).

### 2.4 Python

Testado com **3.14.6**. Se surgirem incompatibilidades, recriar venv com 3.11 ou 3.12:

```powershell
py -3.12 -m venv .venv
```

---

## Fase 3 — Integração frontend local

**Sem deploy em produção.**

### 3.1 API configurável

- Uma única variável `API_BASE` (ex.: injetada por script ou ficheiro `config.js`).
- Dev: `http://127.0.0.1:8000`
- Prod: `https://api.ouviescrevi.pt`

### 3.2 Servir frontend

```powershell
cd frontend
py -m http.server 5500
```

Abrir `http://127.0.0.1:5500/` — **não** usar `file://`.

### 3.3 Consolidar páginas duplicadas

- Definir `index.html` como canónica.
- Arquivar `index2.html` e `admin.html` após confirmação.

---

## Fase 4 — Estabilidade e qualidade

### 4.1 Testes mínimos

- Smoke: arranque, `/debug`, auth token, upload vazio, ficheiro > limite.
- Mocks para OpenAI em CI (quando existir CI).

### 4.2 Bugs conhecidos a corrigir

| Bug | Ficheiro | Notas |
|-----|----------|-------|
| `gerar-video.html` não envia `token` | frontend | 403 em `/generate-video` |
| `style` de legendas ignorado | backend | UI engana o utilizador |
| Backoffice `fetch('/api/logs')` | frontend | Pode bater na função Vercel, não no FastAPI |
| PDF páginas fora de ordem | conversor.html | Race condition |

### 4.3 Observabilidade

- Retenção de `static/videos/` e logs.
- Métricas de duração e custo (sem gravar conteúdo sensível).

---

## Fase 5 — Produto (após P0–P2)

Alinhado com `PROJECT_BRIEF.md`:

1. Exportação DOCX real (hoje é `.doc` textual).
2. OCR imagem → texto no conversor.
3. Diarização real de locutores (hoje é heurística João/Maria).
4. Histórico por utilizador / quotas.
5. Fila de jobs para transcrições longas.

---

## Decisões pendentes (precisam da tua resposta)

| # | Pergunta | Opções |
|---|----------|--------|
| 1 | Relançar só API ou API + site? | API primeiro / conjunto |
| 2 | Manter SQLite ou migrar? | SQLite OK para MVP / Postgres depois |
| 3 | Onde hospedar backend? | VPS, Railway, Render, etc. |
| 4 | Rotacionar token em produção agora? | Sim (recomendado) / só após código novo |
| 5 | Unificar com domínio `transcreveja` ou manter `ouviescrevi.pt`? | — |

---

## Checklist rápido — «Posso abrir ao público?»

```
[ ] OPENAI_API_KEY válida testada com áudio real
[ ] ADMIN_TOKEN novo, só no servidor
[ ] Nenhum segredo no frontend
[ ] /transcribe e /video-subs protegidos ou com rate limit forte
[ ] /api/status POST protegido
[ ] /test-email desativado em produção
[ ] CORS restrito
[ ] Backoffice com auth real
[ ] .env e backups documentados (BACKUP_NOTES.md)
[ ] CHANGELOG atualizado com versão de relançamento
```

**Se algum item falhar: não relançar.**

---

## Referências

- `PROJECT_AUDIT.md` — auditoria completa e riscos
- `BACKUP_NOTES.md` — o que guardar antes de mudanças
- `CHANGELOG.md` — histórico de alterações
- `PROJECT_BRIEF.md` — objetivos e regras do projeto
