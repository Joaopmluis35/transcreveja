# Ouviescrevi (TranscreveJá) — contexto para o Claude Code

Site de transcrição de áudio e vídeo com IA: https://www.ouviescrevi.pt (PT, EN, ES, FR, DE).
**Repositório público** (`Joaopmluis35/transcreveja`) — nunca escrever aqui nada dos outros projetos privados do João.

**Antes de propor alterações, lê o `CURSOR_HANDOFF.md`** (infraestrutura, histórico de commits, regras e ideias
pendentes) e atualiza-o no mesmo commit de cada alteração. O Cursor também trabalha aqui.

## Essencial
- **Frontend:** `frontend/` publicado na **Vercel** a cada push para `main` (regras em `/vercel.json`).
- **API:** `backend/` (FastAPI) no **Render, plano grátis** — adormece sem visitas (1.º pedido pode levar ~1 min).
  `https://api.ouviescrevi.pt`, base de dados Turso.
- **Testes:** `cd backend; py -m pytest -q` (141 a 21/11) e `py scripts/frontend_smoke_test.py`.
- **Cache:** CSS/JS com `?v=N`; ao mudar um ficheiro, sobe o `?v=` em **todas** as páginas que o usam
  (são ~150 páginas; usa um script em vez de editar à mão).
- A porta **8000** do PC está ocupada por outra app: para correr a API em local usa outra (ex.: 8010).

## Regras (detalhe na secção 3 do `CURSOR_HANDOFF.md`)
- Textos da IA em português de Portugal (`ai_output_language()`).
- Botões de IA: na PT o `comLoading` da `index.html`; nas EN/ES/FR/DE o `OuviescreviUI.wireActionLoading`.
  Nunca ler `event` depois de um `await`.
- Nunca `overflow:hidden` na `.oe-pro-nav`.
- Alterações pequenas e explicadas; **push só com pedido do João** (vai direto para o site público).
