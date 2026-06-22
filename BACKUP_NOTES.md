# BACKUP_NOTES — TranscreveJá / Ouviescrevi

**Última atualização:** 22 de junho de 2026  
**Objetivo:** registar o que deve ser guardado, copiado ou preservado **antes** de alterações de código, migrações, rotação de segredos ou deploy.

Seguir estas notas reduz o risco de perder dados, configuração ou acesso a produção.

---

## Princípios

1. **Nunca commitar segredos** — `.env`, chaves API, passwords SMTP.
2. **Assumir comprometimento** — o token `ouviescrevi2025@resumo` e a password `admin123.` estão no código público do repositório/frontend; tratá-los como expostos.
3. **Produção:** seguir a regra do brief — não alterar produção sem plano explícito; estes backups aplicam-se sobretudo antes de mudanças locais e, quando autorizado, antes de deploy.
4. **Testar restores** — um backup só é útil se souberes restaurá-lo.

---

## O que fazer backup

### Crítico (perda = paragem ou custo)

| Item | Localização típica | Como fazer backup | Notas |
|------|-------------------|-------------------|-------|
| `OPENAI_API_KEY` | `backend/.env`, painel OpenAI | Exportar/anotar em gestor de passwords; criar chave de reserva antes de revogar | Revogar chave antiga após rotação |
| `ADMIN_TOKEN` | `backend/.env`, variáveis do host | Gestor de passwords; gerar novo antes de deploy | Token no frontend está comprometido |
| Credenciais SMTP | `backend/.env` (`SMTP_USER`, `SMTP_PASSWORD`, etc.) | Gestor de passwords | Usadas em notificações por email |
| `VERCEL_LOG_TOKEN` / URL | `.env` se usado | Gestor de passwords | Opcional |
| Base de dados SQLite | `backend/ouviescrevi.db` | Copiar ficheiro com servidor **parado** ou usar backup online SQLite | Contém `transcricoes` e `status` |
| Código-fonte | Repositório Git | `git clone` / push para remote privado | Garantir remote atualizado antes de mudanças grandes |

### Importante (perda = inconveniência ou perda de histórico)

| Item | Localização | Backup |
|------|-------------|--------|
| Vídeos/SRT gerados | `backend/static/videos/` | Copiar pasta inteira | Podem conter conteúdo de utilizadores |
| Logs da aplicação | `backend/logs/ouviescrevi.log` (+ rotações `.1`, `.2`…) | Copiar pasta `logs/` | IPs, nomes de ficheiros, metadados |
| `status.json` legado | `backend/status.json` | Copiar | Estado real está em SQLite; ficheiro pode estar desatualizado |
| `log_transcricoes.json` | `backend/log_transcricoes.json` | Copiar | Atualmente vazio no repo |
| Frontend estático | `frontend/` | Git + cópia ZIP antes de refactors grandes | Muitas páginas duplicadas |
| Configuração DNS/domínios | Fora do repo | Anotar registos A/CNAME, SSL | `ouviescrevi.pt`, `api.ouviescrevi.pt` |
| Hospedagem API | Painel do provider | Exportar env vars, systemd, nginx, etc. | Não está no repositório |
| Hospedagem frontend | Vercel/outro | Exportar project settings, env, redirects | `api/logs.ts` pode estar deployado |
| Conta FormSubmit | `sugestoes.html` | Anotar email destino | `ouviescrevi@gmail.com` |

### Opcional / desenvolvimento

| Item | Notas |
|------|-------|
| `backend/venv/` ou `backend/.venv/` | **Não fazer backup** — recriar com `pip install -r requirements.txt` |
| `backend/__pycache__/` | Ignorar |
| FFmpeg source em Downloads | Não necessário ao projeto; binário vem do pip |
| `PROJECT_BRIEF.md`, auditorias, docs | Já no Git |

---

## Segredos conhecidos no código (tratar como comprometidos)

Estes valores aparecem em ficheiros versionados ou públicos — **não confiar neles** após relançamento:

| Segredo | Onde aparece | Ação recomendada |
|---------|--------------|------------------|
| `ouviescrevi2025@resumo` | *(removido do frontend — Sessão 2)* | **Rodar** em produção; usar `API_TOKEN` + `ADMIN_TOKEN` novos |
| `admin123.` | *(removido — Sessão 2)* | Definir `BACKOFFICE_PASSWORD` forte no servidor |
| Emails públicos | `footer.html`, FormSubmit, SMTP defaults | Não são segredos, mas validar se ainda são válidos |

---

## Procedimentos de backup

### A. Antes de alterar código ou dependências

```text
1. git status                    # ver alterações pendentes
2. git stash ou commit em branch # se necessário
3. Copiar backend/ouviescrevi.db → backup/ouviescrevi_YYYY-MM-DD.db
4. Copiar backend/.env           → local seguro FORA do repo (ex.: password manager + USB encriptado)
5. Copiar backend/logs/          → backup/logs_YYYY-MM-DD/ (opcional)
6. Copiar backend/static/videos/ → backup/videos_YYYY-MM-DD/ (se houver ficheiros importantes)
```

### B. Antes de rotação de `ADMIN_TOKEN` ou `OPENAI_API_KEY`

1. Anotar valores **atuais** em gestor de passwords (para rollback de emergência).
2. Gerar novos valores.
3. Atualizar `backend/.env` local.
4. Testar `/debug`, `/summarize` com novo token, `/transcribe` com nova chave OpenAI.
5. Só depois atualizar produção (quando autorizado).
6. Revogar chave/token antigo na OpenAI ou no servidor.

### C. Antes de remover `backend/venv/` do Git

1. Exportar lista de pacotes (referência):

   ```powershell
   cd backend
   .\.venv\Scripts\pip.exe freeze > ..\backup\pip-freeze_YYYY-MM-DD.txt
   ```

2. Confirmar que `requirements.txt` + instalação limpa reproduzem o ambiente.

### D. Backup da base SQLite

**Método simples (servidor parado):**

```powershell
Copy-Item backend\ouviescrevi.db backup\ouviescrevi_2026-06-22.db
```

**Com servidor a correr (SQLite online backup):**

```powershell
cd backend
.\.venv\Scripts\python.exe -c "import sqlite3; s=sqlite3.connect('ouviescrevi.db'); d=sqlite3.connect('../backup/ouviescrevi_backup.db'); s.backup(d); d.close(); s.close(); print('ok')"
```

### E. Antes de deploy em produção (quando autorizado)

Checklist:

- [ ] Backup `ouviescrevi.db` em produção
- [ ] Export env vars do servidor atual
- [ ] Snapshot ou tag Git do commit a deployar
- [ ] Plano de rollback (versão anterior + env antigo)
- [ ] Janela para testar `/debug` e um fluxo crítico após deploy

---

## O que NÃO incluir em backups partilhados

- Ficheiros `.env` em cloud pública ou email
- Chaves OpenAI em screenshots ou chat
- `ouviescrevi.db` com dados pessoais em repositórios públicos
- Vídeos de utilizadores sem consentimento documentado (RGPD)

---

## Estrutura de pasta de backup sugerida

```text
backup/                          # fora do repo ou em repo privado separado
├── env/
│   └── notas_sem_valores.txt    # nomes das variáveis, não os valores
├── db/
│   └── ouviescrevi_2026-06-22.db
├── logs/
│   └── logs_2026-06-22/
├── static_videos/
│   └── videos_2026-06-22/
├── pip-freeze_2026-06-22.txt
└── hosting/
    └── notas_dns_e_paineis.txt
```

Os valores reais de `.env` devem ficar num **gestor de passwords**, não nesta árvore.

---

## Restauro rápido

| Cenário | Ação |
|---------|------|
| `.env` apagado por engano | Restaurar do gestor de passwords |
| `ouviescrevi.db` corrompido | Parar servidor; substituir por cópia de backup |
| Deploy falhou | Reverter para tag/commit anterior; restaurar env vars antigas |
| `pip install` partiu ambiente | Apagar `.venv`; recriar venv; `pip install -r requirements.txt` |
| Token novo não funciona | Reverter temporariamente token antigo **só em dev**; nunca reexpor no frontend |

---

## Retenção recomendada

| Tipo | Sugestão |
|------|----------|
| Backups DB | 30 dias rolling + 1 mensal |
| Logs | 90 dias ou conforme RGPD |
| Vídeos em `static/videos/` | 7–30 dias com job de limpeza (ainda não implementado) |
| Cópias de `.env` | Só em password manager; sem expiração até rotação |

---

## Referências

- `PROJECT_AUDIT.md` — riscos e dados expostos
- `NEXT_STEPS.md` — ordem de trabalho após backups
- `CHANGELOG.md` — registo de quando backups/restores foram feitos
