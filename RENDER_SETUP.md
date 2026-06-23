# Render — API em api.ouviescrevi.pt

Não é possível configurar o Render remotamente sem acesso à tua conta. Este guia reduz o trabalho ao **mínimo** (copiar valores).

---

## Passo 1 — Gerar tokens no teu PC

```powershell
cd C:\Users\joao_\transcreveja\backend
python generate_production_secrets.py
```

Copia os 3 valores gerados. A `OPENAI_API_KEY` vens tu (dashboard OpenAI).

---

## Passo 2 — Turso (base de dados grátis e persistente)

No plano **Free** do Render, o disco do contentor é **efémero**. Usamos **Turso** (SQLite na cloud, plano grátis) para guardar transcrições, visitas e CMS.

### 2a — Criar conta e base

1. Regista-te em [turso.tech](https://turso.tech) (grátis)
2. Instala a CLI (opcional mas útil): [docs.turso.tech/cli](https://docs.turso.tech/cli)
3. Cria a base:

```bash
turso auth login
turso db create ouviescrevi --region fra
turso db show ouviescrevi --url
turso db tokens create ouviescrevi
```

Guarda:
- **URL** → `libsql://ouviescrevi-xxxx.turso.io`
- **Token** → string longa (só aparece uma vez)

### 2b — Variáveis no Render

1. [dashboard.render.com](https://dashboard.render.com) → serviço **api-ouviescrevi**
2. Menu **Environment** → **Add Environment Variable**:

| Key | Value |
|-----|--------|
| `OPENAI_API_KEY` | `sk-...` |
| `ADMIN_TOKEN` | *(do script)* |
| `API_TOKEN` | *(do script)* |
| `BACKOFFICE_PASSWORD` | *(do script)* |
| `TURSO_DATABASE_URL` | `libsql://ouviescrevi-xxxx.turso.io` |
| `TURSO_AUTH_TOKEN` | *(token do passo 2a)* |
| `APP_ENV` | `production` |
| `ENABLE_DEBUG_ENDPOINTS` | `false` |
| `ALLOWED_ORIGINS` | `https://ouviescrevi.pt,https://www.ouviescrevi.pt` |
| `PUBLIC_API_BASE` | `https://api.ouviescrevi.pt` |
| `MAX_FILE_SIZE_MB` | `100` |

3. **Save Changes** (redeploy automático)

O ficheiro `render.yaml` na raiz define plano **Free** + placeholders para Turso.

### 2c — Verificar

No backoffice → **Sistema**:
- `database_backend` = `turso`
- `database_path` = `libsql://...`

No primeiro deploy, as tabelas são criadas automaticamente.

### 2d — Migrar dados antigos (opcional)

Se tiveres cópia local de `ouviescrevi.db`:

```bash
sqlite3 ouviescrevi.db .dump > dump.sql
turso db shell ouviescrevi < dump.sql
```

Ou exporta **Backup JSON** do backoffice e reimporta manualmente.

**Nota:** Dados perdidos no Render Free antes desta configuração **não se recuperam** sem backup.

### Desenvolvimento local

Sem `TURSO_*`, usa ficheiro local `ouviescrevi.db`. Para testar com Turso localmente, copia as mesmas variáveis para `backend/.env`.

---

## Passo 3 — Settings (confirmar)

**Settings** → **Build & Deploy**:

| Campo | Valor |
|-------|--------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

---

## Passo 4 — Verificar

**Logs** → deve aparecer:
```
Uvicorn running on http://0.0.0.0:10000
```
sem `Falta BACKOFFICE_PASSWORD` nem aviso de base efémera.

Testa no site: **https://www.ouviescrevi.pt** → transcrição com áudio pequeno.

---

## Erros comuns

| Log / sintoma | Solução |
|---------------|---------|
| `Falta BACKOFFICE_PASSWORD` | Adicionar variável no Environment |
| `GET /api/frontend-config 404` | Deploy antigo — espera redeploy com commit novo |
| `403 Origem não autorizada` | Incluir `https://www.ouviescrevi.pt` em `ALLOWED_ORIGINS` |
| Demora no 1.º pedido | Plano grátis “adormece” ~50 s |
| Estatísticas zeradas após deploy | Falta `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` |
| `database` erro no backoffice | Token expirado ou URL errada — gera novo token na Turso |
| Aviso base efémera nos logs | Configura Turso no Render |

---

## DNS (já deve estar feito)

Cloudflare: `api` CNAME → `api-ouviescrevi.onrender.com` (proxied).
