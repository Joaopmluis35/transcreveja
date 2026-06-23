# Render — API em api.ouviescrevi.pt

Não é possível configurar o Render remotamente sem acesso à tua conta. Este guia reduz o trabalho ao **mínimo** (copiar 4 valores).

---

## Passo 1 — Gerar tokens no teu PC

```powershell
cd C:\Users\joao_\transcreveja\backend
python generate_production_secrets.py
```

Copia os 3 valores gerados. A `OPENAI_API_KEY` vens tu (dashboard OpenAI).

---

## Passo 2 — Environment no Render

1. [dashboard.render.com](https://dashboard.render.com) → serviço **api-ouviescrevi**
2. Menu **Environment**
3. **Add Environment Variable** — adiciona:

| Key | Value |
|-----|--------|
| `OPENAI_API_KEY` | `sk-...` |
| `ADMIN_TOKEN` | *(do script)* |
| `API_TOKEN` | *(do script)* |
| `BACKOFFICE_PASSWORD` | *(do script)* |
| `APP_ENV` | `production` |
| `ENABLE_DEBUG_ENDPOINTS` | `false` |
| `ALLOWED_ORIGINS` | `https://ouviescrevi.pt,https://www.ouviescrevi.pt` |
| `PUBLIC_API_BASE` | `https://api.ouviescrevi.pt` |
| `MAX_FILE_SIZE_MB` | `100` |

4. **Save Changes** (redeploy automático)

O ficheiro `render.yaml` na raiz do repo define o resto (build, start, root `backend/`).

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
sem `Falta BACKOFFICE_PASSWORD`.

Testa no site: **https://www.ouviescrevi.pt** → transcrição com áudio pequeno.

---

## Erros comuns

| Log | Solução |
|-----|---------|
| `Falta BACKOFFICE_PASSWORD` | Adicionar variável no Environment |
| `GET /api/frontend-config 404` | Deploy antigo ainda ativo — espera redeploy com commit novo |
| `403 Origem não autorizada` | Incluir `https://www.ouviescrevi.pt` em `ALLOWED_ORIGINS` |
| Demora no 1.º pedido | Plano grátis “adormece” ~50 s |

---

## DNS (já deve estar feito)

Cloudflare: `api` CNAME → `api-ouviescrevi.onrender.com` (proxied).
