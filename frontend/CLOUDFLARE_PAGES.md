# Deploy do site — Cloudflare Pages

Guia para publicar a pasta `frontend/` em **https://ouviescrevi.pt** (grátis).

A API continua separada em **https://api.ouviescrevi.pt** (ainda por configurar noutro servidor).

---

## Antes de começar

1. **Logos** — copia para `frontend/logos/`:
   - `ouviescreviicon.png`
   - `ouviescrevimainlogo.png`
2. **Repositório Git** (recomendado) — GitHub ou GitLab com este projeto.
3. Conta grátis em [dash.cloudflare.com](https://dash.cloudflare.com/sign-up).

---

## Parte A — DNS no Cloudflare (domínio)

1. **Add a site** → `ouviescrevi.pt`
2. Escolhe o plano **Free**
3. O Cloudflare mostra **2 nameservers** (ex. `ada.ns.cloudflare.com`)
4. No painel do registrador (onde ativaste o domínio) → **Nameservers** → cola os do Cloudflare
5. Aguarda propagação (minutos a algumas horas)

Quando o domínio estiver **Active** no Cloudflare, segue para a Parte B.

---

## Parte B — Criar o projeto Pages

### Opção 1 — Git (recomendada)

1. [Workers & Pages](https://dash.cloudflare.com/) → **Create** → **Pages** → **Connect to Git**
2. Autoriza GitHub/GitLab e escolhe o repositório `transcreveja`
3. **Build settings:**

   | Campo | Valor |
   |-------|--------|
   | Production branch | `main` (ou a tua branch) |
   | Framework preset | **None** |
   | Build command | *(vazio)* |
   | Build output directory | `frontend` |

   O ficheiro `wrangler.toml` na raiz do repo confirma `pages_build_output_dir = "frontend"`.

4. **Save and Deploy**

Cada push à branch de produção atualiza o site automaticamente.

### Opção 2 — Upload manual (sem Git)

Com [Node.js](https://nodejs.org/) instalado, na raiz do projeto:

```powershell
cd C:\Users\joao_\transcreveja
npx wrangler pages deploy frontend --project-name=ouviescrevi
```

Na primeira vez, o Wrangler pede login na conta Cloudflare. Repete o comando sempre que quiseres publicar alterações.

---

## Parte C — Domínio personalizado

1. No projeto Pages → **Custom domains** → **Set up a custom domain**
2. Adiciona:
   - `ouviescrevi.pt`
   - `www.ouviescrevi.pt`
3. O Cloudflare cria os registos DNS (CNAME/flatten) e o **certificado SSL** automaticamente

**Redirecionar www → raiz (opcional):**

- **Rules** → **Redirect Rules** → Create rule  
- If: hostname equals `www.ouviescrevi.pt`  
- Then: Dynamic redirect → `https://ouviescrevi.pt${http.request.uri.path}` (301)

---

## O que já está configurado no repo

| Ficheiro | Função |
|----------|--------|
| `wrangler.toml` | Diretório de saída `frontend` |
| `frontend/_redirects` | `index2`/`admin` → `index`; `/api/*` → API |
| `frontend/_headers` | Cabeçalhos de segurança + cache CSS/JS |
| `frontend/404.html` | Página de erro personalizada |
| `frontend/robots.txt` + `sitemap.xml` | SEO |

O JavaScript (`ouviescrevi-api.js`) usa automaticamente `https://api.ouviescrevi.pt` quando o site não está em `localhost`.

---

## Checklist após o deploy

```
[ ] https://ouviescrevi.pt abre a homepage
[ ] https://ouviescrevi.pt/ajuda.html e outras páginas carregam
[ ] Logo e favicon visíveis (pasta logos/)
[ ] Menu ☰ funciona no telemóvel
[ ] Consola do browser sem erros 404 em css/js/header.html
```

**Transcrição** só funciona quando a API estiver no ar. Até lá, o site mostra erro ao carregar a configuração — é normal.

---

## API + CORS (quando publicares o backend)

No `.env` de produção da API:

```env
ALLOWED_ORIGINS=https://ouviescrevi.pt,https://www.ouviescrevi.pt
PUBLIC_API_BASE=https://api.ouviescrevi.pt
APP_ENV=production
ENABLE_DEBUG_ENDPOINTS=false
```

Para testar numa pré-visualização Pages (`https://ouviescrevi.pages.dev`), acrescenta esse URL a `ALLOWED_ORIGINS`.

No Cloudflare DNS (zona do domínio):

| Tipo | Nome | Conteúdo |
|------|------|----------|
| A ou CNAME | `api` | IP ou host do servidor da API |

---

## Resolução de problemas

| Problema | Solução |
|----------|---------|
| Site antigo / DNS não resolve | Nameservers ainda no registrador antigo — confirma no Cloudflare |
| 404 em `/` | Confirma que `frontend/index.html` existe no deploy |
| Logo em falta | Adiciona PNGs em `frontend/logos/` e volta a fazer deploy |
| "Configuração da API indisponível" | API ainda não deployada ou CORS sem o domínio do site |
| CSS desatualizado | Hard refresh (Ctrl+F5) ou espera pelo cache |

---

## Ordem sugerida

1. Nameservers → Cloudflare  
2. Pages deploy (Git ou Wrangler)  
3. Domínio `ouviescrevi.pt` no projeto  
4. Verificar site estático  
5. Deploy da API em `api.ouviescrevi.pt`  
6. Testar transcrição em produção
