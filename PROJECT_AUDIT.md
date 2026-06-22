# Auditoria do projeto TranscreveJá / Ouviescrevi

**Data:** 22 de junho de 2026 (atualizado após Sessão 1)  
**Âmbito:** todos os ficheiros próprios do backend e frontend presentes no repositório. Os binários (imagens, vídeos, `.pyc` e executáveis do `venv`) foram inventariados, mas não são código-fonte.  
**Método:** análise estática + validação local do backend (Sessão 1). Não foi alterado código de aplicação e não foram efetuados pedidos à API de produção.

**Documentação relacionada:** `NEXT_STEPS.md`, `BACKUP_NOTES.md`, `CHANGELOG.md`, `PROJECT_BRIEF.md`.

## Resumo executivo

O projeto é uma aplicação web sem framework de frontend, composta por páginas HTML/CSS/JavaScript estáticas e uma API FastAPI monolítica. Implementa transcrição de áudio/vídeo com OpenAI Whisper, legendagem de vídeo com FFmpeg, várias operações de texto com IA, geração de vídeo, conversões locais no browser e um backoffice simples.

No estado atual, **não deve ser relançado publicamente sem correções de segurança**. O token usado para proteger as operações de IA está publicado em vários ficheiros JavaScript, a palavra-passe do backoffice está no próprio HTML, existem endpoints administrativos sem autenticação e há vetores de SSRF. O ambiente virtual incluído no repositório (`backend/venv/`) não é portável; um **venv novo** (`backend/.venv`) com `pip install -r requirements.txt` **arranca e responde** em ambiente local validado (Python 3.14.6).

**Sessão 1 (22 jun 2026):** o backend sobe com Uvicorn, `/debug` e `/api/status` respondem; o pipeline `/transcribe` executa upload → FFmpeg → segmentação; Whisper e GPT falham apenas por falta de chave OpenAI válida no `.env`. A transcrição real com IA **ainda não foi confirmada** — falta `backend/.env` com `OPENAI_API_KEY` real.

Também não é possível testar o conjunto “frontend local + backend local” sem configuração adicional: o frontend chama diretamente `https://api.ouviescrevi.pt`, e abrir os HTML com `file://` quebra o carregamento de `header.html` e `footer.html` em navegadores normais.

## Estrutura do projeto

```text
transcreveja/
├── PROJECT_BRIEF.md          # objetivos e regras de reativação
├── PROJECT_AUDIT.md          # este relatório
├── NEXT_STEPS.md             # plano de trabalho priorizado
├── BACKUP_NOTES.md           # backups e rotação de segredos
├── CHANGELOG.md              # histórico de marcos e alterações
├── README.txt                # instruções antigas e incompletas
├── .gitignore
├── backend/
│   ├── main.py               # API FastAPI, IA, FFmpeg, logs e todas as rotas
│   ├── database.py           # bootstrap SQLite
│   ├── requirements.txt      # dependências Python
│   ├── .venv/                # ambiente local válido (criado na Sessão 1; não versionar)
│   ├── static/               # saída pública de vídeos e SRT gerados
│   ├── venv/                 # ambiente virtual antigo versionado e não portável
│   ├── log_transcricoes.json # artefacto vazio/legado
│   ├── status.json           # artefacto legado; o estado real está em SQLite
│   └── ouviescrevi.log       # ficheiro vazio; o logger atual usa ./logs
└── frontend/
    ├── index.html            # aplicação principal PT
    ├── en/index.html         # aplicação principal EN
    ├── index2.html           # variante antiga/admin duplicada
    ├── admin.html            # variante administrativa da aplicação
    ├── backoffice.html       # estado, contador e estatísticas
    ├── header.html/footer.html e equivalentes EN
    ├── resumo, url-resumo, perguntas, corretor, conversor, gerar-video
    ├── páginas de nicho/ajuda/sugestões
    ├── app.js/style.css      # implementação antiga/mínima
    ├── api/logs.ts           # função Vercel independente da API FastAPI
    ├── logos/, icons/, videos/
    ├── robots.txt
    └── sitemap.xml
```

Não existem `Dockerfile`, `docker-compose`, configuração de CI, testes, `package.json`, configuração Vercel, ficheiro `.env.example`, migrações de base de dados ou manifesto de deployment.

## Como correr o backend localmente

### Pré-requisitos

- Python **3.10 ou superior** (o código usa `str | None` e tipos genéricos modernos); testado com **3.14.6** na Sessão 1; **3.11 ou 3.12** recomendados para maior estabilidade em produção.
- `pip` e acesso à Internet para instalar pacotes.
- Uma chave OpenAI válida em `backend/.env`.
- FFmpeg. O código procura primeiro um binário de sistema e depois o fornecido por `imageio-ffmpeg`. **Não é necessário compilar** o source FFmpeg — o pacote pip inclui binário Windows (`ffmpeg-win-x86_64-v7.1.exe`). O download em `Downloads\ffmpeg-8.1.2\` é código-fonte, não executável.
- Espaço em disco para uploads temporários, áudio convertido e vídeos gerados.

### Passos recomendados no Windows/PowerShell

```powershell
cd backend
py -3.12 -m venv .venv    # ou py -3.14; evitar backend/venv/ versionado
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Criar `backend/.env` (já ignorado pelo Git):

```dotenv
OPENAI_API_KEY=...
ADMIN_TOKEN=um-segredo-longo-e-aleatorio
```

Opcionalmente podem ser definidos `LOG_LEVEL`, `LOG_DIR`, credenciais `SMTP_*`, `VERCEL_LOG_*`, timeouts, limites e nomes dos modelos descritos em `main.py`.

Arrancar **a partir de `backend/`**, porque a base de dados, `static/`, `logs/` e o import `from database import criar_base` dependem do diretório corrente:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verificação básica: `http://127.0.0.1:8000/debug`; documentação automática: `/docs`.

### Limitações encontradas

- Não usar `backend/venv`: o seu `pyvenv.cfg` aponta para `C:\Users\Administrador\...` e falha em máquinas diferentes.
- Usar `backend/.venv` (criado localmente) ou um venv novo — **não commitar** ao Git.
- O README manda inserir a chave em `main.py`, mas o código atual exige `OPENAI_API_KEY` e `ADMIN_TOKEN` no ambiente.
- Executar `uvicorn main:app` na raiz do repositório não encontra `main.py`/`database.py` sem alterar o caminho de importação.
- O arranque cria imediatamente diretórios, base SQLite e logger, e falha se faltarem os dois segredos ou FFmpeg (via sistema ou `imageio-ffmpeg`).

### Validação local (Sessão 1 — 22 jun 2026)

| Teste | Resultado |
|-------|-----------|
| `pip install -r requirements.txt` em `.venv` | OK — inclui `beautifulsoup4`, `gtts`, FastAPI 0.138, OpenAI 2.43 |
| Import `bs4`, `gtts`, `imageio_ffmpeg` | OK |
| `uvicorn main:app` em `127.0.0.1:8000` | OK |
| `GET /debug` | `200` — `{"status":"OK","versao":"1.5"}` |
| `GET /api/status` | `200` — `{"manutencao":false}` |
| `POST /transcribe` (WAV de teste, chave placeholder) | `200` — pipeline FFmpeg OK; Whisper `401` (chave inválida) |
| `POST /summarize` (token correto, chave placeholder) | `500` — OpenAI `401` (esperado) |
| `POST /summarize` (token errado) | `403` — auth por token funciona |
| Transcrição com áudio real + chave OpenAI válida | **Não testado** — falta `backend/.env` |

## Como correr o frontend localmente

O frontend é estático, mas deve ser servido por HTTP; não basta abrir `index.html` diretamente, pois as páginas fazem `fetch()` dos fragmentos de cabeçalho e rodapé.

```powershell
cd frontend
py -m http.server 5500
```

Abrir `http://127.0.0.1:5500/` (PT) ou `http://127.0.0.1:5500/en/` (EN). Alternativas equivalentes como VS Code Live Server ou `npx serve` também servem.

**Atenção:** as páginas têm `https://api.ouviescrevi.pt` fixo no código. Assim, o frontend local continua a usar produção, não `localhost:8000`. Para uma integração inteiramente local seria necessário tornar a base da API configurável ou usar um proxy; essa alteração não foi feita nesta auditoria.

`frontend/api/logs.ts` só pode ser compilado/deployado como função Vercel depois de criar um projeto Node e instalar `@vercel/node`; atualmente não existe `package.json` nem configuração associada.

## Dependências necessárias

### Backend Python

Declaradas: `fastapi`, `uvicorn[standard]`, `gunicorn`, `python-multipart`, `python-dotenv`, `openai`, `requests`, `beautifulsoup4`, `mammoth`, `moviepy==1.0.3`, `imageio>=2.31.1`, `imageio-ffmpeg>=0.4.8`, `gtts` e `jinja2`.

Dependências efetivamente usadas diretamente pelo código: FastAPI/Pydantic/Starlette, Uvicorn para servir, python-multipart, python-dotenv, OpenAI, Requests, Beautiful Soup, gTTS e imageio-ffmpeg. SQLite, SMTP e restante I/O vêm da biblioteca padrão.

`mammoth`, `moviepy`, `imageio` e `jinja2` não são importados pelo backend atual. `gunicorn` destina-se tipicamente ao deployment Unix e não é necessário para desenvolvimento Windows.

### Frontend

- Browser moderno com File API, MediaRecorder, Canvas e Clipboard API.
- Bibliotecas carregadas de CDN, sem integridade SRI: jsPDF 2.5.1, PDF.js 3.4.120 e Mammoth Browser sem versão fixa no URL `unpkg`.
- `@vercel/node` e TypeScript para `api/logs.ts`, embora falte o manifesto Node.
- Ícones sociais remotos do Flaticon; se a CDN falhar, os ícones desaparecem.

## Dependências desatualizadas e reprodutibilidade

O `requirements.txt` só fixa `moviepy==1.0.3`; os restantes pacotes não estão fixos ou têm apenas mínimos. Por isso, uma instalação nova pode obter combinações diferentes das presentes no `venv`, introduzir incompatibilidades sem mudança no Git e não permite afirmar a versão de produção.

O ambiente versionado antigo (`backend/venv/`) contém, entre outras, FastAPI 0.115.12, Uvicorn 0.34.1, OpenAI 1.73.0, Requests 2.32.3, Pillow 10.4.0, Python-dotenv 1.1.0 e MoviePy 1.0.3 — snapshot Python 3.11.0, não portável.

Uma instalação limpa na Sessão 1 (`backend/.venv`, Python 3.14.6) obteve, entre outras: **FastAPI 0.138.0**, **Uvicorn 0.49.0**, **OpenAI 2.43.0**, **Requests 2.34.2**, **Pillow 12.2.0**, **NumPy 2.5.0**, **Starlette 1.3.1**, **python-dotenv 1.2.2**, MoviePy 1.0.3, beautifulsoup4 4.15.0, gTTS 2.5.4. Isto confirma que `requirements.txt` sem pins pode instalar versões muito diferentes entre ambientes.

Há pelo menos estes casos claros:

- **MoviePy 1.0.3:** ramo antigo; a série 2.x introduziu mudanças incompatíveis. O código não usa MoviePy, portanto deve primeiro confirmar-se se pode ser removido.
- **OpenAI SDK:** salto 1.73.0 (venv antigo) → 2.43.0 (instalação nova) — o código atual **funcionou** na Sessão 1 com 2.43.0 para imports e chamadas (com erro 401 por chave inválida); testar com chave real antes de produção.
- **FastAPI 0.115.12 → 0.138.0** e **Uvicorn 0.34.1 → 0.49.0:** validados no arranque local; fixar versões após testes completos.
- **Pillow 10.4.0:** dependência transitiva antiga, mesmo não aparecendo no manifesto.
- **PDF.js 3.4.120:** versão antiga carregada no frontend.
- **Mammoth Browser via `unpkg.com/mammoth/mammoth.browser.min.js`:** URL flutuante, podendo mudar sem controlo do projeto.
- **pip 22.3 e setuptools 65.5.0:** ferramentas antigas dentro do `venv` comprometido.

Antes de atualizar em produção, deve gerar-se um ambiente limpo, fixar versões (`pip freeze` ou constraints) e testar. Atualizações de MoviePy 1.x para 2.x exigem revisão de API. O salto OpenAI 1.x → 2.x **já ocorre** em instalações novas a partir de `requirements.txt` sem pins.

## Endpoints existentes

| Método | Caminho | Função | Proteção atual |
|---|---|---|---|
| GET | `/` | Lista rotas e métodos | Nenhuma |
| GET | `/rotas` | Lista caminhos | Nenhuma |
| GET | `/debug` | Estado/versão 1.5 | Nenhuma |
| POST | `/transcribe` | Upload, FFmpeg, segmentação e Whisper | Nenhuma |
| POST | `/video-subs` | Transcreve, cria SRT e queima legendas | Nenhuma; `style` é recebido mas ignorado |
| POST | `/summarize` | Resumo normal ou minuta, PT/EN | Token no corpo |
| POST | `/translate` | Tradução para seis idiomas | Token no corpo |
| POST | `/classify` | Classificação do conteúdo | Token no corpo |
| POST | `/correct` | Correção ortográfica/gramatical | Token no corpo |
| POST | `/generate-email` | Email profissional por tom | Token no corpo |
| POST | `/generate-questions` | Perguntas de escolha múltipla | Token no corpo |
| POST | `/summarize-url` | Extrai parágrafos de URL e resume | Token no corpo |
| POST | `/notify-whatsapp-share` | Notifica partilha por email | Token no corpo |
| POST | `/notify/whatsapp` | Alias do anterior | Token no corpo |
| POST | `/generate-video` | gTTS + imagem remota + FFmpeg | Token no corpo |
| GET | `/api/status` | Consulta manutenção | Nenhuma |
| POST | `/api/status` | Altera manutenção | **Nenhuma** |
| GET | `/transcricoes-hoje` | Contador diário | Nenhuma |
| GET | `/api/logs` | Lista nomes/data de todos os uploads | **Nenhuma** |
| GET | `/test-email` | Dispara email SMTP | **Nenhuma** |
| GET | `/static/*` | Vídeos e SRT gerados | Público |

FastAPI também expõe por defeito `/docs`, `/redoc` e `/openapi.json`.

A função Vercel `frontend/api/logs.ts` implementa separadamente `GET /api/logs` e `POST /api/logs`, lendo/escrevendo `/tmp/ouviescrevi_frontend.log`. Não usa SQLite e o seu formato de resposta não coincide com o esperado pelo backoffice.

## Funcionalidades existentes

- Upload e transcrição de áudio/vídeo, com normalização FFmpeg, divisão em segmentos, retries, timestamps e resultado parcial em timeout.
- Gravação pelo microfone no browser e posterior transcrição.
- Geração e download de SRT; produção de vídeo MP4 com legendas embutidas e fallback para apenas SRT.
- Resumo, minuta, correção, classificação, tradução, geração de email e perguntas de estudo por IA.
- Resumo de páginas web a partir de URL.
- Geração de vídeo vertical/estático a partir de texto, gTTS e imagem remota.
- Exportação no frontend para TXT, JSON, SRT, PDF e, em alguns fluxos, documento Word simples.
- Extração local de texto de PDF/DOCX e conversões Word→PDF, PDF→texto, imagem→PDF e PDF→ficheiro `.doc` textual.
- Aplicação de etiquetas de locutor heurísticas, cópia e partilha por WhatsApp/email.
- Interface principal em português e inglês, fragmentos partilhados de navegação e páginas informativas por nicho.
- Formulário de sugestões via FormSubmit.
- Backoffice para manutenção, contagem e alegadas estatísticas.
- SQLite para nomes/data das transcrições e estado de manutenção; logs rotativos e envio opcional de warnings para Vercel.

O brief pede exportação DOCX, mas o projeto não cria DOCX real; grava texto com MIME/extensão `.doc`. Não há exportação backend, contas de utilizador, pagamentos, histórico por utilizador ou fila de trabalhos.

## Problemas encontrados

### Execução e arquitetura

1. O `venv` versionado é grande, não portável e está quebrado; `.gitignore` não ignora `venv/`/`.venv/`.
2. O README está desatualizado quanto a segredos, diretório de arranque, frontend e dependências reais.
3. Caminhos de DB, logs e estáticos dependem do diretório corrente.
4. Toda a API está concentrada num `main.py` com mais de mil linhas; configuração, rotas, serviços e persistência estão acoplados.
5. Funções síncronas pesadas (`requests`, SDK OpenAI síncrono, FFmpeg, gTTS, SQLite e SMTP) são chamadas dentro de endpoints `async`, bloqueando o event loop.
6. Não há testes, linting, type checking, CI, health check operacional, métricas ou configuração de deployment reproduzível.
7. `criar_base()` é executado ao importar `database.py` e novamente em `main.py`.
8. O limite do upload só é aplicado depois de escrever o ficheiro completo no disco.
9. Os vídeos/SRT em `static/videos` nunca expiram nem são removidos; o disco crescerá continuamente.
10. `generate-video` pode deixar temporários em caso de erro, pois a limpeza não está num `finally`.
11. O parâmetro `style` de `/video-subs` é ignorado; o frontend apresenta controlo visual que não altera o vídeo final.
12. Respostas de erro são inconsistentes: algumas rotas devolvem HTTP 200 com `warning`; outras devolvem 4xx/5xx e expõem `str(e)`.
13. SQLite usa ligações por pedido, caminhos relativos e não tem migrações, índices, WAL ou política de retenção.
14. Nomes de modelos OpenAI e limites estão configuráveis, mas não há validação de configuração nem controlo de consumo/custo.

### Frontend

1. A API de produção está hardcoded em muitas páginas; existem várias implementações duplicadas (`index.html`, `index2.html`, `admin.html`, páginas PT/EN), propensas a divergência.
2. `README.txt` recomenda `file://`, incompatível com os `fetch()` de header/footer.
3. O backoffice faz `fetch('/api/logs')`, que no frontend pode atingir a função Vercel em vez do FastAPI. A função Vercel devolve `{logs: [...]}`, mas o código espera um array e chama `data.reverse()`: a tabela falha.
4. Mesmo quando recebe dados FastAPI, o backoffice concatena `ficheiro` em `innerHTML`, permitindo XSS persistente através de um nome de ficheiro malicioso.
5. `app.js` também injeta a transcrição externa em `innerHTML`, outro vetor XSS caso essa implementação seja usada.
6. Dependências CDN não têm SRI nem estratégia local; Mammoth nem sequer tem versão fixa.
7. PDF→texto processa páginas em paralelo e concatena à medida que terminam, podendo baralhar a ordem. A conversão PDF→Word não gera um documento Word real.
8. Há funcionalidades anunciadas mas não implementadas como descritas, por exemplo OCR de imagem e DOCX real.
9. Várias funções dependem do objeto global implícito `event`, comportamento frágil entre browsers.
10. Links com `target="_blank"` não usam consistentemente `rel="noopener noreferrer"`.
11. `style.css` e `app.js` parecem restos de uma versão antiga; a maior parte do CSS/JS está embutida nos HTML.
12. Não existe build, minificação, cache busting, testes de browser, Content Security Policy ou gestão central de configuração.

### Validação efetuada

- Todos os endpoints e chamadas `fetch` foram cruzados estaticamente.
- **Sessão 1:** ambiente `backend/.venv` criado; dependências instaladas; servidor Uvicorn testado; `/debug`, `/api/status`, `/transcribe` (pipeline), auth em `/summarize` validados.
- **Não testado:** transcrição/resumo com chave OpenAI real; SMTP; `/video-subs` com vídeo real; frontend integrado com backend local; Vercel; FormSubmit; domínios de produção.
- Ver detalhes em `CHANGELOG.md` (secção Unreleased) e passos seguintes em `NEXT_STEPS.md`.

## Riscos de segurança

### Críticos — estado após Sessão 2 (22 jun 2026)

| Risco | Estado |
|-------|--------|
| Token exposto no frontend | **Mitigado no código** — removido de todos os HTML/JS; **rodar** o valor antigo em produção |
| Backoffice com password no JS | **Corrigido** — `POST /api/admin/login` + `BACKOFFICE_PASSWORD` no servidor |
| Admin sem auth no servidor | **Corrigido** — `POST /api/status`, `GET /api/logs`, `GET /transcricoes-hoje` exigem `ADMIN_TOKEN` |
| SSRF em summarize-url / generate-video | **Corrigido** — validação de URL pública em `security.py` |

### Ainda em aberto (antes de produção)

1. **Deploy coordenado** — backend com novos tokens + frontend sem segredos; o token antigo deve ser considerado comprometido.
2. **`GET /api/status`** continua público (só leitura de manutenção) — aceitável para páginas de ajuda; monitorizar abuso.
3. Uploads sem validação MIME profunda, ficheiros estáticos públicos, XSS residual em CDN/`innerHTML` noutras páginas, função Vercel de logs — ver secções anteriores.

### Histórico (pré-Sessão 2)

1. **Token de backend exposto:** estava em texto simples em várias páginas PT/EN.
2. **Backoffice sem autenticação real:** password no JavaScript e flag `localStorage`.
3. **Administração sem autenticação no servidor:** `/api/status`, `/api/logs`, `/test-email` públicos.
4. **SSRF:** `/summarize-url` e `/generate-video` aceitavam URLs arbitrárias.

### Elevados — estado parcial

5. `/transcribe` e `/video-subs` — **agora** exigem `API_TOKEN` + rate limiting.
6. CORS — **restrito** a `ALLOWED_ORIGINS`.
7. Uploads sem validação MIME profunda; FFmpeg como superfície de ataque — **ainda por endurecer**.
8. Dados sensíveis em SQLite; vídeos/legendas públicos sem expiração — **ainda por endurecer** (`/api/logs` agora exige admin).
9. XSS no backoffice — **mitigado** nos logs (`textContent`); rever outras páginas com `innerHTML`.
10. Limites de comprimento de texto/URL/imagem — **parcial** (rate limits); quotas por utilizador ainda não existem.

### Médios

11. Exceções internas devolvidas ao cliente.
12. Logs com IP/UA/referer — implicações RGPD.
13. Função Vercel de logs pública (se ainda em uso).
14. CDN sem SRI/CSP.
15. Token partilhado por todos os utilizadores via `/api/frontend-config` — melhor que expor no HTML, mas sem identidade individual.

## Melhorias prioritárias

### P0 — estado

| Item | Estado |
|------|--------|
| Rodar segredos comprometidos | **Pendente (deploy)** |
| Auth no servidor | **Feito** |
| Rate limiting em rotas caras | **Feito** |
| SSRF | **Feito** |
| CORS restrito | **Feito** |
| Ficheiros privados / RGPD | **Pendente** |
| XSS backoffice | **Mitigado** nos logs |

### P1 — conseguir instalar, executar e validar

8. Remover `backend/venv` do controlo de versão, ignorar `.venv/` e criar um ambiente novo.
9. Fixar versões num lock/constraints file, remover dependências não usadas e fazer atualização incremental com testes, sobretudo SDK OpenAI, FastAPI/Uvicorn, MoviePy/Pillow e bibliotecas CDN.
10. Criar `.env.example` sem segredos e atualizar o README com os comandos corretos, FFmpeg, diretórios de trabalho e configuração do frontend.
11. Centralizar `API_BASE` por ambiente e permitir frontend local contra `http://127.0.0.1:8000` sem editar dezenas de ficheiros.
12. Adicionar testes mínimos: arranque/configuração, status autenticado, uploads inválidos/limites, mocks OpenAI/HTTP/SMTP, SSRF, geração/limpeza de ficheiros e fluxos browser PT/EN.
13. Corrigir o contrato `/api/logs`, escolher FastAPI **ou** Vercel como fonte, e eliminar a implementação contraditória.

### P2 — estabilidade e manutenção

14. Separar `main.py` em configuração, routers, serviços de OpenAI/FFmpeg, persistência e notificações; executar trabalho bloqueante em thread pool ou, preferencialmente, numa fila de jobs.
15. Usar caminhos baseados em `Path(__file__)`, migrações e uma estratégia de base de dados adequada à concorrência/deployment.
16. Normalizar erros/status HTTP, não expor exceções, adicionar request IDs nas respostas e health/readiness checks sem revelar rotas internas.
17. Consolidar as variantes duplicadas do frontend, extrair CSS/JS comum e adicionar CSP, SRI ou dependências vendorizadas/fixas.
18. Corrigir a ordem da extração PDF, remover dependência do `event` global, implementar DOCX/OCR reais ou ajustar a comunicação para não prometer o que não existe.
19. Definir observabilidade e privacidade: métricas de duração/erros/custos sem conteúdo, rotação/retensão de logs e documentação de tratamento de dados.

## Conclusão

A base funcional existe e cobre grande parte da visão do brief. A **Sessão 1 confirmou** que o backend arranca com um venv novo e que o pipeline de transcrição (upload → FFmpeg → Whisper) está operacional até à chamada OpenAI — falta apenas configurar `backend/.env` com chave válida para fechar a validação de IA.

O projeto passou da fase de **protótipo público inseguro** para **código com controlos P0 implementados** (auth, CORS, SSRF, rate limits, frontend sem segredos). Falta configurar `.env`, rodar tokens antigos e fazer deploy coordenado antes de relançar — ver `NEXT_STEPS.md` e `BACKUP_NOTES.md`.
