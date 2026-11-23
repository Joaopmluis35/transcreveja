# Passagem para o Cursor e o Claude Code — Ouviescrevi / TranscreveJá (26/10 a 21/11/2026)

Há **dois projetos**: o Ouviescrevi (este repositório, `c:\Users\joao_\transcreveja`, **público**) e o simulador
online (outra pasta e repositório **privado** — ver secção 4). Não os misturar e não escrever nada do simulador aqui.
Está tudo publicado no `main` (GitHub Joaopmluis35/transcreveja). **Não refaças nada disto.**
O Cursor e o Claude Code podem trabalhar neste repositório: antes de mexer, `git pull`, `git status` e
`git log --oneline -5`; commit só dos ficheiros que alteraste.

**Prompt para colar no Cursor:**

> Estou a trabalhar no Ouviescrevi (`c:\Users\joao_\transcreveja`, repositório público). Faz `git pull` e lê o `CURSOR_HANDOFF.md` todo. O Claude Code também pode ter feito commits: vê `git log --oneline -5` e não refaças o que já lá está. Segue as regras da secção 3 (alterações pequenas e explicadas antes; publicar só com o meu pedido; subir o `?v=` em todas as páginas que usam um CSS/JS alterado). Testes: `cd backend; py -m pytest -q` e `py scripts/frontend_smoke_test.py`. Nada do simulador de apostas neste repositório. Diz-me o estado (`git status`, testes) antes de propor alterações.

**Prompt para colar no Claude Code:**

> Trabalha no Ouviescrevi em `C:\Users\joao_\transcreveja` (repositório público; o simulador de apostas é outro projecto, privado — não o mencionar nem copiar nada dele para aqui). Faz `git pull`, lê o `CURSOR_HANDOFF.md` e vê `git log --oneline -5` (o Cursor também faz commits). Regras da secção 3; português de Portugal; testes `cd backend; py -m pytest -q` (Python do sistema) e `py scripts/frontend_smoke_test.py`; commit só dos teus ficheiros e só publicar (push) quando eu pedir. Diz-me o estado antes de propor alterações.
---

## 1. Infraestrutura (confirmado em produção)

- **Frontend:** Vercel. Raiz do projeto Vercel = raiz do repositório, pasta publicada = `frontend/`.
  As regras (cabeçalhos de segurança, redirects, rewrite `/s/:id`) estão em **`/vercel.json` na raiz**.
  Não há Cloudflare Pages (`_headers`, `_redirects`, `wrangler.toml` foram apagados).
- **Domínio canónico:** https://www.ouviescrevi.pt (Cloudflare à frente da Vercel).
- **API:** Render, plano free, https://api.ouviescrevi.pt, base de dados Turso. autoDeploy no push para `main`.
  **O servidor corre nos EUA (Ashburn, Virgínia, rede Amazon)**, não em Frankfurt como diz o `render.yaml`.
- **Testes:** `cd backend; py -m pytest -q` → 141 testes (usar o Python do sistema; o `backend/venv` antigo está partido;
  `backend/.venv` tem uvicorn para correr a API em local). Frontend: `py scripts/frontend_smoke_test.py`.
- **Cache:** CSS/JS com `?v=N` nos HTML. Ao mudar um ficheiro, sobe o `?v=` em **todas** as páginas que o usam.
  Versões atuais: `ouviescrevi.css?v=48` (importa `ouviescrevi-pro.css?v=8` e `theme-dark.css?v=5`),
  `index-home.css?v=60`, `ouviescrevi-ui.js?v=31` (cabeçalho/rodapé `LAYOUT_V = "14"`), `theme-ui.js?v=2`, `news-ticker.js?v=4`, `quiz-game.js?v=6`, `quiz-game.css?v=4`,
  `aula-pronta-ui.js?v=7`, `aula-pronta.css?v=6`, `share-ui.js?v=29` (na `index.html` e nas `en|es|fr|de/index.html`).
- Na máquina do dono há outra app na porta **8000** (“AION”). Para testar a API em local usa outra porta (ex. 8010)
  e no browser força `OuviescreviAPI.getBase = () => "http://127.0.0.1:8010"`.

## 2. Commits desta sessão (mais antigo → mais recente)

**Segurança**
- `a2459df` API_TOKEN nunca cai para ADMIN_TOKEN (deriva por HMAC); `secret_equals()` (compare_digest);
  `client_ip()` lê X-Forwarded-For da direita saltando IPs Cloudflare/privados; login admin com rate limit; rota duplicada removida.
- `8b305f3` SSRF: `safe_http_get` segue redirecionamentos à mão (máx. 5) e só aceita IPs globais.

**Limpeza e desempenho**
- `20beb24` Logos de ~3 MB para ~225 KB; manifest com ícones 192/512 reais.
- `8451ecc` venv, `__pycache__`, logs e ficheiros de estado deixaram de estar no git.

**Transcrição**
- `64f62db` Site PT + seletor “automático”: se o Whisper deteta outra língua, repete o pedaço com `language="pt"` e fica com a
  versão de maior `avg_logprob`; áudio mesmo noutra língua é traduzido (GPT, mantém `[mm:ss]`) com “Ver original”.
  Confirmado com áudio real. Testes em `backend/tests/test_pt_language.py`.

**Site**
- `e932833` Faixa de novidades PT com datas por item.
- `1d562b6` Blog com cabeçalho/rodapé/estilos; instrução de upload em telemóvel; `vercel.json` (segurança, redirects,
  `/s/:id` → API); IA em português de Portugal (`ai_output_language`), `/summarize` com os 4 estilos e a língua da página.
- `d33ea20` `vercel.json` movido para a raiz (em `frontend/` era ignorado).
- `a46bcbc` Cartão de progresso da transcrição compacto e sticky.
- `0653b44` Submenus do cabeçalho (tirado `overflow:hidden` da `.oe-pro-nav`); botão do tema 44 px; ES/FR/DE index tinham uma
  tag `<script>` dentro de um template JS que partia o upload; strings partidas no Podcast & YouTube e no conversor;
  removidas 113 chamadas inline `OuviescreviUI.loadHeader/loadFooter`.
- `99ba8ec` + `0ad6785` Links de privacidade nas Ajudas EN/ES/FR/DE; sitemap com 132 URLs e hreflang; H1 na EN;
  jsPDF só ao exportar; aviso de cookies compacto; ficheiros do Cloudflare removidos.
- `b6737ed` Ações de IA da página inicial repetem o pedido (3 s e 8 s) se a ligação falhar ou der 502/503/504
  (o upload de ficheiros não repete, para não pagar duas vezes).
- `c7c3fec` **Telemóvel:** o menu fechado estava fora do ecrã com `translateX(100%)` e alargava as páginas (390 px → 710 px);
  agora é escondido com `clip-path` + `visibility`. Itens do menu deixaram de ficar colados ao fundo.

- `56a201b` (09/11) **Loading em todos os botões de «Ações com IA»** (`frontend/index.html`): Resumo, Classificar,
  Corrigir, Minuta, Email, Traduzir, WhatsApp, Exportar, Partilhar link e os 6 que abrem outra página (Aula Pronta,
  Aula completa, Capítulos, Podcast/YouTube, Flashcards, YouTube). Um só helper `comLoading(btn, texto, fn)` que usa
  `OuviescreviUI.setButtonLoading` (spinner, desativado, «A gerar…»): ignora o 2.º clique e limpa sempre no `finally`
  (também em erro). Os 6 de navegação usam `abrirFerramenta(btn, chave, pagina)` («A abrir…»; um `pageshow` limpa ao
  voltar atrás). O Traduzir mostra o loading no botão da barra lateral (`#btnTraduzir`), porque o modal fecha.
  «Partilhar link» em `js/share-ui.js` («A criar link…»).
- (11/11, ainda não publicado) **O mesmo loading nas páginas EN/ES/FR/DE** (`frontend/{en,es,fr,de}/index.html`), na língua
  da página. Helper partilhado `OuviescreviUI.wireActionLoading({ função: { label, nav, target } })` em
  `js/ouviescrevi-ui.js` (`?v=30` em todas as páginas): envolve as funções globais dos botões (loading, 2.º clique
  ignorado, limpa no fim também em erro; `nav` = abre outra página; `target` = Traduzir mostra no botão da barra e
  Exportar no `#exportBtn`). Cada página chama-o num `<script>` no fim. **Corrigido também:** nas ES/FR/DE, Classificar,
  Corrigir, Minuta, Email e Traduzir **não funcionavam** — liam `event.target` depois de `await ensureApi()` (o `event`
  já não existe) e rebentavam com TypeError. O texto de loading estava em português nessas páginas.

**Aula Pronta e jogo de quiz**
- `3b266c1` Jogo animado (`js/quiz-game.js`, `css/quiz-game.css`): 4 cartões coloridos, 20 s, pontos com bónus de rapidez e
  sequência, confetti, sons com mute, explicações, ecrã final com estrelas; PT/EN/ES/FR/DE.
- `86e3686` Partilha por link curto: tabela `shared_quizzes`, `POST /api/share/quiz`, `GET /api/share/quiz/{id}`,
  página pública `jogo.html?id=…` (noindex).
- `21e0bb8` Caixa de texto da Aula Pronta maior, cresce com o texto, contador de palavras.
- `c7c3fec` Nome do jogador antes de começar; **ranking**: tabela `quiz_scores`, `GET/POST /api/share/quiz/{id}/scores`
  (melhor resultado por nome, posição, top 10, recusa pontuações impossíveis); ranking local para jogos não partilhados.
- `750d4ac` “🏆 Ver ranking” depois de partilhar; lista “Os teus jogos partilhados” (localStorage `oe_my_quiz_games`)
  com jogadores e líder; `GET …/scores` devolve também `players`.

**15/11 — API a acordar (Render grátis adormece; 1.º pedido medido a 53 s)**
- `js/ouviescrevi-api.js` (v=27 em 69 páginas): se o `frontend-config` demorar mais de 4 s aparece um aviso fixo
  «A ligar ao servidor… na primeira utilização pode demorar até 1 minuto.» (PT/EN/ES/FR/DE pelo `lang` da página);
  uma falha do `init()` já não fica guardada (antes todas as chamadas seguintes falhavam até recarregar);
  com o token em `sessionStorage` faz um pedido de aquecimento em segundo plano.
- Auditoria de 15/11: 134 testes do backend a passar; 150 páginas sem links partidos nem scripts com erro;
  cabeçalhos de segurança OK; `api.ouviescrevi.pt/docs` está público (decidir se se fecha).

**21/11 — melhorias pós-auditoria**
- API: `/docs`, `/redoc` e `/openapi.json` só existem com `ENABLE_DEBUG_ENDPOINTS=true` (em produção dão 404).
- `.github/workflows/keep-api-awake.yml`: visita `api.ouviescrevi.pt/api/status` a cada 10 min para o Render grátis
  não adormecer (gasta ~744 h das 750 h grátis/mês do Render: só chega para **um** serviço grátis na conta).
  O GitHub desliga tarefas agendadas após 60 dias sem commits no repositório.
- `backend/requirements.txt` com versões fixas (as testadas em local). Atualizar uma de cada vez e correr os testes.
- `scripts/compare_transcribe_models.py áudio.mp3 [pt]`: compara `whisper-1`, `gpt-4o-transcribe` e
  `gpt-4o-mini-transcribe` (custa cêntimos). **Atenção:** o site usa os segmentos com tempos do `whisper-1`
  (legendas, capítulos, filtros de qualidade); os `gpt-4o-*` não os devolvem, por isso trocar não é só mudar o nome.
- Páginas ES/FR/DE (`frontend/{es,fr,de}/index.html`): ~320 textos que estavam em português traduzidos (painel de
  legendas, frases de espera, erros, janelas de Corrigir/Email, botões). Ficam em PT de propósito só as linhas da
  notificação interna «Alguém clicou em Partilhar no WhatsApp» (email para o dono).
  **Corrigido:** a DE usava as chaves de CMS da PT (`seo_title`…) e mostrava a apresentação em português; agora usa
  `de_seo_*` (textos alemães acrescentados aos valores por omissão em `backend/cms.py`).
- **Chat «Perguntar à IA sobre esta transcrição» nas EN/ES/FR/DE:** `js/transcript-chat.js?v=1` (cria o painel
  debaixo da transcrição, textos na língua da página, estilos próprios). A PT continua com a versão dentro da
  `index.html`. `/transcript-chat` passou a responder em espanhol/francês/alemão (antes só PT ou EN).
  Botões que ainda estavam em PT nas ES/FR/DE: Corrigir, Traduzir, Classificar, Exportar e título das legendas.
- **Jogos partilhados ligados à conta:** coluna `owner` em `shared_quizzes` (migração automática). Com sessão iniciada,
  `POST /api/share/quiz` guarda o email da conta; `GET /api/share/quiz/mine` lista os jogos da conta (401 sem sessão);
  `DELETE /api/share/quiz/{id}/mine` tira-o da lista (o link continua a funcionar). No site, `quiz-game.js?v=6`
  (`syncMyGames`) junta os da conta aos do aparelho e `aula-pronta-ui.js?v=7` chama-o ao abrir a Aula Pronta.
  Jogos partilhados antes disto ou sem sessão continuam só no aparelho.
- **`backend/share_routes.py`:** as rotas de partilha (`/api/share/transcript`, `/api/share/quiz…`, `/share/{id}`) e os
  seus modelos saíram do `main.py` (4841 → 4670 linhas). Recebem do main as origens, o limite de pedidos e a sessão
  por `share_routes.configure(...)`. Próximos candidatos a sair do `main.py`: rotas `/api/auth/*` e de IA.
- **Primeira visita mais clara** (a partir de uma simulação de visitantes com pessoas virtuais; ainda não publicado):
  - `index.html` (PT): debaixo do botão «Transcrever agora», uma lista `.oe-home-facts` com as quotas grátis
    (3/dia sem conta, 20 com conta), o preço do Pro só com `data-pricing-only` (hoje os preços estão escondidos e o
    checkout desligado) e a privacidade em linguagem simples: o áudio é apagado minutos depois e a transcrição é
    feita pela OpenAI (EUA), tal como diz `privacidade.html`. No painel «Como funciona» há um exemplo de resultado
    (`.oe-home-example`, entrevista inventada com marcas de tempo; o painel continua escondido abaixo de 900 px).
    As 14 «Ações com IA» ficam agrupadas com títulos `.oe-output__group` (Texto, Estudar, Reuniões, Vídeo e
    podcast, Guardar). Só mudou a ordem; as funções são as mesmas. `index-home.css?v=59` em todas as páginas.
  - `en|es|fr|de/index.html`: a mesma linha de quotas e de privacidade (estilo inline, porque estas páginas não
    carregam o `index-home.css`). Não têm o preço, porque não carregam o `pricing-visibility.js`.
  - Frase «Recomendamos revisão por um advogado…» (e as traduções) tirada da privacidade: valores por omissão em
    `backend/cms.py` e no HTML estático das 5 páginas. **Se o backoffice tiver o texto guardado na base de dados,
    tem de se apagar também lá.**
  - `precos.html` (5 línguas): por baixo de «Subscrever Pro», a nota «vais para a página segura da Stripe, onde vês
    o valor e confirmas; nada é cobrado antes». Fica escondida quando o checkout está desligado.
  - Não mexido: o aviso «A ligar ao servidor…» já existe em 5 línguas (`ouviescrevi-api.js`) e a API está acordada
    pelo `keep-api-awake.yml`.

**22/11 — proposta de novo design (caderno quadriculado), ainda à parte**
- `frontend/design/inicio-caderno.html` (escolhida pelo João): nova página inicial em estilo caderno quadriculado,
  tinta azul, post-it para carregar o ficheiro, exemplo de entrevista que se escreve sozinho (voz do aparelho só se
  houver voz pt-PT; senão corre sem som) e botões Resumo/Perguntas/Legendas/Email com resultados já preparados.
  Exemplo e resultados são **ilustrativos**. `#pronto` no URL abre com o exemplo já feito. `noindex`.
  **Ainda não substitui a `index.html`:** os botões de carregar levam à ferramenta atual (`/index.html`); para a
  trocar é preciso ligar o carregamento real (API, jobs, recorte) a esta página.
- Logótipo novo (microfone que termina num aparo de caneta, «escrevi» sublinhado a vermelho):
  `frontend/design/assets/logo-ouviescrevi.svg`, `logo-simbolo.svg`, folha de apresentação `frontend/design/logo.html`.
  Ainda só usado na página nova.
- Alternativa não escolhida: `frontend/design/inicio-folha.html` (folha de 25 linhas com carimbo).
- Outra versão a analisar: `frontend/design/inicio-ardosia.html` (quadro de ardósia verde, giz, cartão com íman).
  Animações: título escrito a giz ao abrir, aula de História (Tordesilhas, ilustrativa) a escrever-se, caixa
  «Resumo em 3 pontos» desenhada no fim, apagador ao repetir, pau de giz desenhado em SVG que segue a escrita (treme,
  levanta nos espaços, viaja entre linhas, deixa pó no quadro), letras a aparecer como risco de giz e onda de som
  animada enquanto «ouve». O giz move-se com `requestAnimationFrame` (sem filtros SVG na lista, para ser fluido
  no telemóvel). Ao fazer scroll: títulos do 2.º quadro escrevem-se a giz, linhas da tabela e vistos aparecem
  em sequência (`IntersectionObserver`, classe `.anima` no `<html>`). Respeita «reduzir movimento». `#pronto` também funciona.
- Outra versão a analisar: `frontend/design/inicio-cortica.html` (quadro de cortiça, fichas pregadas com pioneses,
  fio vermelho por trás das fichas a ligá-las; cada fala da entrevista é pregada numa ficha nova; ao fazer scroll as
  fichas da 2.ª parte são pregadas). Letra das fichas: Courier Prime.
- Outra versão a analisar: `frontend/design/inicio-mensagens.html` (mensagens de voz que passam a texto numa
  conversa; fundo azul-claro, botão laranja; não imita nenhuma app real).
- **Menu «Design»** para saltar entre o site clássico e as versões novas: no site atual é um dropdown em
  `frontend/header.html` (depois de «Para quem»); nas páginas `frontend/design/inicio-*.html` é injetado por
  `frontend/design/assets/troca-design.js` (visível também no telemóvel).
- Letras alojadas em `frontend/design/assets/fonts/` (Schibsted Grotesk, Shantell Sans, Martian Mono, Courier Prime).
- Cópia da página inicial antes do redesign: `backups/pagina-inicial-2026-11-22/` (index.html, CSS, logos e capturas).

**22/11 — o design «Mensagens de voz» passa a ser o principal em todo o site**
- Paleta trocada em todo o `frontend/` (exceto backoffice, `theme-dark.css`, `design/` e `archive/`): o roxo/ciano
  antigo passou a azul-claro (`#e4f0fc` fundo), azul-escuro (`#0f2a4f` texto) e laranja (`#d4630d` botões).
  Script usado: troca de cores hex/rgba + gradientes roxo→azul passam a laranja sólido; faixa de novidades azul-escuro.
- Letra do site: Schibsted Grotesk (alojada em `/design/assets/fonts/`); o Inter do Google deixou de ser carregado.
- **Designs na própria página:** o botão «Design» do cabeçalho (5 línguas) muda o aspeto sem sair da página —
  `theme-ui.js?v=3` põe `html[data-design=caderno|ardosia|cortica|folha]` (guardado em `localStorage.oe_design`;
  sem valor = Mensagens de voz). Cores/fundos em `css/temas.css`; a «Ardósia» reaproveita o antigo modo escuro
  (`html.oe-theme-dark`) com o `theme-dark.css` recolorido a verde-ardósia. O botão 🌙 foi escondido.
- O logótipo do cabeçalho era cortado (imagem 40×47 numa caixa 34×40): agora tem 34×40 fixos. Na ardósia o `main`
  tem moldura de madeira e parapeito com giz (`temas.css`); cortiça com moldura; caderno com margem vermelha.
- Em cada design a conversa da página inicial muda de aspeto (`conversa-demo.css`) e há um instrumento a escrever
  as mensagens: **giz** na ardósia, **caneta** no caderno e na folha (`conversa-demo.js`, segue um cursor invisível).
  Logótipo claro na ardósia (`/design/assets/logo-simbolo-giz.svg`); menus e lista de cores usam as variáveis do design;
  a regra escura dos botões no `theme-dark.css` exclui os botões da conversa e das bolinhas.
- A escolha de aspeto é discreta: só um ícone de paleta no cabeçalho (sem a palavra «Design»), que abre uma fila de
  5 bolinhas de cor com o nome por baixo (estilos no fim de `ouviescrevi-pro.css`).
- Menu de três riscas no telemóvel a toda a largura, linhas de 56 px (fim de `ouviescrevi-pro.css`); abrir o
  «Design» fecha o menu.
- Resumo por URL: no telemóvel o botão e o seletor esticavam (flex-basis em coluna); botão laranja `#e46c12`.
- **Resumo por URL melhorado** (`/summarize-url` em `backend/main.py`, testes em `backend/tests/test_url_summary.py`):
  um só pedido à IA com o artigo inteiro (até 12 000 caracteres) em vez de «Parte 1, Parte 2…»; língua via
  `ai_output_language()` (as páginas ES/FR/DE passavam `apiLang: "pt"` — corrigido); estilos `normal`, `minuta`,
  `pontos`, `simples`, `detalhado` (novos também em `SUMMARY_STYLES`); devolve `title`, `site`, `words`,
  `reading_minutes`, `truncated` e `text` (para as perguntas); erros claros nas 5 línguas (422 site bloqueado,
  502 sem texto técnico). Página (`js/url-resumo-ui.js?v=2`): ficha do artigo, botão «Colar», exportar Word/PDF
  (imprimir)/WhatsApp/email e «Perguntar sobre o artigo» (usa `/transcript-chat`).
- Logótipo novo (`/design/assets/logo-simbolo.svg`) nos cabeçalhos e rodapés das 5 línguas. Os favicons PNG
  continuam os antigos.
- **Botão «Design» fora do menu de três riscas** (`<details class="oe-design-troca">` nos 5 `header.html`, estilos
  no fim de `ouviescrevi-pro.css`): principal + caderno, ardósia, cortiça e folha (só páginas iniciais de teste).
- A página inicial (`index.html`, ferramenta real) tem agora o título «Ouviste. Nós escrevemos.» e, no painel da
  direita (`#homeOutputPlaceholder`, que o JS esconde quando aparece um resultado), a **conversa animada** de
  mensagens de voz: `css/conversa-demo.css?v=2` + `js/conversa-demo.js?v=1` (exemplo ilustrativo, voz pt-PT do
  aparelho se existir). Também aparece no telemóvel. Os cartões «Exemplos de uso» entram ao fazer scroll.
  O título grande é um `<p class="oe-home-hero">` fixo, fora do bloco CMS `home_intro_html` (o backoffice tinha texto
  próprio guardado e escondia o título); o texto do CMS aparece por baixo. No telemóvel a conversa não tem scroll
  próprio (prendia o dedo); cresce com a página. Botão «Transcrever agora» em laranja vivo `#e46c12`.

**Simulador (fase 0)**
- `d8c09dd` + `91c1e87` Endpoint temporário de teste, já removido (ver ponto 4).

## 3. Regras a manter

- Nunca `overflow:hidden` na `.oe-pro-nav` (corta submenus). Painel mobile: não usar `translateX` para o esconder.
- Botões novos que não devem ter o estilo genérico: acrescentar à lista `:not(...)` da regra `.oe-page button` em
  **`css/ouviescrevi.css` e também em `css/theme-dark.css`** (há uma cópia com `!important`).
- Não pôr `<script …></script>` literal dentro de strings/templates JS em HTML.
- Prompts de IA em PT: `ai_output_language()` (“português de Portugal, nunca do Brasil”).
- Botões de ações com IA: na PT o `comLoading` da `index.html`; nas EN/ES/FR/DE o `OuviescreviUI.wireActionLoading`
  (não voltar a `btn.innerHTML = "A gerar..."` à mão). Nunca ler `event` depois de um `await`.
- Alterações pequenas, explicadas antes; publicar só com pedido do dono.

## 4. Simulador de apostas (projeto à parte, repositório PRIVADO)

**Não está neste repositório** (que é **público**) e os detalhes dele **não se escrevem aqui**.
O Ouviescrevi só partilha o domínio (subdomínio `simulador.`, Cloudflare).
O contexto, as regras e os prompts para o Cursor e para o Claude Code estão no **`CONTEXTO_CURSOR.md` da pasta do
simulador** (secções 2, 7 e 8). Abrir **essa pasta** no Cursor / Claude Code para trabalhar nele.

## 5. Ideias pendentes (Ouviescrevi)

1. Testar modelo de transcrição mais recente (`gpt-4o-transcribe` / mini) vs `whisper-1`.
2. Render pago (sem adormecer) e monitorização de erros (Sentry).
3. Páginas EN/ES/FR/DE sem escolha de trecho (`media-trim-ui.js`, ~1800 linhas só em PT) e sem histórico de transcrições.
4. Uma só versão de Python (Render 3.11, CI 3.12, local 3.14); continuar a dividir `backend/main.py` (auth, IA).

Confirma que leste isto e diz-me o estado atual (`git status`, testes) antes de propor alterações.
