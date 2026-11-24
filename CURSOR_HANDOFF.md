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
- **Testes:** `cd backend; py -m pytest -q` → 193 testes (24/11) (usar o Python do sistema; o `backend/venv` antigo está partido;
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
  **24/11:** o GitHub só a corria de 4 em 4 a 8 em 8 horas (8 vezes em 3 dias), por isso a API adormecia na mesma.
  O João criou um monitor no **UptimeRobot** (conta grátis «Ouviescrevi», HTTP(s) a `api.ouviescrevi.pt/api/status`
  de 5 em 5 min, avisa por email se a API cair). O workflow do GitHub fica como reserva.
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

- Com sessão iniciada, a «Navegação» (`#oeAccountNav`) deixou de flutuar fixa à esquerda: é uma barra de separadores
  centrada por cima do título (`index-home.css?v=65`), com as cores de cada design (variáveis `--pro-*`). No
  telemóvel fica numa linha e esconde «Ajuda» (está no menu de três riscas). As ligações da barra já não herdam o
  laranja genérico dos links (exclusão em `ouviescrevi-pro.css`). Mensagens de voz: um só laranja nos botões
  principais, `#c2560c` (contraste 4.5:1 com branco), no cabeçalho e no «Transcrever agora». `ouviescrevi.css?v=56`.

- Escolha de aspeto só com 3 designs: Mensagens de voz, Caderno e Ardósia (`theme-ui.js?v=4`, cabeçalhos ×5,
  `LAYOUT_V=17`). Cortiça e Folha desativados; o CSS ficou em `temas.css`/`conversa-demo.css` para poder voltar.
  Quem os tinha escolhido volta a Mensagens de voz. Botões por design em `temas.css?v=4` (variáveis `--oe-btn*`):
  laranja claro do mockup `#fd8c32` (hover `#f07a1c`) · azul `#1e3fa8` · giz amarelo `#f2d24b` com texto escuro; «Ouvir um exemplo» passa a botão de
  contorno. Rodapé por design (`--rod-*`): azul-marinho liso, capa azul com grelha, madeira escura na ardósia
  (sem o roxo antigo `#a5b4fc`). `ouviescrevi.css?v=57`, `conversa-demo.css?v=10`.

- Página inicial como nas demos (`design/inicio-caderno.html`, `inicio-ardosia.html`), sem mudar o HTML da ferramenta:
  `conversa-demo.js?v=4` tem um modo «em linhas» para Caderno e Ardósia (minuto · quem · texto, caneta/giz a seguir as
  letras); no fim, o Caderno mostra o visto vermelho «Transcrito!» e a Ardósia o resumo numa caixa desenhada a giz; ao
  repetir, o apagador passa pelo quadro. Ao trocar de design a conversa volta ao início (MutationObserver em
  `data-design`). CSS em `conversa-demo.css?v=16`: formulário como nota amarela com fita (Caderno) ou cartão branco
  com íman (Ardósia, com as variáveis `--oe-btn-*` do tema escuro reescritas dentro do cartão); título à esquerda no
  Caderno; na Ardósia o título escreve-se a giz e é sublinhado a amarelo. Novo `<p class="oe-tema-titulo">` no
  `#uploadForm` (só visível nesses designs). Risco por baixo do nome no logótipo (`temas.css?v=6`).

- «??» em `professores.html`, `podcasts.html` e `corretor.html`: texto gravado em cp1252 (emojis → «?», travessão →
  U+0097, acentos → «?»). Reposto à mão; `scripts/frontend_smoke_test.py` passa a falhar se voltar a haver «??» em
  texto ou caracteres C1. Fundos de botões no laranja-escuro antigo (e gradientes com roxo `#6366f1`) passaram a
  `var(--oe-btn, #fd8c32)` / `var(--oe-btn-hover, #f07a1c)` em 16 ficheiros CSS (as cores de texto ficaram); em
  `temas.css` os `.oe-lp-btn` (primário cheio, secundário com contorno), o separador ativo do menu e o botão
  flutuante seguem a cor do design. Mensagens de voz como o desenho: no computador (≥1000px) `#adminContent` passa a
  grelha de 2 colunas (título em 2 linhas, caixa, «Exemplo de transcrição» e faixa «Sem registo · …» à esquerda;
  conversa à direita), bolhas de texto verdes com ✓✓ azul, onda azul. Novo bloco `.oe-exemplo-tr` no `index.html`.

- Exemplo da página inicial com «Parar» (`conversa-demo.js?v=5`): enquanto corre, o botão «Ouvir» passa a «Parar» (ícone
  quadrado, `aria-pressed`); parar cala a voz, larga a caneta/giz e deixa o que já foi escrito («Ouvir outra vez»).

- Letras em castanho-laranja: `--pro-accent` passa a azul `#1d4a8f` (hover `#0f2a4f`) em Mensagens de voz; todas as
  cores de texto `#b5520a/#9c4607/#a94c06/#8f4007/#7a4a1c` (CSS, `<style>` das páginas e JS) passaram a
  `var(--pro-accent, #1d4a8f)`, por isso no Caderno/Ardósia seguem o azul/amarelo desses designs. Ícone do separador:
  `logos/ouviescrevi-icon-pro.png`, `ouviescreviicon.png` e `icon-192.png` redesenhados com o novo símbolo (links com
  `?v=2`, cache do service worker `oe-shell-v4`). Mensagens de voz com animações (título, caixa e conversa a subir ao
  abrir, anel no ▶, onda a mexer enquanto toca, linhas do «Exemplo de transcrição» a entrar). Correção: no telemóvel o
  `#uploadForm` tinha `height:100%` e o exemplo ficava por baixo da conversa.

- Últimas cores antigas (bordas, contornos, sombras `rgba(181,82,10,…)`, gradientes, roxos `#7c3aed/#6366f1/…`, texto
  castanho das caixas de aviso) trocadas em 23 ficheiros: laranja → `var(--oe-btn, #fd8c32)`, roxo → azuis da
  paleta, texto de avisos → `#0f2a4f`. Fica de propósito `#9c4607` no fundo do símbolo do logótipo (não se vê).

- Transcrição com frase repetida sem fim («o que eu vi, o que eu vi, …», alucinação do Whisper num trecho de vídeo):
  o filtro só apanhava uma palavra repetida. `main.py`: `_is_phrase_loop` (frase de 2–8 palavras repetida 6+ vezes
  a cobrir ≥85% do bloco → bloco/segmento deitado fora) e, dentro de fala real, a frase repetida 4+ vezes fica uma vez
  (`collapse_word_loops_in_block`). `clean_transcription_text` tira os blocos-loop antes de colapsar. 4 testes novos
  em `tests/test_whisper_filter.py` (refrão de 3 vezes mantém-se).

- Menu nas páginas EN/ES/FR/DE: a regra antiga `input,button{width:100%}` dessas páginas esticava «Log in» e tapava
  «Sign up»; no cabeçalho os botões passam a `width:auto` (`ouviescrevi-pro.css`). Páginas iniciais EN/ES/FR/DE com o
  mesmo desenho da PT: `theme-ui.js` (antes a troca de aspeto não funcionava nelas), título grande traduzido,
  etiquetas, nota/cartão do formulário, «Exemplo de transcrição» e a conversa animada (gerado a partir do painel PT).
  `conversa-demo.js?v=7` tem a tabela `TEXTOS` por `<html lang>` (falas, botões, resumo, visto, voz do aparelho na
  língua da página) e esconde o exemplo quando aparecem `#loading`/`#output`. As regras do botão principal usam
  `:is(#btnTranscrever, #btnTranscribe, #btnTranscribir, #btnTranscrire, #btnTranskribieren)`. Por fazer: as
  funcionalidades que só a PT tem (histórico, trecho de ficheiro longo, etc.) e as outras páginas traduzidas.

- Língua da transcrição: fica sempre na língua original do áudio (pedido do João). Acabou a tradução automática para
  pt no site PT (`main.py`, bloco «original_*» fica vazio; traduzir é com o botão «Traduzir»). A repetição forçada em
  pt só é aceite se for claramente melhor: `PT_RETRY_LOGPROB_MARGIN` passou de 0.1 para -0.15 (músicas em inglês
  saíam com frases inventadas em português). Testes em `tests/test_pt_language.py`. Notícia de 26/10 na faixa
  atualizada (`news-ticker.js?v=5`). Erro ao transcrever (ex.: servidor reiniciou): deixa de ir para a caixa do texto;
  aparece o cartão `#oeErroTranscricao` com botão «Tentar outra vez» (`index.html`, `.oe-erro-card` em
  `conversa-demo.css`). Menu de idiomas: lista limpa com ✓ na língua atual (`ouviescrevi-pro.css`).
- «Ver tradução» na caixa da transcrição (site PT): o servidor volta a enviar `original_language` quando o áudio não
  é português (`original_formatted` continua vazio). O `index.html` mostra «Texto original (inglês).» + «Ver tradução»;
  o 1.º clique chama `/translate` para português e guarda o resultado; depois alterna original/tradução.
- **IA de texto com Claude + reserva OpenAI** (`backend/ai_chat.py`): todas as chamadas que eram
  `client.chat.completions.create` passam por `ai_chat.create(...)` (mesmos argumentos, mesma resposta
  `resp.choices[0].message.content`). Com `ANTHROPIC_API_KEY` no Render usa o Claude (`CLAUDE_MODEL`, por omissão
  `claude-opus-5-5`, `CLAUDE_EFFORT=low`); se falhar (sem crédito, chave errada, serviço em baixo) usa a OpenAI com o
  modelo de sempre. Sem a chave tudo vai para a OpenAI. **O Whisper (transcrição) continua na OpenAI** (`client`).
  Testes em `tests/test_ai_chat.py`; nos testes `ANTHROPIC_API_KEY=""` (conftest). Backoffice: cartão «Créditos da IA»
  com links para carregar Anthropic e OpenAI. Nova dependência `anthropic==2.12.0`.

- Faixa de novidades (`news-ticker.js?v=6`): saiu a notícia de 26/10 (tradução automática, já não existe); entram 3
  novidades de 23/11/2026 em PT (novo aspeto, língua original, «Tentar outra vez») e 2 em EN/ES/FR/DE. Nomes dos
  designs traduzidos nos cabeçalhos EN/ES/FR/DE (Notebook/Chalkboard, Cuaderno/Pizarra, Cahier/Ardoise, Heft/Tafel;
  `LAYOUT_V=18`).
- Cartão «A transcrever» (`#loading`, coluna do resultado) com aspeto próprio em cada design (fim de
  `conversa-demo.css?v=28`, 5 páginas iniciais): Mensagens de voz = balão de mensagem com a barra em forma de onda
  sonora (máscara SVG `--onda`); Caderno = folha pautada com margem vermelha, fita-cola e marcador amarelo; Ardósia =
  contorno a giz no quadro e risco de giz amarelo. Na Ardósia o botão Cancelar redefine `--oe-btn-bg/-border/-text`
  (o `theme-dark.css` pinta os botões com essas variáveis e `!important`).
- Animações de entrada no Caderno (`conversa-demo.css?v=29`, só com `prefers-reduced-motion: no-preference`): título
  escreve-se a caneta (`oe-escreve`), «Nós escrevemos.» leva traço de marcador amarelo (`oe-marca`), a nota amarela cai
  e cola-se (`oe-cola`, sem `to` para manter a rotação de cada largura) com a fita a aparecer, etiquetas e coluna do
  resultado aparecem a seguir; o cartão «A transcrever» também entra com `oe-cola`.
- Risco por baixo de «escrevi» no logótipo também nas Mensagens de voz (laranja); no rodapé do Caderno (azul-escuro)
  o símbolo passa ao claro `logo-simbolo-giz.svg` (`temas.css?v=10`, `ouviescrevi.css?v=66` em 138 páginas).
- Buraco branco entre o chat e «Como correu a transcrição?»: o `#output` esticava até à altura da coluna da esquerda
  (a caixa do texto tem altura máxima e a grelha `.oe-output__layout` tem `align-content:start`). No computador o
  `#output` passa a `height:auto; align-self:start` (`index-home.css?v=69`).
- Menu do telemóvel (`ouviescrevi-pro.css?v=25`, `ouviescrevi-nav.js?v=2` carregado pelo `ouviescrevi-ui.js?v=36`):
  só uma secção aberta de cada vez; ao abrir o menu, abre logo a secção da página atual; ligações em quadrados de
  2 colunas com a página atual marcada; Entrar/Registar/bandeira e «Transcrever grátis» fixos no fundo do painel.
- Caderno no telemóvel: `main` com `padding-left: 38px` para o texto não ficar em cima da linha vermelha (`temas.css?v=11`).
- Blocos do fim da página inicial com aspeto por design (`conversa-demo.css?v=30`): «Exemplos de uso» com «Ver como →»
  (traduzido por `html[lang]`), contador de transcrições e «O que é o Ouviescrevi?». Mensagens de voz = balões;
  Caderno = fichas pautadas, nota amarela e folha com margem; Ardósia = contornos a giz.
- Outras páginas nos 3 designs (`temas.css?v=15`, `ouviescrevi.css?v=68`): botão principal de cada ferramenta
  (`.oe-*-form__submit` e `.oe-pro-btn--primary`) cheio com a cor do design — na Ardósia redefine `--oe-btn-bg/-border/-text`
  porque o `theme-dark.css` os pinta com `!important` e um id; no telemóvel os rodapés dos formulários em coluna tinham
  `flex: 1 1 160px/200px` a virar altura (botão com 160 px e espaço vazio) → `flex-basis:auto`. `--pro-muted` (usado
  em várias páginas e nunca definido) = `--pro-text-muted`. Cabeçalho das `.oe-legal-table` com o tom do design (na
  Ardósia era claro com texto claro). Ardósia: campos de escrever em verde-escuro em vez de azul-marinho.
  `sugestoes.css?v=6`: opções «Tipo» com a cor do design.
- Cookies no telemóvel: a tabela (nomes como `ouviescrevi_api_token` que não partiam) alargava a página toda. Em
  ≤600px a `.oe-legal-table` desliza só dentro dela e os `code` partem (`ouviescrevi.css?v=69`); na Ardósia os `code`
  deixam de ter fundo branco (`temas.css?v=16`).
- «Ouvir um exemplo» convida ao clique nos 3 designs até ao 1.º clique (`:not([aria-pressed])`, `conversa-demo.css?v=31`):
  duas ondas a sair do botão (`::before/::after`, cor `--oe-btn`, porque o `box-shadow` do botão tem `!important`) e
  um pequeno salto a cada 4 s; só com `prefers-reduced-motion: no-preference`.
- Correções da revisão de 23/11: o convite pára sozinho (4–5 vezes) e também ao passar o rato; a entrada do Caderno
  só passa na 1.ª visita da sessão (`conversa-demo.js?v=8` põe `html.oe-intro-vista` via `sessionStorage`); «Ver como →»
  com texto alternativo vazio (leitores de ecrã). **Texto azul-escuro `#0f2a4f` nos botões laranja** de Mensagens de voz
  (`--oe-btn-texto`; branco sobre `#fd8c32` dava 2,3:1) e os ~26 `color:#fff` fixos ao lado de `var(--oe-btn)` nos CSS
  das ferramentas passam a `var(--oe-btn-texto, #fff)`. O CTA do cabeçalho fica fora da regra dos botões principais
  (tinha crescido para 52 px). Menu do telemóvel: lista de línguas abre para cima; quadrados partem palavras longas (DE).
  Ardósia: também campos number/tel/date/time. Caderno: padding também em `main.oe-suggestions`. Preços (×5): tirada a
  segunda cópia antiga `ouviescrevi-pro.css?v=7`.
- **Auditoria às funcionalidades (23/11) — servidor** (`backend/`, 182 testes):
  - Segurança: links do email de recuperação, da partilha e da faturação usam `security.public_site_origin` (só aceita
    a Origin se estiver em `ALLOWED_ORIGINS`; antes um pedido com Origin falsa levava o token a outro site).
  - Transcrição: avisos em linguagem simples (sem «insufficient_quota»/«segmentos falharam»); falta de créditos manda
    email ao dono (`_notify_owner_async`, máx. 1/hora). **Áudio difícil**: `audio_is_low_confidence` (média
    `avg_logprob` < −0.9 ou `no_speech` > 0.6 nos segmentos que ficaram) acrescenta `LOW_CONFIDENCE_WARNING` e
    `low_confidence`; o trabalho devolve também `detected_language`. O filtro deita fora segmentos com pouca fala E
    pouca confiança mesmo que «pareçam frase». Tempos de um trecho contam do início do original (`start_offset`, novo
    campo `time_offset_sec` para cortes feitos no browser). Sem conta: máx. 60 min por ficheiro (`ANON_MAX_DURATION_SEC`);
    a quota conta os trabalhos a decorrer (`_count_active_transcribe_jobs`). `speech_*.wav` apagado no fim de cada
    parte. ffmpeg com tempo limite proporcional à duração (`ffmpeg_timeout_for`). Tempos ≥100 min (`[105:00]`) aceites.
  - Privacidade: `async_jobs` de transcrição apagados ao fim de 3 h (`ASYNC_JOBS_MAX_AGE_HOURS`, limpeza a cada 10 min;
    antes só no arranque, 24 h).
  - IA: `split_text_batches` — Corrigir, Traduzir e Locutores trabalham em partes (antes cortavam textos longos);
    Locutores com nomes «Pessoa 1/2» e, se a IA perder linhas, fica o bloco original. `require_ai_text` (máx.
    150 mil caracteres, mensagem clara). `ai_http_error`: visitante vê «serviço de IA temporariamente indisponível»,
    o detalhe vai para o registo. Resumo com `SUMMARY_MAX_TOKENS` por estilo (600 cortava o detalhado/minuta).
  - Partilha: `delete_key` (só o hash na BD, coluna nova `shared_transcripts.delete_key_hash`) e
    `POST /api/share/transcript/{id}/delete`; a resposta traz `public_notice`. **Falta o botão no site** (`share-ui.js`
    está a ser alterado pelo Cursor — guardar a `delete_key` e mostrar «Apagar link»).
  - Sessões: `POST /api/auth/logout` (o «Sair» do `auth-ui.js?v=27` chama-o) e mudar a palavra-passe termina as
    sessões antigas.

- Aviso «servidor a acordar» (`ouviescrevi-api.js?v=28`): passou do topo (tapava o menu) para o canto de baixo,
  texto mais calmo («estava em pausa… podes ir escolhendo o ficheiro») e contador de segundos a subir.

- SEO (23/11): 20 títulos com mais de 65 caracteres encurtados (PT/EN/ES/FR/DE; `og:title`/`twitter:title` iguais),
  5 descrições curtas completadas (blog reuniões, aulas DE/ES/FR, podcasts DE) e `lastmod` do `sitemap.xml` com a
  data do último commit de cada página.
- SEO (23/11, 2.ª parte): a página inicial passa a ser anunciada como pasta (`https://www.ouviescrevi.pt/`, `/en/`,
  `/es/`, `/fr/`, `/de/`) no canónico, `og:url`, hreflang do sitemap e no `js/ouviescrevi-seo.js` (`cleanUrl`).
  **Atenção:** o `ouviescrevi-seo.js` reescreve título/descrição a partir da lista `PAGES` — ao mudar o `<title>` de
  uma página, mudar também lá. Script agora com `?v=2` (127 páginas). O `frontend_smoke_test.py` aceita URLs de pasta
  no sitemap e verifica o canónico das 5 páginas iniciais.
- Google Search Console (23/11): propriedade `https://www.ouviescrevi.pt/` validada com o ficheiro
  `frontend/googled4d64ba426e88117.html` — **não apagar** (o Google volta a verificar). O smoke test ignora-o.
- Página nova `audio-whatsapp-texto.html` (só PT): converter notas de voz do WhatsApp em texto, passo a passo
  Android/iPhone/computador. Motivo: «converter áudio whatsapp em texto» aparecia no Search Console sem página
  própria. No sitemap e ligada a partir do `blog/index.html` e de `blog/transcrever-audio-online.html`.
  O upload aceita `.opus` (o servidor converte tudo com ffmpeg); **por testar num iPhone real** se o Safari deixa
  escolher o `.opus` guardado em Ficheiros.
- **Bug corrigido (origem das visitas):** `POST /api/track-visit` usava o cabeçalho `Referer` do próprio pedido
  quando `document.referrer` vinha vazio — e esse cabeçalho é sempre o site. Todas as entradas diretas (escrever o
  endereço, apps como o TikTok) apareciam como `https://www.ouviescrevi.pt/`. Agora só conta `document.referrer`.
  Dados de origem fiáveis só a partir de 23/11/2026.
- Backoffice, separador **Marketing** (`js/backoffice-marketing.js?v=1`, `GET /api/admin/marketing?days=N`,
  `analytics.build_marketing_report` + `classify_source`): pessoas (sem bots/equipa) por origem (TikTok, Google,
  ChatGPT, Direto…), campanhas UTM, páginas de entrada, dia a dia com transcrições; links com UTM para as redes
  (bio do TikTok, etc.) e botão «Copiar relatório para o Claude». `backoffice.js?v=33`. Testes em
  `tests/test_marketing_report.py`. Segurança: páginas, UTM e domínios vêm do visitante — o relatório passa-os por
  `analytics.safe_label` (só `A-Za-z0-9_./-`) e o texto copiado põe o JSON entre `<dados>…</dados>`, marcado como
  dados e não instruções (`backoffice-marketing.js?v=2`).
- **Chave do Claude** (só leitura do marketing): Marketing → «Criar chave para o Claude» (`POST/DELETE
  /api/admin/claude-key`, só admin) cria uma sessão `claude-leitura` com o papel `claude_read` (180 dias), que fica
  fora do `ROLE_LEVEL` — falha em todos os `require_role` e só o `GET /api/admin/marketing` a aceita. A chave
  descarrega-se para um ficheiro (`ouviescrevi-claude.key`), nunca aparece no ecrã. Criar outra ou revogar desliga a
  anterior. O Claude guarda-a em `%USERPROFILE%\.ouviescrevi\claude.key` e lê com `py scripts/marketing_report.py
  [dias]`. Teste: `test_claude_key_only_opens_marketing` (percorre todos os GET de `/api/admin/` com a chave).
  `backoffice-marketing.js?v=3`.
- **Opiniões dos utilizadores (testemunhos)**: no pedido de estrelas da PT (`feedback-pulse.js?v=2`), com 4–5
  estrelas aparece «Podem mostrar este comentário no site» + nome/profissão opcional. `POST /api/suggestions` com
  `publicar` + `nome_publico` grava também na tabela `testemunhos` (só com autorização, 4–5 estrelas e comentário
  escrito), com `aprovado=0`. Backoffice → Marketing → «Opiniões dos utilizadores» (`GET/POST/DELETE
  /api/admin/testemunhos[/{id}]`, aprovar = editor). A página inicial PT mostra as aprovadas em
  `#oeUserTestimonials` via `GET /api/testimonials?lang=pt` (máx. 6, só nome/texto/estrelas, `textContent`; vazio
  = escondido). `backoffice-marketing.js?v=4`. Testes: `tests/test_testimonials.py`.
- **3 páginas SEO novas** (modelo de `audio-whatsapp-texto.html`): `transcrever-video-texto.html`,
  `audio-para-word.html` (exportação DOC grátis; não falar do Pro, que não está à venda), `resumir-audio.html`.
  No sitemap e ligadas a partir de `blog/index.html` e `blog/transcrever-audio-online.html`.
- **Backoffice: meta e equipa (25/11)**
  - *Aparelho da equipa em qualquer rede:* ao abrir o backoffice como admin, `POST /api/admin/team-device` marca o IP
    atual e devolve um token guardado no browser (`localStorage.oe_team_device`; o servidor só guarda o hash em
    `team_device_tokens`, máx. 10). O site envia-o em `/api/track-visit` (`team_device`, `ouviescrevi-ui.js?v=38` em
    todas as páginas); com token válido, o IP dessa rede entra na lista da equipa (`owner_visitor_uids`).
    «Desmarcar» no Painel grava `oe_team_device=nao` para não voltar a marcar sozinho.
  - *Topo do Painel* (`backoffice-meta.js?v=1`): pessoas reais hoje, últimos 7 dias vs anteriores, média/dia com
    semáforo da meta (🟢 ≥10/dia, 🟡 3–10, 🔴 <3 até ao fim de fevereiro de 2027), entradas do Google e o percurso.
  - *Percurso:* tabela `eventos` (`carregou` ao aceitar um upload em `/transcribe` e `/video-subs`; `terminou` na
    1.ª consulta de um trabalho concluído). `build_marketing_report` devolve `percurso` {entraram, carregaram,
    terminaram, voltaram (vistos em ≥2 dias)}, sem a equipa.
  - *Resumo por email ao domingo* (`weekly_report.py`): fio iniciado no arranque só com `APP_ENV=production`;
    domingo a partir das 9h UTC, uma vez por semana (`weekly_report_week`; desligar com `weekly_report_enabled=0`).
    Botão «Enviar resumo semanal agora» → `POST /api/admin/weekly-report/test`.
  - *Menu:* Sugestões AI, Estudo AI, Emails e Planos Pro ficam dentro de «Mais» (`<details>`).
    `backoffice.js?v=34`, `backoffice.css?v=29`. Testes: `tests/test_backoffice_meta.py`.
  - Por fazer: dados do Google Search Console no backoffice (precisa de acesso do João à conta Google).

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
