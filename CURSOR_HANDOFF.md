# Passagem para o Cursor — Ouviescrevi / TranscreveJá + simulador online (26/10 a 02/11/2026)

Cola isto no Cursor. Antes de mexer em código: `git pull` e `git log --oneline d65e4f7..HEAD`.
Há **dois projetos**: o Ouviescrevi (este repositório, `c:\Users\joao_\transcreveja`, público) e o simulador
online (`c:\Projetos\Betclic-online`, repositório privado — ver secção 4). Não os misturar.
Está tudo publicado no `main` (GitHub Joaopmluis35/transcreveja). **Não refaças nada disto.**

---

## 1. Infraestrutura (confirmado em produção)

- **Frontend:** Vercel. Raiz do projeto Vercel = raiz do repositório, pasta publicada = `frontend/`.
  As regras (cabeçalhos de segurança, redirects, rewrite `/s/:id`) estão em **`/vercel.json` na raiz**.
  Não há Cloudflare Pages (`_headers`, `_redirects`, `wrangler.toml` foram apagados).
- **Domínio canónico:** https://www.ouviescrevi.pt (Cloudflare à frente da Vercel).
- **API:** Render, plano free, https://api.ouviescrevi.pt, base de dados Turso. autoDeploy no push para `main`.
  **O servidor corre nos EUA (Ashburn, Virgínia, rede Amazon)**, não em Frankfurt como diz o `render.yaml`.
- **Testes:** `cd backend; py -m pytest -q` → 134 testes (usar o Python do sistema; o `backend/venv` antigo está partido;
  `backend/.venv` tem uvicorn para correr a API em local). Frontend: `py scripts/frontend_smoke_test.py`.
- **Cache:** CSS/JS com `?v=N` nos HTML. Ao mudar um ficheiro, sobe o `?v=` em **todas** as páginas que o usam.
  Versões atuais: `ouviescrevi.css?v=44` (importa `ouviescrevi-pro.css?v=4` e `theme-dark.css?v=5`),
  `index-home.css?v=58`, `ouviescrevi-ui.js?v=29`, `news-ticker.js?v=4`, `quiz-game.js?v=5`, `quiz-game.css?v=4`,
  `aula-pronta-ui.js?v=6`, `aula-pronta.css?v=6`.
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

**Simulador (fase 0)**
- `d8c09dd` + `91c1e87` Endpoint temporário de teste, já removido (ver ponto 4).

## 3. Regras a manter

- Nunca `overflow:hidden` na `.oe-pro-nav` (corta submenus). Painel mobile: não usar `translateX` para o esconder.
- Botões novos que não devem ter o estilo genérico: acrescentar à lista `:not(...)` da regra `.oe-page button` em
  **`css/ouviescrevi.css` e também em `css/theme-dark.css`** (há uma cópia com `!important`).
- Não pôr `<script …></script>` literal dentro de strings/templates JS em HTML.
- Prompts de IA em PT: `ai_output_language()` (“português de Portugal, nunca do Brasil”).
- Alterações pequenas, explicadas antes; publicar só com pedido do dono.

## 4. Simulador de apostas — cópia online (projeto à parte, repositório PRIVADO)

**Não está no repositório transcreveja** (que é **público**). O Ouviescrevi só partilha o domínio
(`simulador.ouviescrevi.pt`, Cloudflare).

- **Código:** `c:\Projetos\Betclic-online` → **`github.com/Joaopmluis35/ouviescrevi-simulador` (privado)**.
  Abrir **essa pasta** no Cursor; ler `CONTEXTO_CURSOR.md` (**secções 7 e 8**) e `GUIA_ONLINE.md`.
  **Nunca** mexer em `c:\Projetos\Betclic` (PC local, 5050) excepto espelhar `betclic-boletim.user.js` se mudares o script.
- **Arquitetura actual (03/11/2026):** Flask + **vigia** no **PC Windows** (`127.0.0.1:5051`) → Cloudflare Tunnel + Access
  → `https://simulador.ouviescrevi.pt`. (Termux no Android ficou de reserva; a Betclic dá 403 a partir de datacenters.)
- **Script:** Tampermonkey **1.19+** — PC Chrome/`www`; Android Firefox/`m.betclic.pt`. Depois de atualizar: Betclic F5 → Verificar.
- **Melhorias recentes (secção 8 do CONTEXTO):** Gerar bilhete só no botão; odd min/max; mobile mais rápido (`fast=1`);
  limpar boletim no m. antes de aposta nova; não desmarcar odds com 2.º clique; cruz no popup; logs quando 17→16
  (jogo repetido, não o Al Nassr); `?v=online-28`; ~209 testes.
- **Atualizar:** alterar em `Betclic-online` → testes → `?v=` / `@version` → atualizar CONTEXTO → **commit + push** → reiniciar 5051.
- **Utilizadores:** dono `jonyluis1989@gmail.com`; amigos = email em `online.env` **e** Access.

## 5. Ideias pendentes (Ouviescrevi)

1. Testar modelo de transcrição mais recente (`gpt-4o-transcribe` / mini) vs `whisper-1`.
2. Render pago (sem adormecer) e monitorização de erros (Sentry).
3. Páginas EN/ES/FR/DE sem chat, escolha de trecho e histórico.
4. Fixar versões em `requirements.txt`; uma só versão de Python; dividir `backend/main.py`.
5. Os jogos partilhados só ficam guardados no dispositivo; para os ter em qualquer dispositivo, associar à conta.

Confirma que leste isto e diz-me o estado atual (`git status`, testes) antes de propor alterações.
