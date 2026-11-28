/* Assistente de ajuda (28/11): bolha «Ajuda» no canto de cada página. Pergunta ao /help-chat da API, que só responde
   sobre o Ouviescrevi com os limites e preços reais. Carregado pelo ouviescrevi-ui.js (mountHelpAssistant). */
(function (global) {
  "use strict";
  if (global.OuviescreviHelp) return;
  var doc = global.document;
  var lang = ((doc.documentElement.getAttribute("lang") || "pt").slice(0, 2) || "pt").toLowerCase();
  var T = {
    pt: { open: "Ajuda", title: "Assistente do Ouviescrevi", hello: "Olá! Pergunta-me como transcrever, que formatos aceitamos, limites, privacidade…", ph: "Escreve a tua pergunta…", send: "Enviar", close: "Fechar", thinking: "A pensar… (a primeira resposta pode demorar até 1 minuto)", err: "Não consegui responder agora. Tenta outra vez ou escreve para ouviescrevi@gmail.com.", note: "Respostas da IA: podem ter erros. Não escrevas dados pessoais." },
    en: { open: "Help", title: "Ouviescrevi assistant", hello: "Hi! Ask me how to transcribe, which formats we accept, limits, privacy…", ph: "Type your question…", send: "Send", close: "Close", thinking: "Thinking… (the first answer can take up to 1 minute)", err: "I couldn't answer right now. Try again or email ouviescrevi@gmail.com.", note: "AI answers may contain mistakes. Don't share personal data." },
    es: { open: "Ayuda", title: "Asistente de Ouviescrevi", hello: "¡Hola! Pregúntame cómo transcribir, qué formatos aceptamos, límites, privacidad…", ph: "Escribe tu pregunta…", send: "Enviar", close: "Cerrar", thinking: "Pensando… (la primera respuesta puede tardar hasta 1 minuto)", err: "No he podido responder ahora. Inténtalo de nuevo o escribe a ouviescrevi@gmail.com.", note: "Respuestas de IA: pueden tener errores. No escribas datos personales." },
    fr: { open: "Aide", title: "Assistant Ouviescrevi", hello: "Bonjour ! Demandez-moi comment transcrire, les formats acceptés, les limites, la confidentialité…", ph: "Écrivez votre question…", send: "Envoyer", close: "Fermer", thinking: "Réflexion… (la première réponse peut prendre jusqu'à 1 minute)", err: "Je n'ai pas pu répondre. Réessayez ou écrivez à ouviescrevi@gmail.com.", note: "Réponses de l'IA : elles peuvent contenir des erreurs. N'écrivez pas de données personnelles." },
    de: { open: "Hilfe", title: "Ouviescrevi-Assistent", hello: "Hallo! Frag mich, wie man transkribiert, welche Formate wir annehmen, Limits, Datenschutz…", ph: "Deine Frage…", send: "Senden", close: "Schließen", thinking: "Denke nach… (die erste Antwort kann bis zu 1 Minute dauern)", err: "Ich konnte gerade nicht antworten. Versuch es erneut oder schreib an ouviescrevi@gmail.com.", note: "KI-Antworten können Fehler enthalten. Gib keine persönlichen Daten ein." }
  };
  if (!T[lang]) lang = "pt";   // língua desconhecida: português
  var L = T[lang];

  var css =
    ".oe-help-fab{position:fixed;right:16px;bottom:16px;z-index:9990;display:inline-flex;align-items:center;gap:8px;border:0;border-radius:999px;" +
    "padding:12px 18px;background:var(--oe-purple,#1d4a8f);color:#fff;font:600 15px/1 system-ui,sans-serif;box-shadow:0 8px 24px rgba(15,23,42,.25);cursor:pointer}" +
    ".oe-help-fab:focus-visible,.oe-help-panel button:focus-visible,.oe-help-panel textarea:focus-visible{outline:3px solid var(--oe-gradient,#fd8c32);outline-offset:2px}" +
    ".oe-help-panel{position:fixed;right:16px;bottom:76px;z-index:9991;width:min(380px,calc(100vw - 32px));max-height:min(560px,calc(100vh - 110px));" +
    "display:flex;flex-direction:column;background:var(--oe-surface,#fff);color:var(--oe-text,#1e293b);border:1px solid var(--oe-border,#e2e8f0);" +
    "border-radius:16px;box-shadow:0 10px 40px rgba(15,23,42,.18);font:15px/1.45 system-ui,sans-serif;overflow:hidden}" +
    ".oe-help-panel[hidden]{display:none}" +
    ".oe-help-head{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 14px;background:var(--oe-purple-dark,#0f2a4f);color:#fff}" +
    ".oe-help-head h2{margin:0;font-size:15px;color:#fff}" +
    ".oe-help-head button{border:0;background:transparent;color:#fff;font-size:20px;line-height:1;cursor:pointer;padding:4px 8px}" +
    ".oe-help-log{flex:1;overflow:auto;padding:12px 14px;display:flex;flex-direction:column;gap:8px}" +
    ".oe-help-msg{padding:8px 12px;border-radius:12px;max-width:88%;white-space:pre-wrap;word-wrap:break-word}" +
    ".oe-help-msg.bot{background:color-mix(in srgb,var(--oe-purple,#1d4a8f) 10%,var(--oe-surface,#fff));align-self:flex-start}.oe-help-msg.me{background:var(--oe-purple,#1d4a8f);color:#fff;align-self:flex-end}" +
    ".oe-help-status{padding:0 14px 6px;color:var(--oe-text-muted,#64748b);font-size:13px;min-height:1em}" +
    ".oe-help-form{display:flex;gap:8px;padding:10px 14px;border-top:1px solid var(--oe-border,#e2e8f0)}" +
    ".oe-help-form textarea{flex:1;resize:none;min-height:40px;max-height:96px;padding:8px 10px;border:1px solid var(--oe-border,#e2e8f0);border-radius:10px;font:inherit;color:inherit;background:inherit}" +
    ".oe-help-form button{border:0;border-radius:10px;padding:0 14px;background:var(--oe-gradient,#fd8c32);color:#1e293b;font-weight:700;cursor:pointer}" +
    ".oe-help-form button[disabled]{opacity:.6;cursor:wait}" +
    ".oe-help-note{margin:0;padding:0 14px 10px;color:var(--oe-text-muted,#64748b);font-size:13px}" +
    
    "@media (prefers-reduced-motion:no-preference){.oe-help-fab{transition:transform .15s}.oe-help-fab:hover{transform:translateY(-2px)}}";

  function el(tag, cls, text) {
    var n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function ensureApi() {
    if (global.OuviescreviAPI) return Promise.resolve();
    return new Promise(function (resolve) {
      var s = doc.createElement("script");
      s.src = "/js/ouviescrevi-api.js?v=28";
      s.onload = s.onerror = function () { resolve(); };
      doc.head.appendChild(s);
    });
  }

  function mount() {
    if (doc.getElementById("oeHelpHost")) return;
    // Numa raiz separada (shadow DOM): as regras gerais de botões do site (.oe-page button…) não mexem na bolha.
    var host = el("div");
    host.id = "oeHelpHost";
    var root = host.attachShadow ? host.attachShadow({ mode: "open" }) : host;
    var style = el("style");
    style.textContent = css;
    root.appendChild(style);
    var up = !!doc.getElementById("fabSugestao");   // na página inicial há o botão de sugestões no mesmo canto

    var fab = el("button", "oe-help-fab");
    fab.type = "button";
    fab.setAttribute("aria-expanded", "false");
    fab.setAttribute("aria-controls", "oeHelpPanel");
    fab.append(el("span", "", "💬"), el("span", "", L.open));

    var panel = el("section", "oe-help-panel");
    panel.id = "oeHelpPanel";
    panel.hidden = true;
    panel.setAttribute("aria-label", L.title);
    var head = el("div", "oe-help-head");
    var close = el("button", "", "×");
    close.type = "button";
    close.setAttribute("aria-label", L.close);
    head.append(el("h2", "", L.title), close);
    var log = el("div", "oe-help-log");
    log.setAttribute("role", "log");
    log.setAttribute("aria-live", "polite");
    var status = el("div", "oe-help-status");
    var form = el("form", "oe-help-form");
    var input = el("textarea");
    input.rows = 1;
    input.maxLength = 500;
    input.placeholder = L.ph;
    input.setAttribute("aria-label", L.ph);
    var send = el("button", "", L.send);
    send.type = "submit";
    form.append(input, send);
    panel.append(head, log, status, form, el("p", "oe-help-note", L.note));
    root.append(panel, fab);
    doc.body.appendChild(host);
    // Sobe por cima do aviso de cookies (enquanto estiver aberto) e do botão de sugestões da página inicial.
    function place() {
      var banner = doc.querySelector(".oe-cookie-banner");
      var extra = banner && !banner.hidden && banner.offsetHeight ? banner.offsetHeight : 0;
      var base = (up ? 84 : 16) + extra;
      fab.style.bottom = base + "px";
      panel.style.bottom = base + 60 + "px";
    }
    place();
    global.addEventListener("resize", place);
    doc.addEventListener("click", function () { global.setTimeout(place, 50); }, true);

    var history = [];
    var busy = false;
    function say(role, text) {
      var m = el("div", "oe-help-msg " + (role === "user" ? "me" : "bot"), text);
      log.appendChild(m);
      log.scrollTop = log.scrollHeight;
    }
    say("assistant", L.hello);

    function toggle(open) {
      panel.hidden = !open;
      fab.setAttribute("aria-expanded", String(open));
      if (open) input.focus(); else fab.focus();
    }
    fab.addEventListener("click", function () { toggle(panel.hidden); });
    close.addEventListener("click", function () { toggle(false); });
    panel.addEventListener("keydown", function (e) { if (e.key === "Escape") toggle(false); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit")); }
    });

    form.addEventListener("submit", async function (e) {
      e.preventDefault();
      var question = (input.value || "").trim();
      if (!question || busy) return;
      busy = true;
      send.disabled = true;
      input.value = "";
      say("user", question);
      status.textContent = L.thinking;
      try {
        await ensureApi();
        var api = global.OuviescreviAPI;
        await api.init();
        var res = await fetch(api.getBase() + "/help-chat", {
          method: "POST",
          headers: api.authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(api.authJson({ question: question, history: history.slice(-6), lang: lang }))
        });
        var data = {};
        try { data = await res.json(); } catch (_) {}
        if (!res.ok) {
          var detail = data.detail || L.err;
          if (Array.isArray(detail)) detail = L.err;
          say("assistant", String(detail));
        } else {
          var answer = (data.answer || "").trim() || L.err;
          say("assistant", answer);
          history.push({ role: "user", content: question }, { role: "assistant", content: answer });
        }
      } catch (_) {
        say("assistant", L.err);
      } finally {
        status.textContent = "";
        busy = false;
        send.disabled = false;
        input.focus();
      }
    });
  }

  global.OuviescreviHelp = { mount: mount };
  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", mount);
  else mount();
})(window);
