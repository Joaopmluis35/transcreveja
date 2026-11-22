/**
 * Chat "Perguntar à IA sobre esta transcrição" para as páginas EN/ES/FR/DE.
 * A página PT tem a sua própria versão dentro da index.html.
 * shortcut: código duplicado da PT; se a PT mudar muito, passar a PT a usar este ficheiro.
 */
(function () {
  var T = {
    en: {
      title: "💬 Ask the AI about this transcript", hint: "The AI answers only from the text above.",
      label: "Your question", ph: "Type your question…", send: "Send", clear: "Clear",
      sugg: ["Summarise in 3 points", "What decisions were made?", "Who has action items?"],
      thinking: "Thinking…", empty: "⚠️ Empty transcript.", none: "No answer.",
      err: "❌ Error communicating with the API.", http: "Error"
    },
    es: {
      title: "💬 Pregunta a la IA sobre esta transcripción", hint: "La IA responde solo con el texto de arriba.",
      label: "Tu pregunta", ph: "Escribe tu pregunta…", send: "Enviar", clear: "Borrar",
      sugg: ["Resume en 3 puntos", "¿Qué decisiones se tomaron?", "¿Quién tiene tareas?"],
      thinking: "Pensando…", empty: "⚠️ Transcripción vacía.", none: "Sin respuesta.",
      err: "❌ Error al comunicar con la API.", http: "Error"
    },
    fr: {
      title: "💬 Poser une question à l’IA sur cette transcription", hint: "L’IA répond uniquement à partir du texte ci-dessus.",
      label: "Ta question", ph: "Écris ta question…", send: "Envoyer", clear: "Effacer",
      sugg: ["Résume en 3 points", "Quelles décisions ont été prises ?", "Qui a des actions à faire ?"],
      thinking: "Réflexion…", empty: "⚠️ Transcription vide.", none: "Pas de réponse.",
      err: "❌ Erreur de communication avec l’API.", http: "Erreur"
    },
    de: {
      title: "💬 Frag die KI zu diesem Transkript", hint: "Die KI antwortet nur anhand des Textes oben.",
      label: "Deine Frage", ph: "Schreib deine Frage…", send: "Senden", clear: "Leeren",
      sugg: ["In 3 Punkten zusammenfassen", "Welche Entscheidungen wurden getroffen?", "Wer hat Aufgaben?"],
      thinking: "Denkt nach…", empty: "⚠️ Leeres Transkript.", none: "Keine Antwort.",
      err: "❌ Fehler bei der Verbindung zur API.", http: "Fehler"
    }
  };
  var lang = (document.documentElement.lang || "en").slice(0, 2).toLowerCase();
  var t = T[lang] || T.en;

  var css =
    ".oe-tchat{margin-top:20px;padding-top:16px;border-top:1px solid #e5e7eb}" +
    ".oe-tchat h3{margin:0 0 4px;font-size:1.05rem}" +
    ".oe-tchat__hint{margin:0 0 10px;color:#64748b;font-size:.9rem}" +
    ".oe-tchat__row{display:flex;gap:8px;flex-wrap:wrap}" +
    ".oe-tchat__row input{width:auto;margin:0;flex:1 1 220px;min-width:0;padding:10px 12px;border:1px solid #cbd5e1;border-radius:10px;font:inherit}" +
    ".oe-tchat__sugg{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}" +
    ".oe-tchat__sugg button{width:auto;margin:0;padding:6px 12px;font:inherit;font-size:.85rem;border:1px solid #9fb7d6;border-radius:999px;background:#f4f9ff;color:#a94c06;cursor:pointer}" +
    ".oe-tchat__msgs{display:flex;flex-direction:column;gap:8px;max-height:360px;overflow-y:auto}" +
    ".oe-tchat__b{padding:10px 12px;border-radius:12px;white-space:pre-wrap;max-width:90%}" +
    ".oe-tchat__b--user{align-self:flex-end;background:#e4f0fc;color:#0f2a4f}" +
    ".oe-tchat__b--assistant{align-self:flex-start;background:#f1f5f9;color:#0f172a}" +
    ".oe-tchat__status{color:#64748b;font-size:.9rem}" +
    "@media (max-width:600px){.oe-tchat__row button{flex:1 1 100%}}";

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    for (var k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }

  function init() {
    var ta = document.getElementById("transcriptionText");
    if (!ta || document.getElementById("oeTranscriptChat")) return;

    var style = el("style");
    style.textContent = css;
    document.head.appendChild(style);

    var panel = el("section", { id: "oeTranscriptChat", class: "oe-tchat", "aria-label": t.title, hidden: "" });
    panel.appendChild(el("h3", null, t.title));
    panel.appendChild(el("p", { class: "oe-tchat__hint" }, t.hint));
    var form = el("form", { class: "oe-tchat__row" });
    var input = el("input", { type: "text", maxlength: "2000", autocomplete: "off", placeholder: t.ph, "aria-label": t.label });
    var send = el("button", { type: "submit" }, t.send);
    var clear = el("button", { type: "button" }, t.clear);
    form.appendChild(input); form.appendChild(send); form.appendChild(clear);
    panel.appendChild(form);
    var sugg = el("div", { class: "oe-tchat__sugg" });
    t.sugg.forEach(function (q) { // a classe oe-transcript-chat__chip já está excluída do estilo genérico de botões (ouviescrevi.css)
      sugg.appendChild(el("button", { type: "button", class: "oe-transcript-chat__chip", "data-q": q }, q)); });
    panel.appendChild(sugg);
    var msgs = el("div", { class: "oe-tchat__msgs", role: "log", "aria-live": "polite" });
    var status = el("p", { class: "oe-tchat__status", hidden: "" });
    panel.appendChild(msgs); panel.appendChild(status);
    (ta.nextElementSibling || ta).insertAdjacentElement("afterend", panel);

    var history = [];
    var busy = false;
    var lastText = "";

    function bubble(role, content) {
      var b = el("div", { class: "oe-tchat__b oe-tchat__b--" + role }, content);
      msgs.appendChild(b);
      msgs.scrollTop = msgs.scrollHeight;
    }
    function setStatus(s) { status.hidden = !s; status.textContent = s || ""; }
    function reset() { history = []; msgs.textContent = ""; input.value = ""; setStatus(""); }

    // A transcrição é escrita por código (sem evento "input"), por isso verifica-se de segundo a segundo.
    setInterval(function () {
      var text = (ta.value || "").trim();
      panel.hidden = !text;
      if (text !== lastText) { if (lastText) reset(); lastText = text; }
    }, 1000);

    async function ask(question) {
      question = (question || "").trim();
      if (!question || busy) return;
      var text = (ta.value || "").trim();
      if (!text) { alert(t.empty); return; }
      busy = true;
      send.disabled = true;
      bubble("user", question);
      history.push({ role: "user", content: question });
      setStatus(t.thinking);
      try {
        await OuviescreviAPI.init();
        var res = await fetch(OuviescreviAPI.getBase() + "/transcript-chat", {
          method: "POST",
          headers: OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(OuviescreviAPI.authJson({
            text: text, question: question, history: history.slice(0, -1).slice(-6), lang: lang
          }))
        });
        var data = {};
        try { data = await res.json(); } catch (_) {}
        if (!res.ok) {
          var detail = data.detail || data.error || (t.http + " " + res.status);
          if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg || d; }).join(" ");
          bubble("assistant", "❌ " + detail);
          return;
        }
        var answer = (data.answer || "").trim() || t.none;
        bubble("assistant", answer);
        history.push({ role: "assistant", content: answer });
      } catch (_) {
        bubble("assistant", t.err);
      } finally {
        setStatus("");
        busy = false;
        send.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", function (e) { e.preventDefault(); var q = input.value; input.value = ""; ask(q); });
    clear.addEventListener("click", function () { reset(); input.focus(); });
    sugg.addEventListener("click", function (e) {
      var b = e.target.closest("[data-q]");
      if (b) ask(b.getAttribute("data-q"));
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
