/**
 * Assistente Aula completa — transcrição → resumo → perguntas → flashcards.
 */
(function (global) {
  var STORAGE_KEY = "oe_aula_completa_text";
  var config = { lang: "pt" };
  var state = { step: 1, text: "", summary: "", questions: "", flashcards: null };

  var STRINGS = {
    pt: {
      step1: "1. Transcrição",
      step2: "2. Resumo",
      step3: "3. Perguntas",
      step4: "4. Flashcards",
      step1Title: "Cola a transcrição ou apontamentos",
      step1Hint: "Podes colar diretamente do Ouviescrevi após transcrever.",
      placeholder: "Cola aqui o texto da aula…",
      langLabel: "Idioma",
      numQuestions: "N.º perguntas",
      numCards: "N.º cartões",
      btnSummary: "🧠 Gerar resumo",
      btnQuestions: "📘 Gerar perguntas",
      btnFlashcards: "🃏 Gerar flashcards",
      btnBack: "← Voltar",
      loadingSummary: "A gerar resumo…",
      loadingQuestions: "A gerar perguntas…",
      loadingFlashcards: "A gerar flashcards…",
      needText: "Introduz texto (mín. ~120 caracteres).",
      error: "Erro ao processar.",
      step2Title: "Resumo",
      step3Title: "Perguntas de revisão",
      step4Title: "Flashcards",
      copySummary: "Copiar resumo",
      copyQuestions: "Copiar perguntas",
      copyAllCards: "Copiar cartões",
      exportAnki: "Exportar Anki",
      exportedAnki: "Ficheiro Anki descarregado!",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      flipHint: "Clica no cartão para ver a resposta",
      front: "Frente",
      back: "Verso",
      restart: "Começar de novo",
    },
    en: {
      step1: "1. Transcript",
      step2: "2. Summary",
      step3: "3. Questions",
      step4: "4. Flashcards",
      step1Title: "Paste transcript or notes",
      step1Hint: "You can paste directly from Ouviescrevi after transcribing.",
      placeholder: "Paste lesson text here…",
      langLabel: "Language",
      numQuestions: "Questions",
      numCards: "Cards",
      btnSummary: "🧠 Generate summary",
      btnQuestions: "📘 Generate questions",
      btnFlashcards: "🃏 Generate flashcards",
      btnBack: "← Back",
      loadingSummary: "Generating summary…",
      loadingQuestions: "Generating questions…",
      loadingFlashcards: "Generating flashcards…",
      needText: "Paste some text first (min. ~120 characters).",
      error: "Processing error.",
      step2Title: "Summary",
      step3Title: "Revision questions",
      step4Title: "Flashcards",
      copySummary: "Copy summary",
      copyQuestions: "Copy questions",
      copyAllCards: "Copy cards",
      exportAnki: "Export Anki",
      exportedAnki: "Anki file downloaded!",
      copied: "Copied!",
      copyFail: "Could not copy.",
      flipHint: "Click a card to reveal the answer",
      front: "Front",
      back: "Back",
      restart: "Start over",
    },
    es: {
      step1: "1. Transcripción",
      step2: "2. Resumen",
      step3: "3. Preguntas",
      step4: "4. Flashcards",
      step1Title: "Pega la transcripción o apuntes",
      step1Hint: "Puedes pegar directamente desde Ouviescrevi.",
      placeholder: "Pega aquí el texto de la clase…",
      langLabel: "Idioma",
      numQuestions: "Preguntas",
      numCards: "Tarjetas",
      btnSummary: "🧠 Generar resumen",
      btnQuestions: "📘 Generar preguntas",
      btnFlashcards: "🃏 Generar flashcards",
      btnBack: "← Volver",
      loadingSummary: "Generando resumen…",
      loadingQuestions: "Generando preguntas…",
      loadingFlashcards: "Generando flashcards…",
      needText: "Introduce texto (mín. ~120 caracteres).",
      error: "Error al procesar.",
      step2Title: "Resumen",
      step3Title: "Preguntas de repaso",
      step4Title: "Flashcards",
      copySummary: "Copiar resumen",
      copyQuestions: "Copiar preguntas",
      copyAllCards: "Copiar tarjetas",
      exportAnki: "Exportar Anki",
      exportedAnki: "¡Archivo Anki descargado!",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
      flipHint: "Haz clic para ver la respuesta",
      front: "Anverso",
      back: "Reverso",
      restart: "Empezar de nuevo",
    },
    fr: {
      step1: "1. Transcription",
      step2: "2. Résumé",
      step3: "3. Questions",
      step4: "4. Flashcards",
      step1Title: "Collez la transcription ou les notes",
      step1Hint: "Vous pouvez coller directement depuis Ouviescrevi.",
      placeholder: "Collez ici le texte du cours…",
      langLabel: "Langue",
      numQuestions: "Questions",
      numCards: "Cartes",
      btnSummary: "🧠 Générer le résumé",
      btnQuestions: "📘 Générer les questions",
      btnFlashcards: "🃏 Générer les flashcards",
      btnBack: "← Retour",
      loadingSummary: "Génération du résumé…",
      loadingQuestions: "Génération des questions…",
      loadingFlashcards: "Génération des flashcards…",
      needText: "Saisissez du texte (min. ~120 caractères).",
      error: "Erreur de traitement.",
      step2Title: "Résumé",
      step3Title: "Questions de révision",
      step4Title: "Flashcards",
      copySummary: "Copier le résumé",
      copyQuestions: "Copier les questions",
      copyAllCards: "Copier les cartes",
      exportAnki: "Exporter Anki",
      exportedAnki: "Fichier Anki téléchargé !",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
      flipHint: "Cliquez pour voir la réponse",
      front: "Recto",
      back: "Verso",
      restart: "Recommencer",
    },
    de: {
      step1: "1. Transkript",
      step2: "2. Zusammenfassung",
      step3: "3. Fragen",
      step4: "4. Karteikarten",
      step1Title: "Transkript oder Notizen einfügen",
      step1Hint: "Direkt aus Ouviescrevi nach der Transkription einfügen.",
      placeholder: "Unterrichtstext hier einfügen…",
      langLabel: "Sprache",
      numQuestions: "Fragen",
      numCards: "Karten",
      btnSummary: "🧠 Zusammenfassung generieren",
      btnQuestions: "📘 Fragen generieren",
      btnFlashcards: "🃏 Karteikarten generieren",
      btnBack: "← Zurück",
      loadingSummary: "Zusammenfassung wird erstellt…",
      loadingQuestions: "Fragen werden erstellt…",
      loadingFlashcards: "Karteikarten werden erstellt…",
      needText: "Text eingeben (min. ~120 Zeichen).",
      error: "Verarbeitungsfehler.",
      step2Title: "Zusammenfassung",
      step3Title: "Wiederholungsfragen",
      step4Title: "Karteikarten",
      copySummary: "Zusammenfassung kopieren",
      copyQuestions: "Fragen kopieren",
      copyAllCards: "Karten kopieren",
      exportAnki: "Anki exportieren",
      exportedAnki: "Anki-Datei heruntergeladen!",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
      flipHint: "Karte anklicken für die Antwort",
      front: "Vorderseite",
      back: "Rückseite",
      restart: "Neu starten",
    },
  };

  function t(key) {
    var loc = STRINGS[config.lang] || STRINGS.pt;
    return loc[key] || STRINGS.pt[key] || key;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function copyText(text) {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      return;
    }
    navigator.clipboard.writeText(text).then(
      function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copied"), "success");
      },
      function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      }
    );
  }

  function cardsToPlain(data) {
    var lines = [];
    if (data.title) lines.push(data.title, "");
    (data.cards || []).forEach(function (c, i) {
      lines.push(i + 1 + ". " + c.front);
      lines.push("   → " + c.back);
      lines.push("");
    });
    return lines.join("\n").trim();
  }

  function updateSteps() {
    document.querySelectorAll(".oe-pw-steps li").forEach(function (li, i) {
      var n = i + 1;
      li.classList.remove("is-active", "is-done");
      li.removeAttribute("aria-current");
      if (n < state.step) li.classList.add("is-done");
      if (n === state.step) {
        li.classList.add("is-active");
        li.setAttribute("aria-current", "step");
      }
    });
    ["acStep1", "acStep2", "acStep3", "acStep4"].forEach(function (id, i) {
      var el = document.getElementById(id);
      if (el) el.hidden = state.step !== i + 1;
    });
  }

  function getLang() {
    var el = document.getElementById("acLang");
    return el ? el.value : config.lang;
  }

  function cardsToAnki(data) {
    var lines = ["#separator:tab", "#html:true", "#columns:Front\tBack"];
    (data.cards || []).forEach(function (c) {
      var front = String(c.front || "")
        .replace(/\t/g, " ")
        .replace(/\r?\n/g, "<br>");
      var back = String(c.back || "")
        .replace(/\t/g, " ")
        .replace(/\r?\n/g, "<br>");
      lines.push(front + "\t" + back);
    });
    return lines.join("\n");
  }

  function downloadAnki(data) {
    if (!data || !data.cards || !data.cards.length) return;
    var base = (data.title || "flashcards-ouviescrevi")
      .replace(/[^\w\s-áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .toLowerCase();
    var blob = new Blob([cardsToAnki(data)], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = (base || "flashcards-ouviescrevi") + ".txt";
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 500);
    if (global.OuviescreviUI) global.OuviescreviUI.toast(t("exportedAnki"), "success");
  }

  function renderFlashcards(container, data) {
    var cards = (data.cards || [])
      .map(function (c, i) {
        return (
          '<button type="button" class="oe-fc-card" aria-pressed="false" aria-label="' +
          escapeHtml(t("card") + " " + (i + 1)) +
          '">' +
          '<span class="oe-fc-card__inner">' +
          '<span class="oe-fc-card__face oe-fc-card__face--front">' +
          '<span class="oe-fc-card__label">' +
          escapeHtml(t("front")) +
          "</span><span class=\"oe-fc-card__text\">" +
          escapeHtml(c.front) +
          "</span></span>" +
          '<span class="oe-fc-card__face oe-fc-card__face--back">' +
          '<span class="oe-fc-card__label">' +
          escapeHtml(t("back")) +
          "</span><span class=\"oe-fc-card__text\">" +
          escapeHtml(c.back) +
          "</span></span></span></button>"
        );
      })
      .join("");

    container.innerHTML =
      '<div class="oe-fc-result">' +
      '<header class="oe-fc-result__head">' +
      "<h2 class=\"oe-fc-result__title\">" +
      escapeHtml(data.title || t("step4Title")) +
      "</h2>" +
      '<div class="oe-fc-result__actions">' +
      '<button type="button" class="oe-fc-result__btn" id="btnAcCopyCards">' +
      escapeHtml(t("copyAllCards")) +
      "</button>" +
      '<button type="button" class="oe-fc-result__btn oe-fc-result__btn--secondary" id="btnAcAnki">' +
      escapeHtml(t("exportAnki")) +
      "</button>" +
      '<button type="button" class="oe-fc-result__btn oe-fc-result__btn--secondary" id="btnAcPrint">' +
      escapeHtml(global.FlashcardsExport ? global.FlashcardsExport.label(config.lang) : "Imprimir / PDF") +
      "</button></div></header>" +
      '<p class="oe-fc-hint">' +
      escapeHtml(t("flipHint")) +
      "</p>" +
      '<div class="oe-fc-grid">' +
      cards +
      "</div>" +
      '<button type="button" class="oe-pw-btn oe-pw-btn--ghost" id="btnAcRestart">' +
      escapeHtml(t("restart")) +
      "</button></div>";

    if (global.FlashcardsUI && global.FlashcardsUI.bindFlashcardCards) {
      global.FlashcardsUI.bindFlashcardCards(container, t("card"));
    } else {
      container.querySelectorAll(".oe-fc-card").forEach(function (btn) {
        btn.addEventListener("click", function () {
          btn.classList.toggle("is-flipped");
        });
      });
    }
    var copyBtn = document.getElementById("btnAcCopyCards");
    if (copyBtn) copyBtn.addEventListener("click", function () {
      copyText(cardsToPlain(data));
    });
    var ankiBtn = document.getElementById("btnAcAnki");
    if (ankiBtn) ankiBtn.addEventListener("click", function () {
      downloadAnki(data);
    });
    var printBtn = document.getElementById("btnAcPrint");
    if (printBtn && global.FlashcardsExport) {
      printBtn.addEventListener("click", function () {
        global.FlashcardsExport.open(data, config.lang);
      });
    }
    var restart = document.getElementById("btnAcRestart");
    if (restart) {
      restart.addEventListener("click", function () {
        state = { step: 1, text: "", summary: "", questions: "", flashcards: null };
        var ta = document.getElementById("texto");
        if (ta) ta.value = "";
        updateSteps();
      });
    }
  }

  async function generateSummary() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnAcSummary");
    if (!textoEl || !btn) return;
    var texto = textoEl.value.trim();
    if (texto.length < 120) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }
    state.text = texto;
    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loadingSummary"));
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/summarize", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI.authJson({ text: texto, lang: getLang(), mode: "normal" })
        ),
      });
      var data = await res.json();
      if (!res.ok) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(data.detail || t("error"), "error");
        return;
      }
      state.summary = data.summary || "";
      state.step = 2;
      updateSteps();
      var pre = document.getElementById("acSummaryOut");
      if (pre) pre.textContent = state.summary;
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  async function generateQuestions() {
    var btn = document.getElementById("btnAcQuestions");
    if (!btn || !state.summary) return;
    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loadingQuestions"));
    try {
      await global.OuviescreviAPI.init();
      var nEl = document.getElementById("acNumQuestions");
      var payload = {
        text: state.text,
        lang: getLang(),
        num_questions: nEl ? parseInt(nEl.value, 10) || 10 : 10,
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-questions", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(data.detail || t("error"), "error");
        return;
      }
      state.questions = data.questions || "";
      state.step = 3;
      updateSteps();
      var pre = document.getElementById("acQuestionsOut");
      if (pre) pre.textContent = state.questions;
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  async function generateFlashcards() {
    var btn = document.getElementById("btnAcFlashcards");
    var out = document.getElementById("acFlashcardsOut");
    if (!btn || !out) return;
    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loadingFlashcards"));
    try {
      await global.OuviescreviAPI.init();
      var cEl = document.getElementById("acNumCards");
      var payload = {
        text: state.text,
        lang: getLang(),
        num_cards: cEl ? parseInt(cEl.value, 10) || 15 : 15,
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-flashcards", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(data.detail || t("error"), "error");
        return;
      }
      state.flashcards = data;
      state.step = 4;
      updateSteps();
      renderFlashcards(out, data);
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function applyLabels() {
    var map = {
      acStep1Label: "step1",
      acStep2Label: "step2",
      acStep3Label: "step3",
      acStep4Label: "step4",
      acStep1Title: "step1Title",
      acStep1Hint: "step1Hint",
      acLangLabel: "langLabel",
      acNumQuestionsLabel: "numQuestions",
      acNumCardsLabel: "numCards",
      acStep2Title: "step2Title",
      acStep3Title: "step3Title",
      acStep4Title: "step4Title",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });
    var ta = document.getElementById("texto");
    if (ta) ta.placeholder = t("placeholder");
    var ids = {
      btnAcSummary: "btnSummary",
      btnAcQuestions: "btnQuestions",
      btnAcFlashcards: "btnFlashcards",
      btnAcBack2: "btnBack",
      btnAcBack3: "btnBack",
      btnAcCopySummary: "copySummary",
      btnAcCopyQuestions: "copyQuestions",
    };
    Object.keys(ids).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(ids[id]);
    });
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyLabels();
    updateSteps();

    var s = document.getElementById("btnAcSummary");
    if (s) s.addEventListener("click", generateSummary);
    var q = document.getElementById("btnAcQuestions");
    if (q) q.addEventListener("click", generateQuestions);
    var f = document.getElementById("btnAcFlashcards");
    if (f) f.addEventListener("click", generateFlashcards);
    var b2 = document.getElementById("btnAcBack2");
    if (b2) b2.addEventListener("click", function () {
      state.step = 1;
      updateSteps();
    });
    var b3 = document.getElementById("btnAcBack3");
    if (b3) b3.addEventListener("click", function () {
      state.step = 2;
      updateSteps();
    });
    var cs = document.getElementById("btnAcCopySummary");
    if (cs) cs.addEventListener("click", function () {
      copyText(state.summary);
    });
    var cq = document.getElementById("btnAcCopyQuestions");
    if (cq) cq.addEventListener("click", function () {
      copyText(state.questions);
    });

    try {
      var saved = sessionStorage.getItem(STORAGE_KEY);
      var ta = document.getElementById("texto");
      if (saved && ta && !ta.value.trim()) {
        ta.value = saved;
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {}
  }

  global.AulaCompletaUI = { init: init };
})(typeof window !== "undefined" ? window : this);
