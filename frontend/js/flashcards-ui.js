/**
 * Flashcards — cartões frente/verso com IA.
 */
(function (global) {
  var STORAGE_KEY = "oe_flashcards_text";
  var config = { lang: "pt" };
  var lastData = null;

  var STRINGS = {
    pt: {
      placeholder: "Cola aqui o texto da aula ou transcrição…",
      btn: "🃏 Gerar flashcards",
      loading: "A gerar flashcards…",
      needText: "Introduz texto (mín. ~80 caracteres).",
      error: "Erro ao gerar flashcards.",
      truncated: "O texto foi truncado — os cartões baseiam-se no início do conteúdo.",
      copyAll: "Copiar tudo",
      exportAnki: "Exportar Anki",
      exportedAnki: "Ficheiro Anki descarregado!",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      flipHint: "Clica no cartão para ver a resposta",
      front: "Frente",
      back: "Verso",
      card: "Cartão",
    },
    en: {
      placeholder: "Paste lesson text or transcript here…",
      btn: "🃏 Generate flashcards",
      loading: "Generating flashcards…",
      needText: "Paste some text first (min. ~80 characters).",
      error: "Error generating flashcards.",
      truncated: "Text was truncated — cards are based on the beginning.",
      copyAll: "Copy all",
      exportAnki: "Export Anki",
      exportedAnki: "Anki file downloaded!",
      copied: "Copied!",
      copyFail: "Could not copy.",
      flipHint: "Click a card to reveal the answer",
      front: "Front",
      back: "Back",
      card: "Card",
    },
    es: {
      placeholder: "Pega aquí el texto de la clase o transcripción…",
      btn: "🃏 Generar flashcards",
      loading: "Generando flashcards…",
      needText: "Introduce texto (mín. ~80 caracteres).",
      error: "Error al generar flashcards.",
      truncated: "El texto fue truncado — las tarjetas se basan en el inicio.",
      copyAll: "Copiar todo",
      exportAnki: "Exportar Anki",
      exportedAnki: "¡Archivo Anki descargado!",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
      flipHint: "Haz clic en la tarjeta para ver la respuesta",
      front: "Anverso",
      back: "Reverso",
      card: "Tarjeta",
    },
    fr: {
      placeholder: "Collez ici le texte du cours ou la transcription…",
      btn: "🃏 Générer des flashcards",
      loading: "Génération des flashcards…",
      needText: "Saisissez du texte (min. ~80 caractères).",
      error: "Erreur lors de la génération.",
      truncated: "Texte tronqué — les cartes sont basées sur le début.",
      copyAll: "Tout copier",
      exportAnki: "Exporter Anki",
      exportedAnki: "Fichier Anki téléchargé !",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
      flipHint: "Cliquez sur une carte pour voir la réponse",
      front: "Recto",
      back: "Verso",
      card: "Carte",
    },
    de: {
      placeholder: "Fügen Sie hier Unterrichtstext oder Transkript ein…",
      btn: "🃏 Karteikarten generieren",
      loading: "Karteikarten werden erstellt…",
      needText: "Text eingeben (min. ~80 Zeichen).",
      error: "Fehler beim Generieren.",
      truncated: "Text gekürzt — Karten basieren auf dem Anfang.",
      copyAll: "Alles kopieren",
      exportAnki: "Anki exportieren",
      exportedAnki: "Anki-Datei heruntergeladen!",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
      flipHint: "Karte anklicken, um die Antwort zu sehen",
      front: "Vorderseite",
      back: "Rückseite",
      card: "Karte",
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

  function cardsToPlain(data) {
    var lines = [];
    if (data.title) lines.push(data.title, "");
    (data.cards || []).forEach(function (c, i) {
      lines.push((i + 1) + ". " + c.front);
      lines.push("   → " + c.back);
      lines.push("");
    });
    return lines.join("\n").trim();
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

  function downloadTextFile(content, filename) {
    var blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 500);
  }

  function exportAnki(data) {
    if (!data || !data.cards || !data.cards.length) return;
    var base = (data.title || "flashcards-ouviescrevi")
      .replace(/[^\w\s-áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .toLowerCase();
    downloadTextFile(cardsToAnki(data), (base || "flashcards-ouviescrevi") + ".txt");
    if (global.OuviescreviUI) global.OuviescreviUI.toast(t("exportedAnki"), "success");
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

  function renderResult(container, data, truncated) {
    lastData = data;
    var cards = (data.cards || [])
      .map(function (c) {
        return (
          '<button type="button" class="oe-fc-card" aria-label="' +
          escapeHtml(t("card") + " " + c.index) +
          '">' +
          '<span class="oe-fc-card__inner">' +
          '<span class="oe-fc-card__face oe-fc-card__face--front">' +
          '<span class="oe-fc-card__label">' +
          escapeHtml(t("front")) +
          "</span>" +
          '<span class="oe-fc-card__text">' +
          escapeHtml(c.front) +
          "</span></span>" +
          '<span class="oe-fc-card__face oe-fc-card__face--back">' +
          '<span class="oe-fc-card__label">' +
          escapeHtml(t("back")) +
          "</span>" +
          '<span class="oe-fc-card__text">' +
          escapeHtml(c.back) +
          "</span></span></span></button>"
        );
      })
      .join("");

    container.innerHTML =
      '<div class="oe-fc-result">' +
      (truncated ? '<p class="oe-fc-warn">' + escapeHtml(t("truncated")) + "</p>" : "") +
      '<header class="oe-fc-result__head">' +
      '<h2 class="oe-fc-result__title">' +
      escapeHtml(data.title || t("card")) +
      "</h2>" +
      '<div class="oe-fc-result__actions">' +
      '<button type="button" class="oe-fc-result__btn" id="btnFcCopy">' +
      escapeHtml(t("copyAll")) +
      "</button>" +
      '<button type="button" class="oe-fc-result__btn oe-fc-result__btn--secondary" id="btnFcAnki">' +
      escapeHtml(t("exportAnki")) +
      "</button></div></header>" +
      '<p class="oe-fc-hint">' +
      escapeHtml(t("flipHint")) +
      "</p>" +
      '<div class="oe-fc-grid">' +
      cards +
      "</div></div>";

    container.hidden = false;
    container.querySelectorAll(".oe-fc-card").forEach(function (btn) {
      btn.addEventListener("click", function () {
        btn.classList.toggle("is-flipped");
      });
    });
    var copyBtn = document.getElementById("btnFcCopy");
    if (copyBtn) {
      copyBtn.addEventListener("click", function () {
        if (lastData) copyText(cardsToPlain(lastData));
      });
    }
    var ankiBtn = document.getElementById("btnFcAnki");
    if (ankiBtn) {
      ankiBtn.addEventListener("click", function () {
        if (lastData) exportAnki(lastData);
      });
    }
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function generate() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnFlashcards");
    var out = document.getElementById("resultado");
    if (!textoEl || !btn || !out) return;

    var texto = textoEl.value.trim();
    if (texto.length < 80) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));

    try {
      await global.OuviescreviAPI.init();
      var langEl = document.getElementById("fcLang");
      var countEl = document.getElementById("fcCount");
      var payload = {
        text: texto,
        lang: langEl ? langEl.value : config.lang,
        num_cards: countEl ? parseInt(countEl.value, 10) || 15 : 15,
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-flashcards", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        out.innerHTML = "<pre>" + escapeHtml(data.detail || t("error")) + "</pre>";
        out.hidden = false;
        return;
      }
      renderResult(out, data, !!data.truncated);
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    var ta = document.getElementById("texto");
    if (ta) ta.placeholder = t("placeholder");
    var btn = document.getElementById("btnFlashcards");
    if (btn) {
      btn.textContent = t("btn");
      btn.addEventListener("click", generate);
    }
    try {
      var saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved && ta && !ta.value.trim()) {
        ta.value = saved;
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {}
  }

  global.FlashcardsUI = { init: init };
})(typeof window !== "undefined" ? window : this);
