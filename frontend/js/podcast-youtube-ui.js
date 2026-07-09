/**
 * Assistente Podcast & YouTube — transcrição → capítulos → descrição.
 */
(function (global) {
  var STORAGE_KEY = "oe_podcast_wizard_text";
  var config = { lang: "pt" };
  var state = { step: 1, text: "", chaptersData: null, youtubeData: null };

  var STRINGS = {
    pt: {
      step1: "1. Transcrição",
      step2: "2. Capítulos",
      step3: "3. Descrição YouTube",
      step1Title: "Cola a transcrição",
      step1Hint: "Funciona melhor com timestamps [MM:SS] — como a transcrição do Ouviescrevi.",
      placeholder: "Cola aqui a transcrição com timestamps…",
      titleHint: "Título do episódio (opcional)",
      titlePlaceholder: "Ex.: Ep. 12 — Entrevista com…",
      langLabel: "Idioma",
      maxChapters: "Máx. capítulos",
      btnChapters: "⏱️ Gerar capítulos",
      loadingChapters: "A gerar capítulos…",
      needText: "Introduz texto (mín. ~120 caracteres).",
      error: "Erro ao processar.",
      truncated: "O texto foi truncado.",
      noTimestamps: "Sem timestamps detectados — os horários podem ser estimados.",
      step2Title: "Capítulos gerados",
      chaptersPreview: "Pré-visualização",
      btnYoutube: "▶️ Gerar descrição YouTube",
      btnBack: "← Voltar",
      loadingYoutube: "A gerar descrição…",
      step3Title: "Descrição para YouTube",
      titles: "Títulos sugeridos",
      description: "Descrição",
      tags: "Tags",
      copyDesc: "Copiar descrição",
      copyTags: "Copiar tags",
      copyTitle: "Copiar título",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      restart: "Começar de novo",
    },
    en: {
      step1: "1. Transcript",
      step2: "2. Chapters",
      step3: "3. YouTube description",
      step1Title: "Paste your transcript",
      step1Hint: "Works best with [MM:SS] timestamps — like Ouviescrevi formatted output.",
      placeholder: "Paste timestamped transcript here…",
      titleHint: "Episode title (optional)",
      titlePlaceholder: "E.g. Ep. 12 — Interview with…",
      langLabel: "Language",
      maxChapters: "Max chapters",
      btnChapters: "⏱️ Generate chapters",
      loadingChapters: "Generating chapters…",
      needText: "Paste some text first (min. ~120 characters).",
      error: "Processing error.",
      truncated: "Text was truncated.",
      noTimestamps: "No timestamps detected — times may be estimated.",
      step2Title: "Chapters ready",
      chaptersPreview: "Preview",
      btnYoutube: "▶️ Generate YouTube description",
      btnBack: "← Back",
      loadingYoutube: "Generating description…",
      step3Title: "YouTube description",
      titles: "Suggested titles",
      description: "Description",
      tags: "Tags",
      copyDesc: "Copy description",
      copyTags: "Copy tags",
      copyTitle: "Copy title",
      copied: "Copied!",
      copyFail: "Could not copy.",
      restart: "Start over",
    },
    es: {
      step1: "1. Transcripción",
      step2: "2. Capítulos",
      step3: "3. Descripción YouTube",
      step1Title: "Pega la transcripción",
      step1Hint: "Funciona mejor con marcas [MM:SS] — como la transcripción de Ouviescrevi.",
      placeholder: "Pega aquí la transcripción con marcas de tiempo…",
      titleHint: "Título del episodio (opcional)",
      titlePlaceholder: "Ej.: Ep. 12 — Entrevista con…",
      langLabel: "Idioma",
      maxChapters: "Máx. capítulos",
      btnChapters: "⏱️ Generar capítulos",
      loadingChapters: "Generando capítulos…",
      needText: "Introduce texto (mín. ~120 caracteres).",
      error: "Error al procesar.",
      truncated: "El texto fue truncado.",
      noTimestamps: "Sin marcas de tiempo — los horarios pueden ser estimados.",
      step2Title: "Capítulos listos",
      chaptersPreview: "Vista previa",
      btnYoutube: "▶️ Generar descripción YouTube",
      btnBack: "← Volver",
      loadingYoutube: "Generando descripción…",
      step3Title: "Descripción para YouTube",
      titles: "Títulos sugeridos",
      description: "Descripción",
      tags: "Etiquetas",
      copyDesc: "Copiar descripción",
      copyTags: "Copiar etiquetas",
      copyTitle: "Copiar título",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
      restart: "Empezar de nuevo",
    },
    fr: {
      step1: "1. Transcription",
      step2: "2. Chapitres",
      step3: "3. Description YouTube",
      step1Title: "Collez la transcription",
      step1Hint: "Idéal avec des horodatages [MM:SS] — comme la sortie Ouviescrevi.",
      placeholder: "Collez ici la transcription horodatée…",
      titleHint: "Titre de l'épisode (optionnel)",
      titlePlaceholder: "Ex. : Ép. 12 — Interview avec…",
      langLabel: "Langue",
      maxChapters: "Chapitres max.",
      btnChapters: "⏱️ Générer les chapitres",
      loadingChapters: "Génération des chapitres…",
      needText: "Saisissez du texte (min. ~120 caractères).",
      error: "Erreur de traitement.",
      truncated: "Le texte a été tronqué.",
      noTimestamps: "Pas d'horodatages — les temps peuvent être estimés.",
      step2Title: "Chapitres prêts",
      chaptersPreview: "Aperçu",
      btnYoutube: "▶️ Générer la description YouTube",
      btnBack: "← Retour",
      loadingYoutube: "Génération de la description…",
      step3Title: "Description YouTube",
      titles: "Titres suggérés",
      description: "Description",
      tags: "Tags",
      copyDesc: "Copier la description",
      copyTags: "Copier les tags",
      copyTitle: "Copier le titre",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
      restart: "Recommencer",
    },
    de: {
      step1: "1. Transkript",
      step2: "2. Kapitel",
      step3: "3. YouTube-Beschreibung",
      step1Title: "Transkript einfügen",
      step1Hint: "Am besten mit [MM:SS]-Zeitstempeln — wie bei Ouviescrevi.",
      placeholder: "Transkript mit Zeitstempeln hier einfügen…",
      titleHint: "Episodentitel (optional)",
      titlePlaceholder: "Z. B. Folge 12 — Interview mit…",
      langLabel: "Sprache",
      maxChapters: "Max. Kapitel",
      btnChapters: "⏱️ Kapitel generieren",
      loadingChapters: "Kapitel werden erstellt…",
      needText: "Text eingeben (min. ~120 Zeichen).",
      error: "Verarbeitungsfehler.",
      truncated: "Text wurde gekürzt.",
      noTimestamps: "Keine Zeitstempel — Zeiten können geschätzt sein.",
      step2Title: "Kapitel fertig",
      chaptersPreview: "Vorschau",
      btnYoutube: "▶️ YouTube-Beschreibung generieren",
      btnBack: "← Zurück",
      loadingYoutube: "Beschreibung wird erstellt…",
      step3Title: "YouTube-Beschreibung",
      titles: "Vorgeschlagene Titel",
      description: "Beschreibung",
      tags: "Tags",
      copyDesc: "Beschreibung kopieren",
      copyTags: "Tags kopieren",
      copyTitle: "Titel kopieren",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
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

  function youtubeLines(chapters) {
    return (chapters || [])
      .map(function (ch) {
        return (ch.start || "0:00") + " " + (ch.title || "");
      })
      .join("\n");
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

  function updateSteps() {
    var items = document.querySelectorAll(".oe-pw-steps li");
    items.forEach(function (li, i) {
      var n = i + 1;
      li.classList.remove("is-active", "is-done");
      if (n < state.step) li.classList.add("is-done");
      if (n === state.step) li.classList.add("is-active");
    });
    ["pwStep1", "pwStep2", "pwStep3"].forEach(function (id, i) {
      var el = document.getElementById(id);
      if (el) el.hidden = state.step !== i + 1;
    });
  }

  function renderStep3(container, data, truncated) {
    var titles = (data.titles || [])
      .map(function (title, i) {
        return (
          '<li class="oe-yt-title">' +
          '<span class="oe-yt-title__text">' +
          escapeHtml(title) +
          "</span>" +
          '<button type="button" class="oe-yt-copy-sm" data-copy-title="' +
          i +
          '">' +
          escapeHtml(t("copyTitle")) +
          "</button></li>"
        );
      })
      .join("");

    container.innerHTML =
      '<div class="oe-yt-result">' +
      (truncated ? '<p class="oe-yt-warn">' + escapeHtml(t("truncated")) + "</p>" : "") +
      '<section class="oe-yt-block"><h2>' +
      escapeHtml(t("titles")) +
      '</h2><ol class="oe-yt-titles">' +
      titles +
      "</ol></section>" +
      '<section class="oe-yt-block"><div class="oe-yt-block__head"><h2>' +
      escapeHtml(t("description")) +
      '</h2><button type="button" class="oe-yt-result__btn oe-yt-result__btn--primary" data-copy="desc">' +
      escapeHtml(t("copyDesc")) +
      "</button></div><pre class="oe-yt-pre">" +
      escapeHtml(data.description || "") +
      "</pre></section>" +
      '<section class="oe-yt-block"><div class="oe-yt-block__head"><h2>' +
      escapeHtml(t("tags")) +
      '</h2><button type="button" class="oe-yt-result__btn" data-copy="tags">' +
      escapeHtml(t("copyTags")) +
      "</button></div><p class="oe-yt-tags">" +
      escapeHtml(data.tags_csv || (data.tags || []).join(", ")) +
      "</p></section>" +
      '<button type="button" class="oe-pw-btn oe-pw-btn--ghost" id="btnPwRestart">' +
      escapeHtml(t("restart")) +
      "</button></div>";

    container.querySelector('[data-copy="desc"]').addEventListener("click", function () {
      copyText(data.description || "");
    });
    container.querySelector('[data-copy="tags"]').addEventListener("click", function () {
      copyText(data.tags_csv || (data.tags || []).join(", "));
    });
    container.querySelectorAll("[data-copy-title]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var i = parseInt(btn.getAttribute("data-copy-title"), 10);
        if (data.titles && data.titles[i]) copyText(data.titles[i]);
      });
    });
    var restart = document.getElementById("btnPwRestart");
    if (restart) {
      restart.addEventListener("click", function () {
        state = { step: 1, text: "", chaptersData: null, youtubeData: null };
        var ta = document.getElementById("texto");
        if (ta) ta.value = "";
        updateSteps();
      });
    }
  }

  async function generateChapters() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnPwChapters");
    if (!textoEl || !btn) return;

    var texto = textoEl.value.trim();
    if (texto.length < 120) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    state.text = texto;
    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loadingChapters"));

    try {
      await global.OuviescreviAPI.init();
      var langEl = document.getElementById("pwLang");
      var maxEl = document.getElementById("pwMax");
      var payload = {
        text: texto,
        lang: langEl ? langEl.value : config.lang,
        max_chapters: maxEl ? parseInt(maxEl.value, 10) || 12 : 12,
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-chapters", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(data.detail || t("error"), "error");
        return;
      }
      state.chaptersData = data;
      state.step = 2;
      updateSteps();

      var preview = document.getElementById("pwChaptersPreview");
      var warns = document.getElementById("pwChaptersWarn");
      if (preview) preview.textContent = youtubeLines(data.chapters);
      if (warns) {
        var parts = [];
        if (data.truncated) parts.push(t("truncated"));
        if (!data.has_timestamps) parts.push(t("noTimestamps"));
        warns.textContent = parts.join(" ");
        warns.hidden = !parts.length;
      }
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  async function generateYoutube() {
    var btn = document.getElementById("btnPwYoutube");
    var out = document.getElementById("pwYoutubeResult");
    if (!btn || !out || !state.chaptersData) return;

    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loadingYoutube"));

    try {
      await global.OuviescreviAPI.init();
      var langEl = document.getElementById("pwLang");
      var hintEl = document.getElementById("pwTitleHint");
      var payload = {
        text: state.text,
        lang: langEl ? langEl.value : config.lang,
        title_hint: hintEl ? hintEl.value.trim() : "",
        chapters_text: youtubeLines(state.chaptersData.chapters),
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-youtube-description", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(data.detail || t("error"), "error");
        return;
      }
      state.youtubeData = data;
      state.step = 3;
      updateSteps();
      renderStep3(out, data, !!data.truncated);
      out.hidden = false;
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
      pwStep1Label: "step1",
      pwStep2Label: "step2",
      pwStep3Label: "step3",
      pwStep1Title: "step1Title",
      pwStep1Hint: "step1Hint",
      pwTitleHintLabel: "titleHint",
      pwLangLabel: "langLabel",
      pwMaxLabel: "maxChapters",
      pwStep2Title: "step2Title",
      pwChaptersPreviewLabel: "chaptersPreview",
      pwStep3Title: "step3Title",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });
    var ta = document.getElementById("texto");
    if (ta) ta.placeholder = t("placeholder");
    var hint = document.getElementById("pwTitleHint");
    if (hint) hint.placeholder = t("titlePlaceholder");
    var btn1 = document.getElementById("btnPwChapters");
    if (btn1) btn1.textContent = t("btnChapters");
    var btn2 = document.getElementById("btnPwYoutube");
    if (btn2) btn2.textContent = t("btnYoutube");
    var back = document.getElementById("btnPwBack");
    if (back) back.textContent = t("btnBack");
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyLabels();
    updateSteps();

    var btnCh = document.getElementById("btnPwChapters");
    if (btnCh) btnCh.addEventListener("click", generateChapters);
    var btnYt = document.getElementById("btnPwYoutube");
    if (btnYt) btnYt.addEventListener("click", generateYoutube);
    var back = document.getElementById("btnPwBack");
    if (back) {
      back.addEventListener("click", function () {
        state.step = 1;
        updateSteps();
      });
    }

    try {
      var saved = sessionStorage.getItem(STORAGE_KEY);
      var ta = document.getElementById("texto");
      if (saved && ta && !ta.value.trim()) {
        ta.value = saved;
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {}
  }

  global.PodcastYoutubeUI = { init: init };
})(typeof window !== "undefined" ? window : this);
