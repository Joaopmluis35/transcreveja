/**
 * Descrição para YouTube — título, descrição e tags com IA.
 */
(function (global) {
  var STORAGE_TEXT = "oe_youtube_desc_text";
  var STORAGE_CHAPTERS = "oe_youtube_desc_chapters";
  var config = { lang: "pt" };
  var lastData = null;

  var STRINGS = {
    pt: {
      btn: "▶️ Gerar descrição",
      loading: "A gerar descrição…",
      needText: "Introduz texto (mín. ~120 caracteres).",
      error: "Erro ao gerar descrição.",
      truncated: "O texto foi truncado.",
      titles: "Títulos sugeridos",
      description: "Descrição",
      tags: "Tags",
      copyDesc: "Copiar descrição",
      copyTags: "Copiar tags",
      copyTitle: "Copiar título",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
    },
    en: {
      btn: "▶️ Generate description",
      loading: "Generating description…",
      needText: "Paste some text first (min. ~120 characters).",
      error: "Error generating description.",
      truncated: "Text was truncated.",
      titles: "Suggested titles",
      description: "Description",
      tags: "Tags",
      copyDesc: "Copy description",
      copyTags: "Copy tags",
      copyTitle: "Copy title",
      copied: "Copied!",
      copyFail: "Could not copy.",
    },
    es: {
      btn: "▶️ Generar descripción",
      loading: "Generando descripción…",
      needText: "Introduce texto (mín. ~120 caracteres).",
      error: "Error al generar descripción.",
      truncated: "El texto fue truncado.",
      titles: "Títulos sugeridos",
      description: "Descripción",
      tags: "Etiquetas",
      copyDesc: "Copiar descripción",
      copyTags: "Copiar etiquetas",
      copyTitle: "Copiar título",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
    },
    fr: {
      btn: "▶️ Générer la description",
      loading: "Génération de la description…",
      needText: "Saisissez du texte (min. ~120 caractères).",
      error: "Erreur lors de la génération.",
      truncated: "Le texte a été tronqué.",
      titles: "Titres suggérés",
      description: "Description",
      tags: "Tags",
      copyDesc: "Copier la description",
      copyTags: "Copier les tags",
      copyTitle: "Copier le titre",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
    },
    de: {
      btn: "▶️ Beschreibung generieren",
      loading: "Beschreibung wird erstellt…",
      needText: "Text eingeben (min. ~120 Zeichen).",
      error: "Fehler beim Generieren.",
      truncated: "Text wurde gekürzt.",
      titles: "Vorgeschlagene Titel",
      description: "Beschreibung",
      tags: "Tags",
      copyDesc: "Beschreibung kopieren",
      copyTags: "Tags kopieren",
      copyTitle: "Titel kopieren",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
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

  function renderResult(container, data, truncated) {
    lastData = data;
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
      '<section class="oe-yt-block">' +
      "<h2>" +
      escapeHtml(t("titles")) +
      "</h2><ol class=\"oe-yt-titles\">" +
      titles +
      "</ol></section>" +
      '<section class="oe-yt-block">' +
      '<div class="oe-yt-block__head"><h2>' +
      escapeHtml(t("description")) +
      '</h2><button type="button" class="oe-yt-result__btn oe-yt-result__btn--primary" data-copy="desc">' +
      escapeHtml(t("copyDesc")) +
      "</button></div>" +
      '<pre class="oe-yt-pre" id="ytDescOut">' +
      escapeHtml(data.description || "") +
      "</pre></section>" +
      '<section class="oe-yt-block">' +
      '<div class="oe-yt-block__head"><h2>' +
      escapeHtml(t("tags")) +
      '</h2><button type="button" class="oe-yt-result__btn" data-copy="tags">' +
      escapeHtml(t("copyTags")) +
      "</button></div>" +
      '<p class="oe-yt-tags">' +
      escapeHtml(data.tags_csv || (data.tags || []).join(", ")) +
      "</p></section></div>";

    container.hidden = false;
    container.querySelector('[data-copy="desc"]').addEventListener("click", function () {
      if (lastData) copyText(lastData.description || "");
    });
    container.querySelector('[data-copy="tags"]').addEventListener("click", function () {
      if (lastData) copyText(lastData.tags_csv || (lastData.tags || []).join(", "));
    });
    container.querySelectorAll("[data-copy-title]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var i = parseInt(btn.getAttribute("data-copy-title"), 10);
        if (lastData && lastData.titles && lastData.titles[i]) {
          copyText(lastData.titles[i]);
        }
      });
    });
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function generate() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnYoutube");
    var out = document.getElementById("resultado");
    if (!textoEl || !btn || !out) return;

    var texto = textoEl.value.trim();
    if (texto.length < 120) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));

    try {
      await global.OuviescreviAPI.init();
      var langEl = document.getElementById("ytLang");
      var hintEl = document.getElementById("ytTitleHint");
      var chEl = document.getElementById("ytChapters");
      var payload = {
        text: texto,
        lang: langEl ? langEl.value : config.lang,
        title_hint: hintEl ? hintEl.value.trim() : "",
        chapters_text: chEl ? chEl.value.trim() : "",
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-youtube-description", {
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
    var btn = document.getElementById("btnYoutube");
    if (btn) {
      btn.textContent = t("btn");
      btn.addEventListener("click", generate);
    }
    try {
      var ta = document.getElementById("texto");
      var ch = document.getElementById("ytChapters");
      var saved = sessionStorage.getItem(STORAGE_TEXT);
      var savedCh = sessionStorage.getItem(STORAGE_CHAPTERS);
      if (saved && ta && !ta.value.trim()) {
        ta.value = saved;
        sessionStorage.removeItem(STORAGE_TEXT);
      }
      if (savedCh && ch && !ch.value.trim()) {
        ch.value = savedCh;
        sessionStorage.removeItem(STORAGE_CHAPTERS);
      }
    } catch (e) {}
  }

  global.DescricaoYoutubeUI = { init: init };
})(typeof window !== "undefined" ? window : this);
