/**
 * Capítulos & timestamps — divisão de transcrições longas.
 */
(function (global) {
  var STORAGE_KEY = "oe_capitulos_text";
  var config = { lang: "pt" };
  var lastResult = null;

  var STRINGS = {
    pt: {
      hint: "Funciona melhor com texto no formato [MM:SS] por bloco — como a transcrição formatada do Ouviescrevi.",
      placeholder: "Cola aqui a transcrição com timestamps…",
      btn: "⏱️ Gerar capítulos",
      loading: "A gerar capítulos…",
      needText: "Introduz texto (mín. ~120 caracteres).",
      error: "Erro ao gerar capítulos.",
      truncated: "O texto foi truncado — os capítulos baseiam-se no início do conteúdo.",
      noTimestamps: "Sem timestamps detetados — os capítulos são lógicos (sem horários).",
      copyYoutube: "Copiar YouTube",
      openYoutubeDesc: "Descrição YouTube",
      copyAll: "Copiar lista",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      youtubeHint: "Cola na descrição do YouTube (um capítulo por linha):",
      chapter: "Capítulo",
    },
    en: {
      hint: "Works best with [MM:SS] blocks — like Ouviescrevi formatted transcription.",
      placeholder: "Paste timestamped transcript here…",
      btn: "⏱️ Generate chapters",
      loading: "Generating chapters…",
      needText: "Paste some text first (min. ~120 characters).",
      error: "Error generating chapters.",
      truncated: "Text was truncated — chapters are based on the beginning.",
      noTimestamps: "No timestamps detected — chapters are logical (no times).",
      copyYoutube: "Copy YouTube",
      openYoutubeDesc: "YouTube description",
      copyAll: "Copy list",
      copied: "Copied!",
      copyFail: "Could not copy.",
      youtubeHint: "Paste in YouTube description (one chapter per line):",
      chapter: "Chapter",
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
        if (ch.youtube_start) return ch.youtube_start + " " + ch.title;
        if (ch.start) return ch.start + " " + ch.title;
        return ch.title;
      })
      .join("\n");
  }

  function allLines(data) {
    var lines = [];
    if (data.title) lines.push(data.title, "");
    (data.chapters || []).forEach(function (ch, i) {
      var time = ch.start ? "[" + ch.start + "] " : "";
      lines.push((i + 1) + ". " + time + ch.title);
      if (ch.summary) lines.push("   " + ch.summary);
    });
    return lines.join("\n");
  }

  function copyText(text, okMsg) {
    if (!navigator.clipboard || !navigator.clipboard.writeText) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      return;
    }
    navigator.clipboard.writeText(text).then(
      function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(okMsg || t("copied"), "success");
      },
      function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      }
    );
  }

  function renderResult(container, data, truncated, hasTimestamps) {
    lastResult = data;
    var warns = [];
    if (truncated) warns.push(t("truncated"));
    if (!hasTimestamps) warns.push(t("noTimestamps"));

    var list = (data.chapters || [])
      .map(function (ch) {
        var time = ch.start
          ? '<span class="oe-cap-chapter__time">' + escapeHtml(ch.start) + "</span>"
          : '<span class="oe-cap-chapter__time oe-cap-chapter__time--empty">—</span>';
        return (
          '<li class="oe-cap-chapter">' +
          time +
          '<h3 class="oe-cap-chapter__title">' +
          escapeHtml(ch.title) +
          "</h3>" +
          (ch.summary
            ? '<p class="oe-cap-chapter__summary">' + escapeHtml(ch.summary) + "</p>"
            : "") +
          "</li>"
        );
      })
      .join("");

    container.innerHTML =
      '<div class="oe-cap-result">' +
      (warns.length
        ? '<p class="oe-cap-warn">' + escapeHtml(warns.join(" ")) + "</p>"
        : "") +
      '<header class="oe-cap-result__head">' +
      '<h2 class="oe-cap-result__title">' +
      escapeHtml(data.title || t("chapter")) +
      "</h2>" +
      '<div class="oe-cap-result__actions">' +
      '<button type="button" class="oe-cap-result__btn" data-cap-copy="all">' +
      escapeHtml(t("copyAll")) +
      "</button>" +
      '<button type="button" class="oe-cap-result__btn oe-cap-result__btn--primary" data-cap-copy="youtube">' +
      escapeHtml(t("copyYoutube")) +
      "</button>" +
      '<button type="button" class="oe-cap-result__btn" data-cap-youtube-desc>' +
      escapeHtml(t("openYoutubeDesc")) +
      "</button>" +
      "</div></header>" +
      '<ol class="oe-cap-chapters">' +
      list +
      "</ol>" +
      '<p class="oe-cap-youtube">' +
      escapeHtml(t("youtubeHint")) +
      "</p></div>";

    container.hidden = false;
    container.querySelectorAll("[data-cap-copy]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (!lastResult) return;
        if (btn.getAttribute("data-cap-copy") === "youtube") {
          copyText(youtubeLines(lastResult.chapters), t("copied"));
        } else {
          copyText(allLines(lastResult), t("copied"));
        }
      });
    });
    var ytBtn = container.querySelector("[data-cap-youtube-desc]");
    if (ytBtn) {
      ytBtn.addEventListener("click", function () {
        var ta = document.getElementById("texto");
        var text = ta ? ta.value.trim() : "";
        try {
          if (text) sessionStorage.setItem("oe_youtube_desc_text", text);
          if (lastResult) {
            sessionStorage.setItem("oe_youtube_desc_chapters", youtubeLines(lastResult.chapters));
          }
        } catch (e) {}
        var dest =
          config.lang === "en"
            ? "en/descricao-youtube.html"
            : config.lang === "es"
              ? "es/descricao-youtube.html"
              : config.lang === "fr"
                ? "fr/descricao-youtube.html"
                : config.lang === "de"
                  ? "de/descricao-youtube.html"
                  : "descricao-youtube.html";
        window.location.href = dest;
      });
    }
    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function generate() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnCapitulos");
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
      var langEl = document.getElementById("capLang");
      var maxEl = document.getElementById("capMax");
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
        out.innerHTML = "<pre>" + escapeHtml(data.detail || t("error")) + "</pre>";
        out.hidden = false;
        return;
      }
      renderResult(out, data, !!data.truncated, !!data.has_timestamps);
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("error"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    var hint = document.getElementById("capFormHint");
    if (hint) hint.textContent = t("hint");
    var ta = document.getElementById("texto");
    if (ta) ta.placeholder = t("placeholder");
    var btn = document.getElementById("btnCapitulos");
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

  global.CapitulosUI = { init: init };
})(typeof window !== "undefined" ? window : this);
