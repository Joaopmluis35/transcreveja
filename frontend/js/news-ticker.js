(function (global) {
  "use strict";

  var RELEASE = "11/07/2026";

  var PACKS = {
    pt: {
      badge: "NOVO",
      items: [
        "Flashcards com IA — PDF e impressão personalizada",
        "Assistente Aula completa: transcrição → flashcards",
        "Descrição YouTube com IA",
        "Conversor de imagens: comprimir e unir em PDF",
      ],
    },
    en: {
      badge: "NEW",
      items: [
        "AI flashcards — custom PDF export and print",
        "Full Lesson Assistant: transcript to flashcards",
        "AI YouTube descriptions",
        "Image converter: compress and merge to PDF",
      ],
    },
    es: {
      badge: "NUEVO",
      items: [
        "Flashcards con IA — PDF e impresión personalizada",
        "Asistente Clase completa: transcripción → flashcards",
        "Descripción YouTube con IA",
        "Conversor de imágenes: comprimir y unir en PDF",
      ],
    },
    fr: {
      badge: "NOUVEAU",
      items: [
        "Flashcards IA — PDF et impression personnalisée",
        "Assistant Cours complet : transcription → flashcards",
        "Descriptions YouTube par IA",
        "Convertisseur d'images : compresser et fusionner en PDF",
      ],
    },
    de: {
      badge: "NEU",
      items: [
        "KI-Karteikarten — PDF exportieren und drucken",
        "Assistent Vollständige Lektion: Transkript → Karteikarten",
        "YouTube-Beschreibungen mit KI",
        "Bildkonverter: komprimieren und zu PDF zusammenführen",
      ],
    },
  };

  function lang() {
    var code = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    return PACKS[code] ? code : "pt";
  }

  function displayDate(code) {
    if (code === "de") return "11.07.2026";
    return RELEASE;
  }

  function shouldShow() {
    if (document.body && document.body.dataset.oeNoNewsTicker === "true") return false;
    var path = (global.location && global.location.pathname) || "";
    if (/backoffice/i.test(path)) return false;
    if (/admin\.html$/i.test(path)) return false;
    return true;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function itemHtml(badge, text, date) {
    return (
      '<span class="oe-news-ticker__item">' +
      '<span class="oe-news-ticker__badge">' +
      escapeHtml(badge) +
      "</span>" +
      '<span class="oe-news-ticker__text">' +
      escapeHtml(date + " – " + text) +
      "</span>" +
      "</span>"
    );
  }

  function fillTrack(track, pack, date) {
    var chunks = [];
    pack.items.forEach(function (text, index) {
      if (index > 0) {
        chunks.push('<span class="oe-news-ticker__dot" aria-hidden="true">•</span>');
      }
      chunks.push(itemHtml(pack.badge, text, date));
    });
    var once = chunks.join("");
    track.innerHTML =
      once +
      '<span class="oe-news-ticker__dot" aria-hidden="true">•</span>' +
      once;
  }

  function upgradeLegacyTicker(pack, date) {
    var legacy = document.querySelector(".ticker:not(.oe-news-ticker)");
    if (!legacy) return false;
    legacy.id = "oeNewsTicker";
    legacy.className = "oe-news-ticker";
    var track =
      legacy.querySelector("#updatesTrack") ||
      legacy.querySelector(".ticker__inner") ||
      legacy;
    track.className = "oe-news-ticker__track";
    fillTrack(track, pack, date);
    return true;
  }

  function ariaLabel(code) {
    var labels = {
      pt: "Novidades",
      en: "What's new",
      es: "Novedades",
      fr: "Nouveautés",
      de: "Neuigkeiten",
    };
    return labels[code] || labels.pt;
  }

  function mount() {
    if (!shouldShow()) return;
    if (document.getElementById("oeNewsTicker")) return;

    var code = lang();
    var pack = PACKS[code];
    var date = displayDate(code);

    if (upgradeLegacyTicker(pack, date)) return;

    var header = document.getElementById("header");
    if (!header || !header.querySelector("#oeProHeader")) return;

    var wrap = document.createElement("div");
    wrap.id = "oeNewsTicker";
    wrap.className = "oe-news-ticker";
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", ariaLabel(code));
    wrap.innerHTML = '<div class="oe-news-ticker__track" aria-hidden="true"></div>';
    header.insertAdjacentElement("afterend", wrap);
    fillTrack(wrap.querySelector(".oe-news-ticker__track"), pack, date);
  }

  global.OuviescreviNewsTicker = { mount: mount };

  function watchHeaderForTicker() {
    if (!shouldShow()) return;
    if (document.getElementById("oeNewsTicker")) return;
    var header = document.getElementById("header");
    if (!header) return;
    if (header.querySelector("#oeProHeader")) {
      mount();
      return;
    }
    var obs = new MutationObserver(function () {
      if (document.getElementById("oeNewsTicker")) {
        obs.disconnect();
        return;
      }
      if (header.querySelector("#oeProHeader")) {
        mount();
        obs.disconnect();
      }
    });
    obs.observe(header, { childList: true, subtree: true });
  }

  function boot() {
    mount();
    watchHeaderForTicker();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})(window);
