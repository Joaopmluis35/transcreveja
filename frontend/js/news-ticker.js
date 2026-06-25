(function (global) {
  "use strict";

  var RELEASE = "25/06/2026";

  var PACKS = {
    pt: {
      badge: "NOVO",
      items: [
        "Partilha no WhatsApp disponível",
        "Legendar vídeo (SRT + MP4)",
      ],
    },
    en: {
      badge: "NEW",
      items: [
        "Share on WhatsApp available",
        "Video subtitling (SRT + MP4)",
      ],
    },
    es: {
      badge: "NUEVO",
      items: [
        "Compartir en WhatsApp disponible",
        "Subtitular vídeo (SRT + MP4)",
      ],
    },
    fr: {
      badge: "NOUVEAU",
      items: [
        "Partage sur WhatsApp disponible",
        "Sous-titrage vidéo (SRT + MP4)",
      ],
    },
    de: {
      badge: "NEU",
      items: [
        "WhatsApp-Teilen verfügbar",
        "Video-Untertitel (SRT + MP4)",
      ],
    },
  };

  function lang() {
    var code = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    return PACKS[code] ? code : "pt";
  }

  function displayDate(code) {
    if (code === "de") return "25.06.2026";
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

  function boot() {
    mount();
    var tries = 0;
    var timer = global.setInterval(function () {
      mount();
      tries += 1;
      if (document.getElementById("oeNewsTicker") || tries > 48) {
        global.clearInterval(timer);
      }
    }, 250);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})(window);
