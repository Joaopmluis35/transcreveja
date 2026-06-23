/**
 * Páginas CMS ES/FR/DE — fallback se a API ainda não tiver o schema atualizado.
 */
(function (global) {
  function localeCmsPages(lang, langLabel) {
    var p = lang;
    var base = "/" + lang;
    return [
      {
        id: "home_" + lang,
        label: "Homepage (" + langLabel + ")",
        lang: lang,
        path: base + "/index.html",
        fields: [
          { key: p + "_home_intro_html", label: "Texto de boas-vindas (topo)", type: "rich" },
        ],
      },
      {
        id: "ajuda_" + lang,
        label: "Ajuda (" + langLabel + ")",
        lang: lang,
        path: base + "/ajuda.html",
        fields: [
          { key: p + "_ajuda_title", label: "Título da página", type: "text" },
          { key: p + "_ajuda_intro", label: "Introdução", type: "rich" },
          { key: p + "_ajuda_faq", label: "Perguntas frequentes", type: "rich" },
          { key: p + "_ajuda_contact", label: "Secção de contacto", type: "rich" },
        ],
      },
      {
        id: "conversor_" + lang,
        label: "Conversor (" + langLabel + ")",
        lang: lang,
        path: base + "/conversor.html",
        fields: [
          { key: p + "_conversor_title", label: "Título", type: "text" },
          { key: p + "_conversor_lead", label: "Subtítulo", type: "text" },
          { key: p + "_conversor_notice", label: "Aviso", type: "rich" },
          { key: p + "_conversor_seo", label: "Texto SEO (rodapé)", type: "rich" },
        ],
      },
      {
        id: "sugestoes_" + lang,
        label: "Sugestões (" + langLabel + ")",
        lang: lang,
        path: base + "/sugestoes.html",
        fields: [
          { key: p + "_sugestoes_title", label: "Título", type: "text" },
          { key: p + "_sugestoes_lead", label: "Subtítulo", type: "text" },
        ],
      },
      {
        id: "privacidade_" + lang,
        label: "Legal — Privacidade (" + langLabel + ")",
        lang: lang,
        path: base + "/privacy.html",
        fields: [
          { key: p + "_privacidade_meta", label: "Linha «última atualização»", type: "rich" },
          { key: p + "_privacidade_disclaimer", label: "Aviso introdutório", type: "rich" },
        ],
      },
      {
        id: "termos_" + lang,
        label: "Legal — Termos (" + langLabel + ")",
        lang: lang,
        path: base + "/terms.html",
        fields: [
          { key: p + "_termos_meta", label: "Linha «última atualização»", type: "rich" },
          { key: p + "_termos_intro", label: "Aviso introdutório", type: "rich" },
        ],
      },
      {
        id: "cookies_" + lang,
        label: "Legal — Cookies (" + langLabel + ")",
        lang: lang,
        path: base + "/cookies.html",
        fields: [
          { key: p + "_cookies_meta", label: "Linha «última atualização»", type: "rich" },
          { key: p + "_cookies_intro", label: "Introdução (secção 1)", type: "rich" },
        ],
      },
    ];
  }

  var FALLBACK_LOCALE_PAGES = []
    .concat(localeCmsPages("es", "ES"))
    .concat(localeCmsPages("fr", "FR"))
    .concat(localeCmsPages("de", "DE"));

  function mergeLocaleCmsPages(apiPages) {
    var byId = {};
    (apiPages || []).forEach(function (p) {
      byId[p.id] = p;
    });
    FALLBACK_LOCALE_PAGES.forEach(function (p) {
      if (!byId[p.id]) byId[p.id] = p;
    });
    return Object.keys(byId).map(function (id) {
      return byId[id];
    });
  }

  var LOCALE_HOME_INTRO = {
    es: (
      "<p><strong>🧠 Ouviescrevi</strong> es tu asistente con IA para<br>" +
      "<strong>transcribir</strong> 🎙️, <strong>traducir</strong> 🌍, <strong>resumir</strong> 📌 " +
      "y <strong>convertir archivos</strong> 📄<br>— simple, rápido y gratuito.</p>"
    ),
    fr: (
      "<p><strong>🧠 Ouviescrevi</strong> est ton assistant IA pour<br>" +
      "<strong>transcrire</strong> 🎙️, <strong>traduire</strong> 🌍, <strong>résumer</strong> 📌 " +
      "et <strong>convertir des fichiers</strong> 📄<br>— simple, rapide et gratuit.</p>"
    ),
    de: (
      "<p><strong>🧠 Ouviescrevi</strong> ist dein KI-Assistent zum<br>" +
      "<strong>Transkribieren</strong> 🎙️, <strong>Übersetzen</strong> 🌍, <strong>Zusammenfassen</strong> 📌 " +
      "und <strong>Dateien konvertieren</strong> 📄<br>— einfach, schnell und kostenlos.</p>"
    ),
  };

  /** Preenche chaves es_/fr_/de_ quando a API ainda não as tem na BD. */
  function mergeLocaleCmsContent(content) {
    var out = Object.assign({}, content || {});
    ["es", "fr", "de"].forEach(function (lang) {
      Object.keys(out).forEach(function (key) {
        if (!key.startsWith("en_")) return;
        var localeKey = lang + key.slice(2);
        if (!out[localeKey] && out[key]) out[localeKey] = out[key];
      });
      var homeKey = lang + "_home_intro_html";
      if (!out[homeKey]) {
        if (LOCALE_HOME_INTRO[lang]) out[homeKey] = LOCALE_HOME_INTRO[lang];
        else if (out.home_intro_html) out[homeKey] = out.home_intro_html;
      }
    });
    return out;
  }

  global.OuviescreviCmsLocales = {
    mergeLocaleCmsPages: mergeLocaleCmsPages,
    mergeLocaleCmsContent: mergeLocaleCmsContent,
    FALLBACK_LOCALE_PAGES: FALLBACK_LOCALE_PAGES,
  };
})(window);
