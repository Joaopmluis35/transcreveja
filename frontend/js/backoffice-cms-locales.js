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

  global.OuviescreviCmsLocales = {
    mergeLocaleCmsPages: mergeLocaleCmsPages,
    FALLBACK_LOCALE_PAGES: FALLBACK_LOCALE_PAGES,
  };
})(window);
