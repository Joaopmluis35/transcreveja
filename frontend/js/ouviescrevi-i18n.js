/**
 * Locales do site — PT (raiz), EN/ES/FR/DE em subpastas.
 */
(function (global) {
  "use strict";

  var LOCALES = ["pt", "en", "es", "fr", "de"];

  var LOCALE_META = {
    pt: { label: "Português", flag: "/icons/pt.png?v=2", og: "pt_PT", html: "pt" },
    en: { label: "English", flag: "/icons/en.png?v=2", og: "en_GB", html: "en" },
    es: { label: "Español", flag: "/icons/es.png?v=2", og: "es_ES", html: "es" },
    fr: { label: "Français", flag: "/icons/fr.png?v=2", og: "fr_FR", html: "fr" },
    de: { label: "Deutsch", flag: "/icons/de.png?v=2", og: "de_DE", html: "de" },
  };

  /** Páginas equivalentes entre locales (slug lógico → ficheiro por locale). */
  var PAGES = {
    index: { pt: "index.html", en: "index.html", es: "index.html", fr: "index.html", de: "index.html" },
    ajuda: { pt: "ajuda.html", en: "ajuda.html", es: "ajuda.html", fr: "ajuda.html", de: "ajuda.html" },
    conversor: { pt: "conversor.html", en: "conversor.html", es: "conversor.html", fr: "conversor.html", de: "conversor.html" },
    "conversor-imagens": { pt: "conversor-imagens.html", en: "conversor-imagens.html", es: "conversor-imagens.html", fr: "conversor-imagens.html", de: "conversor-imagens.html" },
    sugestoes: { pt: "sugestoes.html", en: "sugestoes.html", es: "sugestoes.html", fr: "sugestoes.html", de: "sugestoes.html" },
    resumo: { pt: "resumo.html", en: "resumo.html", es: "resumo.html", fr: "resumo.html", de: "resumo.html" },
    "url-resumo": { pt: "url-resumo.html", en: "url-resumo.html", es: "url-resumo.html", fr: "url-resumo.html", de: "url-resumo.html" },
    perguntas: { pt: "perguntas.html", en: "perguntas.html", es: "perguntas.html", fr: "perguntas.html", de: "perguntas.html" },
    "aula-pronta": { pt: "aula-pronta.html", en: "aula-pronta.html", es: "aula-pronta.html", fr: "aula-pronta.html", de: "aula-pronta.html" },
    capitulos: { pt: "capitulos.html", en: "capitulos.html", es: "capitulos.html", fr: "capitulos.html", de: "capitulos.html" },
    flashcards: { pt: "flashcards.html", en: "flashcards.html", es: "flashcards.html", fr: "flashcards.html", de: "flashcards.html" },
    "aula-completa": { pt: "aula-completa.html", en: "aula-completa.html", es: "aula-completa.html", fr: "aula-completa.html", de: "aula-completa.html" },
    "podcast-youtube": { pt: "podcast-youtube.html", en: "podcast-youtube.html", es: "podcast-youtube.html", fr: "podcast-youtube.html", de: "podcast-youtube.html" },
    "descricao-youtube": { pt: "descricao-youtube.html", en: "descricao-youtube.html", es: "descricao-youtube.html", fr: "descricao-youtube.html", de: "descricao-youtube.html" },
    corretor: { pt: "corretor.html", en: "corretor.html", es: "corretor.html", fr: "corretor.html", de: "corretor.html" },
    cookies: { pt: "cookies.html", en: "cookies.html", es: "cookies.html", fr: "cookies.html", de: "cookies.html" },
    privacy: { pt: "privacidade.html", en: "privacy.html", es: "privacy.html", fr: "privacy.html", de: "privacy.html" },
    terms: { pt: "termos.html", en: "terms.html", es: "terms.html", fr: "terms.html", de: "terms.html" },
  };

  function localeFromPath(path) {
    path = path || (global.location && global.location.pathname) || "/";
    var m = path.match(/^\/(en|es|fr|de)(\/|$)/);
    return m ? m[1] : "pt";
  }

  function localePrefix(locale) {
    if (!locale || locale === "pt") return "";
    return "/" + locale;
  }

  function pageSlugFromPath(path) {
    path = (path || "/").replace(/\/$/, "");
    var file = path.split("/").pop() || "index.html";
    if (file === "" || file === "index.html") return "index";
    var name = file.replace(".html", "");
    if (PAGES[name]) return name;
    if (name === "privacidade" || name === "privacy") return "privacy";
    if (name === "termos" || name === "terms") return "terms";
    return null;
  }

  function pathFor(locale, slug) {
    var map = PAGES[slug];
    if (!map) return localePrefix(locale) + "/index.html";
    var file = map[locale] || map.en || map.pt;
    var prefix = localePrefix(locale);
    return (prefix || "") + "/" + file;
  }

  function currentPagePath() {
    var path = global.location.pathname.replace(/\/$/, "") || "/index.html";
    if (path === "" || path === "/") return "/index.html";
    return path;
  }

  function hreflangMapForPath(path) {
    var slug = pageSlugFromPath(path);
    if (!slug) return null;
    var out = {};
    LOCALES.forEach(function (loc) {
      out[loc] = pathFor(loc, slug);
    });
    return out;
  }

  function switchLanguage(target) {
    if (LOCALES.indexOf(target) === -1) return;
    try {
      localStorage.setItem("lang", target);
    } catch (e) {}
    var slug = pageSlugFromPath(global.location.pathname) || "index";
    global.location.href = pathFor(target, slug);
  }

  function uiStrings(locale) {
    var L = {
      pt: {
        openMenu: "Abrir menu",
        closeMenu: "Fechar menu",
        cookieAria: "Aviso de cookies",
        cookieText: "Utilizamos armazenamento essencial no browser para o serviço funcionar. ",
        cookieLink: "Política de Cookies",
        cookieBtn: "Compreendi",
        cookiesPath: "/cookies.html",
      },
      en: {
        openMenu: "Open menu",
        closeMenu: "Close menu",
        cookieAria: "Cookie notice",
        cookieText: "We use essential browser storage for the service to work. ",
        cookieLink: "Cookie Policy",
        cookieBtn: "OK",
        cookiesPath: "/en/cookies.html",
      },
      es: {
        openMenu: "Abrir menú",
        closeMenu: "Cerrar menú",
        cookieAria: "Aviso de cookies",
        cookieText: "Usamos almacenamiento esencial en el navegador para que el servicio funcione. ",
        cookieLink: "Política de cookies",
        cookieBtn: "Entendido",
        cookiesPath: "/es/cookies.html",
      },
      fr: {
        openMenu: "Ouvrir le menu",
        closeMenu: "Fermer le menu",
        cookieAria: "Avis sur les cookies",
        cookieText: "Nous utilisons un stockage essentiel dans le navigateur pour faire fonctionner le service. ",
        cookieLink: "Politique de cookies",
        cookieBtn: "Compris",
        cookiesPath: "/fr/cookies.html",
      },
      de: {
        openMenu: "Menü öffnen",
        closeMenu: "Menü schließen",
        cookieAria: "Cookie-Hinweis",
        cookieText: "Wir verwenden wesentlichen Browserspeicher, damit der Dienst funktioniert. ",
        cookieLink: "Cookie-Richtlinie",
        cookieBtn: "Verstanden",
        cookiesPath: "/de/cookies.html",
      },
    };
    return L[locale] || L.en;
  }

  function langMenuHtml(current) {
    return LOCALES.map(function (loc) {
      var meta = LOCALE_META[loc];
      return (
        '<button type="button" data-lang="' + loc + '" role="menuitem">' +
        '<img src="' + meta.flag + '" alt="" width="22" height="16"> ' +
        meta.label +
        "</button>"
      );
    }).join("\n");
  }

  global.OuviescreviI18n = {
    LOCALES: LOCALES,
    LOCALE_META: LOCALE_META,
    PAGES: PAGES,
    localeFromPath: localeFromPath,
    localePrefix: localePrefix,
    pageSlugFromPath: pageSlugFromPath,
    pathFor: pathFor,
    currentPagePath: currentPagePath,
    hreflangMapForPath: hreflangMapForPath,
    switchLanguage: switchLanguage,
    uiStrings: uiStrings,
    langMenuHtml: langMenuHtml,
  };
})(window);
