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

  function enExtraCmsPages() {
    return [
      {
        id: "home_en",
        label: "Homepage (EN)",
        lang: "en",
        path: "/en/index.html",
        fields: [{ key: "en_home_intro_html", label: "Welcome text (top)", type: "rich" }],
      },
      {
        id: "resumo_en",
        label: "Summary tool (EN)",
        lang: "en",
        path: "/en/resumo.html",
        fields: [
          { key: "en_resumo_title", label: "Title", type: "text" },
          { key: "en_resumo_lead", label: "Subtitle", type: "text" },
        ],
      },
      {
        id: "url_resumo_en",
        label: "URL summary (EN)",
        lang: "en",
        path: "/en/url-resumo.html",
        fields: [
          { key: "en_url_resumo_title", label: "Title", type: "text" },
          { key: "en_url_resumo_lead", label: "Subtitle", type: "text" },
        ],
      },
      {
        id: "perguntas_en",
        label: "Quiz generator (EN)",
        lang: "en",
        path: "/en/perguntas.html",
        fields: [
          { key: "en_perguntas_title", label: "Title", type: "text" },
          { key: "en_perguntas_lead", label: "Subtitle", type: "text" },
        ],
      },
    ];
  }

  var FALLBACK_EN_PAGES = enExtraCmsPages();

  function localeSeoPages(lang, langLabel) {
    var base = "/" + lang;
    return [
      {
        id: "seo_home_" + lang,
        label: "SEO — Homepage (" + langLabel + ")",
        lang: lang,
        path: base + "/index.html",
        category: "seo",
        fields: [
          { key: "meta_home_title_" + lang, label: "Meta title", type: "text" },
          { key: "meta_home_description_" + lang, label: "Meta description", type: "text" },
        ],
      },
      {
        id: "seo_ajuda_" + lang,
        label: "SEO — Ajuda (" + langLabel + ")",
        lang: lang,
        path: base + "/ajuda.html",
        category: "seo",
        fields: [
          { key: "meta_ajuda_title_" + lang, label: "Meta title", type: "text" },
          { key: "meta_ajuda_description_" + lang, label: "Meta description", type: "text" },
        ],
      },
      {
        id: "seo_conversor_" + lang,
        label: "SEO — Conversor (" + langLabel + ")",
        lang: lang,
        path: base + "/conversor.html",
        category: "seo",
        fields: [
          { key: "meta_conversor_title_" + lang, label: "Meta title", type: "text" },
          { key: "meta_conversor_description_" + lang, label: "Meta description", type: "text" },
        ],
      },
      {
        id: "seo_resumo_" + lang,
        label: "SEO — Resumo (" + langLabel + ")",
        lang: lang,
        path: base + "/resumo.html",
        category: "seo",
        fields: [
          { key: "meta_resumo_title_" + lang, label: "Meta title", type: "text" },
          { key: "meta_resumo_description_" + lang, label: "Meta description", type: "text" },
        ],
      },
    ];
  }

  var FALLBACK_LOCALE_SEO_PAGES = []
    .concat(localeSeoPages("en", "EN"))
    .concat(localeSeoPages("es", "ES"))
    .concat(localeSeoPages("fr", "FR"))
    .concat(localeSeoPages("de", "DE"));

  var LOCALE_SEO_DEFAULTS = {
    meta_home_title_en: "Ouviescrevi — Free AI Audio & Video Transcription",
    meta_home_description_en:
      "Transcribe audio and video online with AI for free. Summaries, translation, SRT subtitles and file conversion. No sign-up required.",
    meta_ajuda_title_en: "Help & FAQ — How to Use Ouviescrevi",
    meta_ajuda_description_en:
      "Frequently asked questions about AI transcription, supported formats, privacy and Ouviescrevi features.",
    meta_conversor_title_en: "Free Online File Converter — Word, PDF, Image | Ouviescrevi",
    meta_conversor_description_en:
      "Convert Word to PDF, PDF to text and images to PDF in your browser. Free, fast and no installation.",
    meta_resumo_title_en: "AI Summary Generator — PDF, Word & Text | Ouviescrevi",
    meta_resumo_description_en:
      "Generate smart AI summaries from PDF, Word or plain text. Formal, simple, bullet points or meeting minutes.",
    meta_home_title_es: "Ouviescrevi — Transcripción de audio y vídeo con IA gratis",
    meta_home_description_es:
      "Transcribe audio y vídeo online con inteligencia artificial, gratis y sin registro. Resúmenes, traducción, subtítulos SRT y conversión de archivos.",
    meta_ajuda_title_es: "Ayuda y FAQ — Cómo usar Ouviescrevi",
    meta_ajuda_description_es: "Preguntas frecuentes sobre transcripción con IA, formatos compatibles y privacidad.",
    meta_conversor_title_es: "Conversor de archivos online gratis — Word, PDF, imagen | Ouviescrevi",
    meta_conversor_description_es: "Convierte Word a PDF, PDF a texto e imágenes a PDF en el navegador. Gratis y sin instalación.",
    meta_resumo_title_es: "Resumen automático con IA — PDF, Word y texto | Ouviescrevi",
    meta_resumo_description_es: "Genera resúmenes inteligentes con IA a partir de PDF, Word o texto.",
    meta_home_title_fr: "Ouviescrevi — Transcription audio et vidéo IA gratuite",
    meta_home_description_fr:
      "Transcrivez audio et vidéo en ligne avec l'IA, gratuitement et sans inscription. Résumés, traduction, sous-titres SRT et conversion de fichiers.",
    meta_ajuda_title_fr: "Aide et FAQ — Utiliser Ouviescrevi",
    meta_ajuda_description_fr: "Questions fréquentes sur la transcription IA, les formats pris en charge et la confidentialité.",
    meta_conversor_title_fr: "Convertisseur de fichiers en ligne gratuit — Word, PDF | Ouviescrevi",
    meta_conversor_description_fr: "Convertissez Word en PDF, PDF en texte et images en PDF dans le navigateur.",
    meta_resumo_title_fr: "Résumé automatique avec IA — PDF, Word et texte | Ouviescrevi",
    meta_resumo_description_fr: "Générez des résumés intelligents avec l'IA à partir de PDF, Word ou texte.",
    meta_home_title_de: "Ouviescrevi — Kostenlose KI-Audio- und Video-Transkription",
    meta_home_description_de:
      "Transkribiere Audio und Video online mit KI, kostenlos und ohne Anmeldung. Zusammenfassungen, Übersetzung, SRT-Untertitel und Dateikonvertierung.",
    meta_ajuda_title_de: "Hilfe & FAQ — Ouviescrevi nutzen",
    meta_ajuda_description_de: "Häufige Fragen zu KI-Transkription, unterstützten Formaten und Datenschutz.",
    meta_conversor_title_de: "Kostenloser Online-Dateikonverter — Word, PDF | Ouviescrevi",
    meta_conversor_description_de: "Word zu PDF, PDF zu Text und Bilder zu PDF im Browser konvertieren.",
    meta_resumo_title_de: "KI-Zusammenfassung — PDF, Word & Text | Ouviescrevi",
    meta_resumo_description_de: "Erstelle intelligente Zusammenfassungen mit KI aus PDF, Word oder Text.",
  };

  function mergeLocaleCmsPages(apiPages) {
    var byId = {};
    (apiPages || []).forEach(function (p) {
      byId[p.id] = p;
    });
    FALLBACK_LOCALE_PAGES.forEach(function (p) {
      if (!byId[p.id]) byId[p.id] = p;
    });
    FALLBACK_EN_PAGES.forEach(function (p) {
      if (!byId[p.id]) byId[p.id] = p;
    });
    return Object.keys(byId).map(function (id) {
      return byId[id];
    });
  }

  var LOCALE_HOME_INTRO = {
    en: (
      "<p><strong>🧠 Ouviescrevi</strong> is your AI assistant to<br>" +
      "<strong>transcribe</strong> 🎙️, <strong>translate</strong> 🌍, <strong>summarise</strong> 📌 " +
      "and <strong>convert files</strong> 📄<br>— simple, fast and free.</p>"
    ),
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
    var enDefaults = {
      en_home_intro_html: LOCALE_HOME_INTRO.en,
      en_resumo_title: "📌 Smart Summary",
      en_resumo_lead: "Paste your text or upload a PDF or Word file, then choose a summary style.",
      en_url_resumo_title: "🔗 Smart Summary from URL",
      en_url_resumo_lead: "Paste an article link to generate an automatic AI summary.",
      en_perguntas_title: "📘 AI Quiz Generator",
      en_perguntas_lead:
        "Paste your text here to generate multiple-choice questions with answers and explanations.",
    };
    Object.keys(enDefaults).forEach(function (key) {
      if (!out[key]) out[key] = enDefaults[key];
    });
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

  function mergeLocaleSeoPages(apiPages) {
    var seoFromApi = (apiPages || []).filter(function (p) {
      return p.category === "seo";
    });
    var byId = {};
    seoFromApi.forEach(function (p) {
      byId[p.id] = p;
    });
    FALLBACK_LOCALE_SEO_PAGES.forEach(function (p) {
      if (!byId[p.id]) byId[p.id] = p;
    });
    return Object.keys(byId).map(function (id) {
      return byId[id];
    });
  }

  function mergeLocaleSeoContent(content) {
    var out = Object.assign({}, content || {});
    Object.keys(LOCALE_SEO_DEFAULTS).forEach(function (key) {
      if (!out[key]) out[key] = LOCALE_SEO_DEFAULTS[key];
    });
    return out;
  }

  global.OuviescreviCmsLocales = {
    mergeLocaleCmsPages: mergeLocaleCmsPages,
    mergeLocaleCmsContent: mergeLocaleCmsContent,
    mergeLocaleSeoPages: mergeLocaleSeoPages,
    mergeLocaleSeoContent: mergeLocaleSeoContent,
    FALLBACK_LOCALE_PAGES: FALLBACK_LOCALE_PAGES,
  };
})(window);
