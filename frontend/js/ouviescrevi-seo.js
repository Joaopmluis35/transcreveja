/**
 * SEO — canonical, Open Graph, Twitter, hreflang e JSON-LD.
 */
(function (global) {
  var SITE = "https://www.ouviescrevi.pt";
  var OG_IMAGE = SITE + "/logos/ouviescrevimainlogo.png";
  var ORG_ID = SITE + "/#organization";

  var HREFLANG = {
    "/index.html": { pt: "/index.html", en: "/en/index.html" },
    "/en/index.html": { pt: "/index.html", en: "/en/index.html" },
    "/ajuda.html": { pt: "/ajuda.html", en: "/en/ajuda.html" },
    "/en/ajuda.html": { pt: "/ajuda.html", en: "/en/ajuda.html" },
    "/conversor.html": { pt: "/conversor.html", en: "/en/conversor.html" },
    "/en/conversor.html": { pt: "/conversor.html", en: "/en/conversor.html" },
    "/sugestoes.html": { pt: "/sugestoes.html", en: "/en/sugestoes.html" },
    "/en/sugestoes.html": { pt: "/sugestoes.html", en: "/en/sugestoes.html" },
    "/resumo.html": { pt: "/resumo.html", en: "/en/resumo.html" },
    "/en/resumo.html": { pt: "/resumo.html", en: "/en/resumo.html" },
    "/url-resumo.html": { pt: "/url-resumo.html", en: "/en/url-resumo.html" },
    "/en/url-resumo.html": { pt: "/url-resumo.html", en: "/en/url-resumo.html" },
    "/perguntas.html": { pt: "/perguntas.html", en: "/en/perguntas.html" },
    "/en/perguntas.html": { pt: "/perguntas.html", en: "/en/perguntas.html" },
    "/privacidade.html": { pt: "/privacidade.html", en: "/en/privacy.html" },
    "/en/privacy.html": { pt: "/privacidade.html", en: "/en/privacy.html" },
    "/termos.html": { pt: "/termos.html", en: "/en/terms.html" },
    "/en/terms.html": { pt: "/termos.html", en: "/en/terms.html" },
    "/cookies.html": { pt: "/cookies.html", en: "/en/cookies.html" },
    "/en/cookies.html": { pt: "/cookies.html", en: "/en/cookies.html" },
  };

  var PAGES = {
    "/index.html": {
      title: "Ouviescrevi — Transcrição de Áudio e Vídeo com IA Grátis",
      description:
        "Transcreve áudio e vídeo online com inteligência artificial, grátis e sem registo. Resumos, tradução, legendas SRT e conversão de ficheiros. Feito em Portugal.",
      type: "home",
    },
    "/en/index.html": {
      title: "Ouviescrevi — Free AI Audio & Video Transcription",
      description:
        "Transcribe audio and video online with AI for free. Summaries, translation, SRT subtitles and file conversion. No sign-up required.",
      type: "home",
      lang: "en",
    },
    "/conversor.html": {
      title: "Conversor de Ficheiros Online Grátis — Word, PDF, Imagem | Ouviescrevi",
      description:
        "Converte Word para PDF, PDF para texto e imagens para PDF no browser. Conversor gratuito, rápido e sem instalação.",
    },
    "/en/conversor.html": {
      title: "Free Online File Converter — Word, PDF, Image | Ouviescrevi",
      description:
        "Convert Word to PDF, PDF to text and images to PDF in your browser. Free, fast and no installation.",
      lang: "en",
    },
    "/resumo.html": {
      title: "Resumo Automático com IA — PDF, Word e Texto | Ouviescrevi",
      description:
        "Gera resumos inteligentes com IA a partir de PDF, Word ou texto. Estilos formal, simples, tópicos ou minuta de reunião.",
    },
    "/en/resumo.html": {
      title: "AI Summary Generator — PDF, Word & Text | Ouviescrevi",
      description:
        "Generate smart AI summaries from PDF, Word or plain text. Formal, simple, bullet points or meeting minutes.",
      lang: "en",
    },
    "/ajuda.html": {
      title: "Ajuda e FAQ — Como Usar o Ouviescrevi",
      description:
        "Respostas às perguntas frequentes sobre transcrição com IA, formatos suportados, privacidade e funcionalidades do Ouviescrevi.",
      faq: true,
    },
    "/en/ajuda.html": {
      title: "Help & FAQ — How to Use Ouviescrevi",
      description:
        "Frequently asked questions about AI transcription, supported formats, privacy and Ouviescrevi features.",
      lang: "en",
      faq: true,
    },
    "/aulas.html": {
      title: "Transcrição de Aulas com IA — Estudantes e Professores | Ouviescrevi",
      description:
        "Transforma gravações de aulas em texto editável com IA. Ideal para estudantes, professores e ensino à distância.",
    },
    "/professores.html": {
      title: "IA para Professores — Transcrever e Resumir Aulas | Ouviescrevi",
      description:
        "Ferramentas de IA para educadores: transcrição de aulas, resumos automáticos e preparação de materiais didáticos.",
    },
    "/jornalistas.html": {
      title: "Transcrição de Entrevistas para Jornalistas | Ouviescrevi",
      description:
        "Transcreve entrevistas de áudio e vídeo com IA. Poupa horas de trabalho e exporta texto pronto a editar.",
    },
    "/podcasts.html": {
      title: "Transcrever Podcast com IA — Episódios em Texto | Ouviescrevi",
      description:
        "Converte episódios de podcast em transcrições e resumos com inteligência artificial. Grátis e online.",
    },
    "/reunioes.html": {
      title: "Transcrição de Reuniões e Minutas com IA | Ouviescrevi",
      description:
        "Grava e transcreve reuniões automaticamente. Gera minutas e resumos para equipas e empresas.",
    },
    "/testemunhos.html": {
      title: "Transcrição de Testemunhos e Declarações | Ouviescrevi",
      description:
        "Transcreve testemunhos, declarações e gravações com precisão. Útil em contextos jurídicos e administrativos.",
    },
    "/corretor.html": {
      title: "Corretor de Texto com IA — Ortografia e Gramática | Ouviescrevi",
      description:
        "Corrige ortografia e gramática automaticamente com inteligência artificial. Grátis e no browser.",
    },
    "/perguntas.html": {
      title: "Gerador de Perguntas de Escolha Múltipla com IA | Ouviescrevi",
      description:
        "Cria perguntas de estudo e testes a partir de qualquer texto com IA. Ideal para professores e alunos.",
    },
    "/en/perguntas.html": {
      title: "AI Multiple-Choice Question Generator | Ouviescrevi",
      description:
        "Generate study questions and quizzes from any text with AI. Perfect for teachers and students.",
      lang: "en",
    },
    "/url-resumo.html": {
      title: "Resumo de Artigo por URL com IA | Ouviescrevi",
      description:
        "Cola o link de um artigo online e obtém um resumo automático com inteligência artificial.",
    },
    "/en/url-resumo.html": {
      title: "AI Article Summary from URL | Ouviescrevi",
      description:
        "Paste an article link and get an automatic AI-generated summary in seconds.",
      lang: "en",
    },
    "/gerar-video.html": {
      title: "Gerar Vídeo com Voz e Legendas — Ouviescrevi",
      description:
        "Cria vídeos com narração automática e legendas a partir de texto com inteligência artificial.",
    },
    "/sugestoes.html": {
      title: "Enviar Sugestões — Ouviescrevi",
      description: "Partilha ideias para melhorar o Ouviescrevi. O teu feedback ajuda-nos a evoluir.",
    },
    "/en/sugestoes.html": {
      title: "Send Suggestions — Ouviescrevi",
      description: "Share ideas to improve Ouviescrevi. Your feedback helps us grow.",
      lang: "en",
    },
    "/privacidade.html": {
      title: "Política de Privacidade — Ouviescrevi (RGPD)",
      description:
        "Como o Ouviescrevi trata dados pessoais em conformidade com o RGPD. Ficheiros, cookies e os teus direitos.",
    },
    "/termos.html": {
      title: "Termos de Utilização — Ouviescrevi",
      description: "Condições de uso do serviço de transcrição e ferramentas de IA do Ouviescrevi.",
    },
    "/cookies.html": {
      title: "Política de Cookies — Ouviescrevi",
      description: "Informação sobre cookies e armazenamento local utilizados no website Ouviescrevi.",
    },
    "/en/privacy.html": {
      title: "Privacy Policy — Ouviescrevi (GDPR)",
      description: "How Ouviescrevi processes personal data under GDPR. Files, cookies and your rights.",
      lang: "en",
    },
    "/en/terms.html": {
      title: "Terms of Use — Ouviescrevi",
      description: "Terms and conditions for using Ouviescrevi AI transcription and tools.",
      lang: "en",
    },
    "/en/cookies.html": {
      title: "Cookie Policy — Ouviescrevi",
      description: "Information about cookies and local storage used on the Ouviescrevi website.",
      lang: "en",
    },
  };

  var FAQ_PT = [
    { q: "O que é o Ouviescrevi?", a: "É uma ferramenta automática que converte ficheiros de áudio, vídeo ou texto em transcrições, resumos, traduções e muito mais, usando inteligência artificial." },
    { q: "Que formatos de ficheiros são suportados?", a: "Suportamos .mp3, .mp4, .wav, .m4a, .mov e outros formatos comuns." },
    { q: "É necessário criar conta?", a: "Não. O Ouviescrevi está disponível gratuitamente e sem necessidade de registo." },
    { q: "A transcrição é 100% precisa?", a: "A precisão depende da qualidade do áudio. Utilizamos o modelo Whisper da OpenAI para alta qualidade." },
    { q: "Os meus ficheiros são guardados?", a: "Os ficheiros são eliminados após o processamento. Registamos apenas o nome e a data para estatísticas." },
  ];

  var FAQ_EN = [
    { q: "What is Ouviescrevi?", a: "A fully automated tool that converts audio, video, or text files into transcriptions, summaries, translations, and more using AI." },
    { q: "Which file formats are supported?", a: "We support .mp3, .mp4, .wav, .m4a, .mov and other common formats." },
    { q: "Do I need to create an account?", a: "No. Ouviescrevi is completely free and requires no registration." },
    { q: "Is the transcription 100% accurate?", a: "Accuracy depends on audio quality. We use OpenAI's Whisper model for high-quality results." },
    { q: "Are my files stored?", a: "Files are deleted after processing. Only file name and timestamp are stored for statistics." },
  ];

  function pagePath() {
    var p = (global.location.pathname || "/").replace(/\/$/, "");
    if (!p || p === "") return "/index.html";
    if (p.indexOf(".") === -1) return p + "/index.html";
    return p;
  }

  function upsertMeta(attr, key, value) {
    if (!value) return;
    var sel = "meta[" + attr + '="' + key + '"]';
    var el = document.querySelector(sel);
    if (!el) {
      el = document.createElement("meta");
      el.setAttribute(attr, key);
      document.head.appendChild(el);
    }
    el.setAttribute("content", value);
  }

  function upsertLink(rel, href, extra) {
    var sel = 'link[rel="' + rel + '"]';
    if (extra && extra.hreflang) sel += '[hreflang="' + extra.hreflang + '"]';
    var el = document.querySelector(sel);
    if (!el) {
      el = document.createElement("link");
      el.setAttribute("rel", rel);
      document.head.appendChild(el);
    }
    el.setAttribute("href", href);
    if (extra && extra.hreflang) el.setAttribute("hreflang", extra.hreflang);
  }

  function injectJsonLd(data) {
    var el = document.createElement("script");
    el.type = "application/ld+json";
    el.textContent = JSON.stringify(data);
    document.head.appendChild(el);
  }

  function breadcrumb(path, title) {
    var parts = path.split("/").filter(Boolean);
    var items = [{ "@type": "ListItem", position: 1, name: "Ouviescrevi", item: SITE + "/index.html" }];
    if (parts.length && parts[parts.length - 1] !== "index.html") {
      items.push({
        "@type": "ListItem",
        position: 2,
        name: title,
        item: SITE + path,
      });
    }
    return {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: items,
    };
  }

  function apply() {
    var path = pagePath();
    if (path.indexOf("backoffice") !== -1 || path.indexOf("/archive/") !== -1) return;

    var cfg = PAGES[path] || {};
    var title = cfg.title || document.title || "Ouviescrevi";
    var desc =
      cfg.description ||
      (document.querySelector('meta[name="description"]') || {}).content ||
      "Transcrição de áudio e vídeo com IA — Ouviescrevi";
    var lang = cfg.lang || (path.indexOf("/en/") === 0 ? "en" : "pt");
    var canonical = SITE + path;

    document.title = title;
    upsertMeta("name", "description", desc);
    upsertMeta("name", "robots", "index, follow");
    upsertMeta("name", "author", "Ouviescrevi");
    upsertLink("canonical", canonical);

    upsertMeta("property", "og:type", cfg.type === "home" ? "website" : "article");
    upsertMeta("property", "og:site_name", "Ouviescrevi");
    upsertMeta("property", "og:title", title);
    upsertMeta("property", "og:description", desc);
    upsertMeta("property", "og:url", canonical);
    upsertMeta("property", "og:image", OG_IMAGE);
    upsertMeta("property", "og:locale", lang === "en" ? "en_GB" : "pt_PT");

    upsertMeta("name", "twitter:card", "summary_large_image");
    upsertMeta("name", "twitter:title", title);
    upsertMeta("name", "twitter:description", desc);
    upsertMeta("name", "twitter:image", OG_IMAGE);

    var alternates = HREFLANG[path];
    if (alternates) {
      upsertLink("alternate", SITE + alternates.pt, { hreflang: "pt" });
      upsertLink("alternate", SITE + alternates.en, { hreflang: "en" });
      upsertLink("alternate", SITE + alternates.pt, { hreflang: "x-default" });
    }

    injectJsonLd(breadcrumb(path, title));
    injectJsonLd({
      "@context": "https://schema.org",
      "@type": "WebPage",
      name: title,
      description: desc,
      url: canonical,
      inLanguage: lang === "en" ? "en" : "pt-PT",
      isPartOf: { "@id": SITE + "/#website" },
    });

    if (cfg.type === "home") {
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "WebSite",
        "@id": SITE + "/#website",
        name: "Ouviescrevi",
        url: SITE + "/",
        description: desc,
        inLanguage: ["pt-PT", "en"],
        publisher: { "@id": ORG_ID },
      });
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "Organization",
        "@id": ORG_ID,
        name: "Ouviescrevi",
        url: SITE + "/",
        logo: OG_IMAGE,
        email: "ouviescrevi@gmail.com",
        areaServed: "PT",
      });
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        name: "Ouviescrevi",
        applicationCategory: "BusinessApplication",
        operatingSystem: "Web",
        offers: { "@type": "Offer", price: "0", priceCurrency: "EUR" },
        description: desc,
        url: canonical,
      });
    }

    if (cfg.faq) {
      var faqItems = lang === "en" ? FAQ_EN : FAQ_PT;
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "FAQPage",
        mainEntity: faqItems.map(function (item) {
          return {
            "@type": "Question",
            name: item.q,
            acceptedAnswer: { "@type": "Answer", text: item.a },
          };
        }),
      });
    }
  }

  function applyOverrides(seoMap) {
    if (!seoMap) return;
    var path = pagePath();
    var o = seoMap[path];
    if (!o) return;
    if (o.title) document.title = o.title;
    if (o.description) upsertMeta("name", "description", o.description);
    if (o.title) upsertMeta("property", "og:title", o.title);
    if (o.description) upsertMeta("property", "og:description", o.description);
  }

  global.OuviescreviSEO = { apply: apply, pagePath: pagePath, applyOverrides: applyOverrides };
})(window);
