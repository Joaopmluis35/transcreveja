/**
 * Resumo por URL — formulário, progresso e painel de resultado.
 */
(function (global) {
  var config = { lang: "pt", apiLang: "pt" };
  var lastSummary = "";

  var STRINGS = {
    pt: {
      eyebrow: "Artigos · Notícias · Blogues",
      formTitle: "Link do artigo",
      formHint: "Cola o URL público de uma página com texto — ideal para notícias, blogues e artigos online.",
      placeholder: "https://exemplo.com/artigo",
      modeLabel: "Estilo de resumo",
      modeNormal: "Clássico e direto",
      modeMinuta: "Minuta em tópicos",
      formNote: "Sites com paywall, login ou vídeo podem não funcionar. Nesse caso, cola o texto em Resumo Inteligente.",
      btnGenerate: "Gerar Resumo",
      loading: "A resumir...",
      needUrl: "Introduz um link válido.",
      resultTitle: "Resumo gerado",
      errorTitle: "Não foi possível resumir",
      copy: "Copiar",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      serverError: "Erro ao contactar o servidor.",
      unexpected: "Ocorreu um erro inesperado.",
      extractFail: "Não foi possível ler o texto desta página.",
      extractTips: [
        "Confirma que o link abre um artigo público (não uma homepage ou rede social).",
        "Alguns jornais bloqueiam leitura automática — tenta outro site ou cola o texto em Resumo Inteligente.",
        "Evita PDFs, páginas de login e conteúdo só em vídeo.",
      ],
      phrases: [
        "A ler o artigo com atenção...",
        "A organizar as ideias principais...",
        "A pensar como um humano...",
        "A resumir com clareza...",
        "A preparar a resposta final...",
      ],
    },
    en: {
      eyebrow: "Articles · News · Blogs",
      formTitle: "Article link",
      formHint: "Paste a public URL with readable text — news, blogs and online articles work best.",
      placeholder: "https://example.com/article",
      modeLabel: "Summary style",
      modeNormal: "Classic and direct",
      modeMinuta: "Bullet-point minutes",
      formNote: "Paywalled, login-only or video pages may not work. Paste the text on Smart Summary instead.",
      btnGenerate: "Generate Summary",
      loading: "Summarizing...",
      needUrl: "Enter a valid URL.",
      resultTitle: "Summary",
      errorTitle: "Could not summarize",
      copy: "Copy",
      copied: "Copied!",
      copyFail: "Could not copy.",
      serverError: "Failed to contact the server.",
      unexpected: "An unexpected error occurred.",
      extractFail: "Could not read text from this page.",
      extractTips: [
        "Make sure the link opens a public article (not a homepage or social feed).",
        "Some news sites block automatic reading — try another site or paste the text on Smart Summary.",
        "Avoid PDFs, login pages and video-only content.",
      ],
      phrases: [
        "Reading the article carefully...",
        "Organizing key ideas...",
        "Thinking like a human...",
        "Summarizing clearly...",
        "Preparing the final answer...",
      ],
    },
    es: {
      eyebrow: "Artículos · Noticias · Blogs",
      formTitle: "Enlace del artículo",
      formHint: "Pega una URL pública con texto legible — ideal para noticias, blogs y artículos online.",
      placeholder: "https://ejemplo.com/articulo",
      modeLabel: "Estilo de resumen",
      modeNormal: "Clásico y directo",
      modeMinuta: "Minuta en viñetas",
      formNote: "Sitios con paywall, login o solo vídeo pueden fallar. Pega el texto en Resumen Inteligente.",
      btnGenerate: "Generar resumen",
      loading: "Resumiendo...",
      needUrl: "Introduce un enlace válido.",
      resultTitle: "Resumen generado",
      errorTitle: "No se pudo resumir",
      copy: "Copiar",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
      serverError: "Error al contactar el servidor.",
      unexpected: "Ocurrió un error inesperado.",
      extractFail: "No se pudo leer el texto de esta página.",
      extractTips: [
        "Confirma que el enlace abre un artículo público (no una portada o red social).",
        "Algunos periódicos bloquean la lectura automática — prueba otro sitio o pega el texto en Resumen Inteligente.",
        "Evita PDFs, páginas de login y contenido solo en vídeo.",
      ],
      phrases: [
        "Leyendo el artículo con atención...",
        "Organizando las ideas principales...",
        "Pensando como un humano...",
        "Resumiendo con claridad...",
        "Preparando la respuesta final...",
      ],
    },
    fr: {
      eyebrow: "Articles · Actualités · Blogs",
      formTitle: "Lien de l'article",
      formHint: "Collez une URL publique avec du texte lisible — idéal pour articles, blogs et actualités.",
      placeholder: "https://exemple.com/article",
      modeLabel: "Style de résumé",
      modeNormal: "Classique et direct",
      modeMinuta: "Compte-rendu en puces",
      formNote: "Les sites payants, avec login ou vidéo seule peuvent échouer. Collez le texte dans Résumé Intelligent.",
      btnGenerate: "Générer le résumé",
      loading: "Résumé en cours...",
      needUrl: "Entrez une URL valide.",
      resultTitle: "Résumé généré",
      errorTitle: "Impossible de résumer",
      copy: "Copier",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
      serverError: "Erreur de connexion au serveur.",
      unexpected: "Une erreur inattendue s'est produite.",
      extractFail: "Impossible de lire le texte de cette page.",
      extractTips: [
        "Vérifiez que le lien ouvre un article public (pas une page d'accueil ou un réseau social).",
        "Certains journaux bloquent la lecture automatique — essayez un autre site ou collez le texte dans Résumé Intelligent.",
        "Évitez les PDF, pages de connexion et contenu uniquement vidéo.",
      ],
      phrases: [
        "Lecture attentive de l'article...",
        "Organisation des idées principales...",
        "Réflexion en cours...",
        "Résumé en clarté...",
        "Préparation de la réponse finale...",
      ],
    },
    de: {
      eyebrow: "Artikel · Nachrichten · Blogs",
      formTitle: "Artikellink",
      formHint: "Füge eine öffentliche URL mit lesbarem Text ein — ideal für Nachrichten, Blogs und Online-Artikel.",
      placeholder: "https://beispiel.de/artikel",
      modeLabel: "Zusammenfassungsstil",
      modeNormal: "Klassisch und direkt",
      modeMinuta: "Stichpunkt-Protokoll",
      formNote: "Paywall-, Login- oder reine Videoseiten funktionieren oft nicht. Texte stattdessen unter Intelligente Zusammenfassung einfügen.",
      btnGenerate: "Zusammenfassung erstellen",
      loading: "Wird zusammengefasst...",
      needUrl: "Bitte eine gültige URL eingeben.",
      resultTitle: "Zusammenfassung",
      errorTitle: "Zusammenfassung fehlgeschlagen",
      copy: "Kopieren",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
      serverError: "Server nicht erreichbar.",
      unexpected: "Ein unerwarteter Fehler ist aufgetreten.",
      extractFail: "Text von dieser Seite konnte nicht gelesen werden.",
      extractTips: [
        "Stelle sicher, dass der Link einen öffentlichen Artikel öffnet (keine Startseite oder Social Feed).",
        "Manche Nachrichtenseiten blockieren automatisches Lesen — andere Seite versuchen oder Text unter Intelligente Zusammenfassung einfügen.",
        "PDFs, Login-Seiten und reine Video-Inhalte vermeiden.",
      ],
      phrases: [
        "Artikel wird aufmerksam gelesen...",
        "Hauptideen werden sortiert...",
        "Wird durchdacht...",
        "Klare Zusammenfassung...",
        "Antwort wird vorbereitet...",
      ],
    },
  };

  function t(key) {
    var pack = STRINGS[config.lang] || STRINGS.pt;
    return pack[key] != null ? pack[key] : STRINGS.pt[key];
  }

  function applyFormLabels() {
    var map = {
      urlFormEyebrow: "eyebrow",
      urlFormTitle: "formTitle",
      urlFormHint: "formHint",
      urlModoLabel: "modeLabel",
      urlFormNote: "formNote",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });

    var input = document.getElementById("urlInput");
    if (input) input.placeholder = t("placeholder");

    var btn = document.getElementById("btnUrlResumo");
    if (btn) btn.textContent = t("btnGenerate");

    var modo = document.getElementById("urlModo");
    if (modo && modo.options.length >= 2) {
      modo.options[0].textContent = t("modeNormal");
      modo.options[1].textContent = t("modeMinuta");
    }
  }

  function hideOutput(out) {
    if (!out) return;
    out.hidden = true;
    out.innerHTML = "";
    out.classList.remove("oe-url-output--error");
  }

  function showSuccess(out, text) {
    lastSummary = text;
    out.hidden = false;
    out.classList.remove("oe-url-output--error");
    out.innerHTML =
      '<div class="oe-url-output__head">' +
      '<h2 class="oe-url-output__title">' + escapeHtml(t("resultTitle")) + "</h2>" +
      '<div class="oe-url-output__actions">' +
      '<button type="button" class="oe-url-output__btn" data-url-copy>' +
      escapeHtml(t("copy")) +
      "</button></div></div>" +
      '<pre class="oe-url-output__body">' +
      escapeHtml(text) +
      "</pre>";

    var copyBtn = out.querySelector("[data-url-copy]");
    if (copyBtn) {
      copyBtn.addEventListener("click", function () {
        copySummary(copyBtn);
      });
    }
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(out, message, isExtract) {
    lastSummary = "";
    out.hidden = false;
    out.classList.add("oe-url-output--error");
    var tips = isExtract ? t("extractTips") : [];
    var tipsHtml = "";
    if (tips && tips.length) {
      tipsHtml = "<ul class=\"oe-url-output__tips\">";
      tips.forEach(function (tip) {
        tipsHtml += "<li>" + escapeHtml(tip) + "</li>";
      });
      tipsHtml += "</ul>";
    }
    out.innerHTML =
      '<div class="oe-url-output__head">' +
      '<h2 class="oe-url-output__title">' + escapeHtml(t("errorTitle")) + "</h2>" +
      "</div>" +
      '<pre class="oe-url-output__body">' +
      escapeHtml(message) +
      "</pre>" +
      tipsHtml;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function isExtractError(detail) {
    if (!detail) return false;
    var s = String(detail).toLowerCase();
    return (
      s.indexOf("extrair conteúdo") !== -1 ||
      s.indexOf("extrair conteudo") !== -1 ||
      s.indexOf("extract") !== -1 ||
      s.indexOf("ler o texto") !== -1 ||
      s.indexOf("read text") !== -1 ||
      s.indexOf("bloquear") !== -1 ||
      s.indexOf("block") !== -1 ||
      s.indexOf("não está acessível") !== -1 ||
      s.indexOf("not accessible") !== -1
    );
  }

  function copySummary(btn) {
    if (!lastSummary) return;
    var done = function () {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copied"), "success");
      if (btn) {
        var prev = btn.textContent;
        btn.textContent = t("copied");
        setTimeout(function () {
          btn.textContent = prev;
        }, 1800);
      }
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(lastSummary).then(done).catch(function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      });
    } else if (global.OuviescreviUI) {
      global.OuviescreviUI.toast(t("copyFail"), "error");
    }
  }

  async function summarizeFromPage() {
    var urlInput = document.getElementById("urlInput");
    var btn = document.getElementById("btnUrlResumo");
    var out = document.getElementById("resultado");
    var progress = document.getElementById("urlProgress");
    var progressBar = document.getElementById("progressBar");
    var loadingPhrase = document.getElementById("loadingPhrase");
    var modo = document.getElementById("urlModo");

    if (!urlInput || !btn || !out) return;

    var url = urlInput.value.trim();
    if (!url) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needUrl"), "error");
      return;
    }

    hideOutput(out);
    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));
    }
    if (progress) progress.hidden = false;
    if (progressBar) progressBar.style.width = "0%";
    if (loadingPhrase) loadingPhrase.textContent = "";

    var phrases = t("phrases");
    var percent = 0;
    var fraseIndex = 0;
    var interval = setInterval(function () {
      if (percent >= 92) return;
      percent += Math.floor(Math.random() * 5) + 1;
      if (progressBar) progressBar.style.width = percent + "%";
      if (loadingPhrase && phrases && phrases.length) {
        loadingPhrase.textContent = phrases[fraseIndex];
        fraseIndex = (fraseIndex + 1) % phrases.length;
      }
    }, 700);

    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/summarize-url", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI.authJson({
            url: url,
            lang: config.apiLang,
            mode: modo ? modo.value : "normal",
          })
        ),
      });
      var data = await res.json().catch(function () {
        return {};
      });
      clearInterval(interval);
      if (progressBar) progressBar.style.width = "100%";
      if (loadingPhrase) loadingPhrase.textContent = "";

      if (res.ok && data.summary) {
        showSuccess(out, data.summary);
      } else {
        var detail = data.detail || data.error || t("unexpected");
        if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg || d; }).join(" ");
        var extract = isExtractError(detail);
        showError(out, extract ? t("extractFail") : String(detail), extract);
      }
    } catch (err) {
      clearInterval(interval);
      if (progressBar) progressBar.style.width = "100%";
      if (loadingPhrase) loadingPhrase.textContent = "";
      console.error(err);
      showError(out, t("serverError"), false);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
      setTimeout(function () {
        if (progress) progress.hidden = true;
      }, 800);
    }
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyFormLabels();
    var btn = document.getElementById("btnUrlResumo");
    if (btn) btn.addEventListener("click", summarizeFromPage);
    var input = document.getElementById("urlInput");
    if (input) {
      input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          summarizeFromPage();
        }
      });
    }
    global.resumirArtigoPorURL = summarizeFromPage;
    global.summarizeArticleFromURL = summarizeFromPage;
  }

  global.UrlResumoUI = { init: init, summarize: summarizeFromPage };
})(window);
