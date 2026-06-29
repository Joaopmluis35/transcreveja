/**
 * Corretor ortográfico — formulário, diff, exportação, histórico.
 */
(function (global) {
  var config = { lang: "pt", signatureUrl: "https://ouviescrevi.pt/corretor.html" };
  var lastOriginal = "";
  var lastCorrected = "";
  var compareVisible = false;

  var STRINGS = {
    pt: {
      eyebrow: "Ortografia · Gramática · Estilo",
      formTitle: "O teu texto",
      formHint: "Cola, escreve ou carrega um PDF/Word. A IA corrige erros e melhora a clareza.",
      dropLabel: "Arrasta um PDF ou Word (.docx) aqui, ou clica para escolher",
      placeholder: "Escreve ou cola aqui o teu texto...",
      modeLabel: "Tipo de correção",
      modeNormal: "Completa (ortografia + gramática)",
      modeSpelling: "Só ortografia e pontuação",
      modeFormal: "Tom mais formal",
      modeSimple: "Linguagem mais simples",
      shortcutHint: "Atalho: Ctrl+Enter para corrigir",
      btnCorrect: "Corrigir texto",
      loading: "A corrigir...",
      needText: "Introduz texto para corrigir.",
      chars: "%n caracteres",
      words: "%n palavras",
      placeholderTitle: "Resultado",
      placeholderHint: "O texto corrigido aparece aqui, lado a lado com o original.",
      loadingTitle: "A corrigir…",
      resultTitle: "Texto corrigido",
      resultSubtitle: "Palavras alteradas estão sublinhadas. Revê o resultado e escolhe uma ação.",
      actionsExport: "Exportar",
      errorTitle: "Não foi possível corrigir",
      copy: "Copiar",
      apply: "Aplicar ao texto",
      compare: "Comparar",
      hideCompare: "Só resultado",
      download: "TXT",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      recorrect: "Corrigir outra vez",
      copied: "Copiado!",
      applied: "Texto atualizado!",
      copyFail: "Não foi possível copiar.",
      pdfFail: "Não foi possível gerar PDF.",
      serverError: "Erro ao contactar o servidor.",
      unexpected: "Ocorreu um erro inesperado.",
      compareOriginal: "Original",
      compareFixed: "Corrigido",
      progressHint: "Textos longos podem demorar alguns segundos.",
      signature: "\n\n— Corrigido com Ouviescrevi: https://ouviescrevi.pt/corretor.html",
      filePdfFail: "Erro ao ler o PDF.",
      fileDocxFail: "Erro ao ler o ficheiro Word.",
      fileUnsupported: "Formato não suportado. Usa PDF ou DOCX.",
      historyTitle: "Correções recentes",
      historyRefresh: "Atualizar",
      historyLoading: "A carregar…",
      historyEmpty: "Ainda não tens correções guardadas.",
      historyLogin: "Inicia sessão para guardar e rever correções anteriores.",
      historyLoginBtn: "Entrar ou registar",
      historyLoaded: "Correção carregada do histórico.",
      historyDeleted: "Correção apagada.",
      historyFail: "Não foi possível carregar o histórico.",
      historyDeleteConfirm: "Apagar esta correção do histórico?",
      phrases: [
        "A ler o texto com atenção...",
        "A identificar erros ortográficos...",
        "A rever gramática e pontuação...",
        "A aplicar correções inteligentes...",
        "A preparar o texto final...",
      ],
    },
    en: {
      eyebrow: "Spelling · Grammar · Style",
      formTitle: "Your text",
      formHint: "Paste, type or upload a PDF/Word file. AI fixes errors and improves clarity.",
      dropLabel: "Drag a PDF or Word (.docx) here, or click to choose",
      placeholder: "Write or paste your text here...",
      modeLabel: "Correction type",
      modeNormal: "Full (spelling + grammar)",
      modeSpelling: "Spelling and punctuation only",
      modeFormal: "More formal tone",
      modeSimple: "Simpler language",
      shortcutHint: "Shortcut: Ctrl+Enter to correct",
      btnCorrect: "Correct text",
      loading: "Correcting...",
      needText: "Enter some text to correct.",
      chars: "%n characters",
      words: "%n words",
      placeholderTitle: "Result",
      placeholderHint: "The corrected text appears here, side by side with the original.",
      loadingTitle: "Correcting…",
      resultTitle: "Corrected text",
      resultSubtitle: "Changed words are highlighted. Review the result and choose an action.",
      actionsExport: "Export",
      errorTitle: "Could not correct",
      copy: "Copy",
      apply: "Apply to text",
      compare: "Compare",
      hideCompare: "Result only",
      download: "TXT",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      recorrect: "Correct again",
      copied: "Copied!",
      applied: "Text updated!",
      copyFail: "Could not copy.",
      pdfFail: "Could not generate PDF.",
      serverError: "Failed to contact the server.",
      unexpected: "An unexpected error occurred.",
      compareOriginal: "Original",
      compareFixed: "Corrected",
      progressHint: "Long texts may take a few seconds.",
      signature: "\n\n— Corrected with Ouviescrevi: https://ouviescrevi.pt/en/corretor.html",
      filePdfFail: "Failed to read PDF.",
      fileDocxFail: "Failed to read Word file.",
      fileUnsupported: "Unsupported format. Use PDF or DOCX.",
      historyTitle: "Recent corrections",
      historyRefresh: "Refresh",
      historyLoading: "Loading…",
      historyEmpty: "No saved corrections yet.",
      historyLogin: "Sign in to save and review past corrections.",
      historyLoginBtn: "Sign in or register",
      historyLoaded: "Correction loaded from history.",
      historyDeleted: "Correction deleted.",
      historyFail: "Could not load history.",
      historyDeleteConfirm: "Delete this correction from history?",
      phrases: [
        "Reading the text carefully...",
        "Finding spelling mistakes...",
        "Checking grammar and punctuation...",
        "Applying smart fixes...",
        "Preparing the final text...",
      ],
    },
    es: {
      eyebrow: "Ortografía · Gramática · Estilo",
      formTitle: "Tu texto",
      formHint: "Pega, escribe o sube un PDF/Word. La IA corrige errores y mejora la claridad.",
      dropLabel: "Arrastra un PDF o Word (.docx) aquí, o haz clic para elegir",
      placeholder: "Escribe o pega tu texto aquí...",
      modeLabel: "Tipo de corrección",
      modeNormal: "Completa (ortografía + gramática)",
      modeSpelling: "Solo ortografía y puntuación",
      modeFormal: "Tono más formal",
      modeSimple: "Lenguaje más simple",
      shortcutHint: "Atajo: Ctrl+Enter para corregir",
      btnCorrect: "Corregir texto",
      loading: "Corrigiendo...",
      needText: "Introduce texto para corregir.",
      chars: "%n caracteres",
      words: "%n palabras",
      placeholderTitle: "Resultado",
      placeholderHint: "El texto corregido aparece aquí, junto al original.",
      loadingTitle: "Corrigiendo…",
      resultTitle: "Texto corregido",
      resultSubtitle: "Las palabras cambiadas están resaltadas. Revisa el resultado y elige una acción.",
      actionsExport: "Exportar",
      errorTitle: "No se pudo corregir",
      copy: "Copiar",
      apply: "Aplicar al texto",
      compare: "Comparar",
      hideCompare: "Solo resultado",
      download: "TXT",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      recorrect: "Corregir otra vez",
      copied: "¡Copiado!",
      applied: "¡Texto actualizado!",
      copyFail: "No se pudo copiar.",
      pdfFail: "No se pudo generar PDF.",
      serverError: "Error al contactar el servidor.",
      unexpected: "Ocurrió un error inesperado.",
      compareOriginal: "Original",
      compareFixed: "Corregido",
      progressHint: "Los textos largos pueden tardar unos segundos.",
      signature: "\n\n— Corregido con Ouviescrevi: https://ouviescrevi.pt/es/corretor.html",
      filePdfFail: "Error al leer el PDF.",
      fileDocxFail: "Error al leer el Word.",
      fileUnsupported: "Formato no soportado. Usa PDF o DOCX.",
      historyTitle: "Correcciones recientes",
      historyRefresh: "Actualizar",
      historyLoading: "Cargando…",
      historyEmpty: "Aún no tienes correcciones guardadas.",
      historyLogin: "Inicia sesión para guardar y revisar correcciones anteriores.",
      historyLoginBtn: "Entrar o registrarse",
      historyLoaded: "Corrección cargada del historial.",
      historyDeleted: "Corrección eliminada.",
      historyFail: "No se pudo cargar el historial.",
      historyDeleteConfirm: "¿Eliminar esta corrección del historial?",
      phrases: [
        "Leyendo el texto con atención...",
        "Detectando errores ortográficos...",
        "Revisando gramática y puntuación...",
        "Aplicando correcciones...",
        "Preparando el texto final...",
      ],
    },
    fr: {
      eyebrow: "Orthographe · Grammaire · Style",
      formTitle: "Votre texte",
      formHint: "Collez, saisissez ou téléversez un PDF/Word. L'IA corrige les erreurs et améliore la clarté.",
      dropLabel: "Glissez un PDF ou Word (.docx) ici, ou cliquez pour choisir",
      placeholder: "Écrivez ou collez votre texte ici...",
      modeLabel: "Type de correction",
      modeNormal: "Complète (orthographe + grammaire)",
      modeSpelling: "Orthographe et ponctuation seulement",
      modeFormal: "Ton plus formel",
      modeSimple: "Langage plus simple",
      shortcutHint: "Raccourci : Ctrl+Entrée pour corriger",
      btnCorrect: "Corriger le texte",
      loading: "Correction en cours...",
      needText: "Entrez du texte à corriger.",
      chars: "%n caractères",
      words: "%n mots",
      placeholderTitle: "Résultat",
      placeholderHint: "Le texte corrigé s'affiche ici, côte à côte avec l'original.",
      loadingTitle: "Correction…",
      resultTitle: "Texte corrigé",
      resultSubtitle: "Les mots modifiés sont surlignés. Relisez le résultat et choisissez une action.",
      actionsExport: "Exporter",
      errorTitle: "Impossible de corriger",
      copy: "Copier",
      apply: "Appliquer au texte",
      compare: "Comparer",
      hideCompare: "Résultat seul",
      download: "TXT",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      recorrect: "Corriger à nouveau",
      copied: "Copié !",
      applied: "Texte mis à jour !",
      copyFail: "Impossible de copier.",
      pdfFail: "Impossible de générer le PDF.",
      serverError: "Erreur de connexion au serveur.",
      unexpected: "Une erreur inattendue s'est produite.",
      compareOriginal: "Original",
      compareFixed: "Corrigé",
      progressHint: "Les textes longs peuvent prendre quelques secondes.",
      signature: "\n\n— Corrigé avec Ouviescrevi: https://ouviescrevi.pt/fr/corretor.html",
      filePdfFail: "Erreur de lecture du PDF.",
      fileDocxFail: "Erreur de lecture du Word.",
      fileUnsupported: "Format non pris en charge. Utilisez PDF ou DOCX.",
      historyTitle: "Corrections récentes",
      historyRefresh: "Actualiser",
      historyLoading: "Chargement…",
      historyEmpty: "Aucune correction enregistrée.",
      historyLogin: "Connectez-vous pour enregistrer et revoir vos corrections.",
      historyLoginBtn: "Connexion ou inscription",
      historyLoaded: "Correction chargée depuis l'historique.",
      historyDeleted: "Correction supprimée.",
      historyFail: "Impossible de charger l'historique.",
      historyDeleteConfirm: "Supprimer cette correction de l'historique ?",
      phrases: [
        "Lecture attentive du texte...",
        "Détection des fautes d'orthographe...",
        "Vérification grammaire et ponctuation...",
        "Application des corrections...",
        "Préparation du texte final...",
      ],
    },
    de: {
      eyebrow: "Rechtschreibung · Grammatik · Stil",
      formTitle: "Dein Text",
      formHint: "Einfügen, tippen oder PDF/Word hochladen. Die KI korrigiert Fehler und verbessert die Klarheit.",
      dropLabel: "PDF oder Word (.docx) hierher ziehen oder klicken",
      placeholder: "Schreibe oder füge deinen Text hier ein...",
      modeLabel: "Korrekturtyp",
      modeNormal: "Vollständig (Rechtschreibung + Grammatik)",
      modeSpelling: "Nur Rechtschreibung und Zeichensetzung",
      modeFormal: "Formellerer Ton",
      modeSimple: "Einfachere Sprache",
      shortcutHint: "Tastenkürzel: Strg+Eingabe zum Korrigieren",
      btnCorrect: "Text korrigieren",
      loading: "Wird korrigiert...",
      needText: "Bitte Text eingeben.",
      chars: "%n Zeichen",
      words: "%n Wörter",
      placeholderTitle: "Ergebnis",
      placeholderHint: "Der korrigierte Text erscheint hier neben dem Original.",
      loadingTitle: "Wird korrigiert…",
      resultTitle: "Korrigierter Text",
      resultSubtitle: "Geänderte Wörter sind hervorgehoben. Prüfe das Ergebnis und wähle eine Aktion.",
      actionsExport: "Exportieren",
      errorTitle: "Korrektur fehlgeschlagen",
      copy: "Kopieren",
      apply: "In Text übernehmen",
      compare: "Vergleichen",
      hideCompare: "Nur Ergebnis",
      download: "TXT",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      recorrect: "Erneut korrigieren",
      copied: "Kopiert!",
      applied: "Text aktualisiert!",
      copyFail: "Kopieren fehlgeschlagen.",
      pdfFail: "PDF konnte nicht erstellt werden.",
      serverError: "Server nicht erreichbar.",
      unexpected: "Ein unerwarteter Fehler ist aufgetreten.",
      compareOriginal: "Original",
      compareFixed: "Korrigiert",
      progressHint: "Lange Texte können einige Sekunden dauern.",
      signature: "\n\n— Korrigiert mit Ouviescrevi: https://ouviescrevi.pt/de/corretor.html",
      filePdfFail: "PDF konnte nicht gelesen werden.",
      fileDocxFail: "Word-Datei konnte nicht gelesen werden.",
      fileUnsupported: "Format nicht unterstützt. PDF oder DOCX verwenden.",
      historyTitle: "Letzte Korrekturen",
      historyRefresh: "Aktualisieren",
      historyLoading: "Laden…",
      historyEmpty: "Noch keine Korrekturen gespeichert.",
      historyLogin: "Melde dich an, um Korrekturen zu speichern und anzusehen.",
      historyLoginBtn: "Anmelden oder registrieren",
      historyLoaded: "Korrektur aus dem Verlauf geladen.",
      historyDeleted: "Korrektur gelöscht.",
      historyFail: "Verlauf konnte nicht geladen werden.",
      historyDeleteConfirm: "Diese Korrektur aus dem Verlauf löschen?",
      phrases: [
        "Text wird aufmerksam gelesen...",
        "Rechtschreibfehler werden gesucht...",
        "Grammatik und Zeichensetzung prüfen...",
        "Korrekturen anwenden...",
        "Finalen Text vorbereiten...",
      ],
    },
  };

  function t(key) {
    var pack = STRINGS[config.lang] || STRINGS.pt;
    return pack[key] != null ? pack[key] : STRINGS.pt[key];
  }

  function fmt(key, n) {
    return String(t(key)).replace("%n", String(n));
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function isSiteUser() {
    return (
      sessionStorage.getItem("ouviescrevi_site_role") === "user" &&
      sessionStorage.getItem("ouviescrevi_site_session")
    );
  }

  function tokenizeWords(s) {
    return String(s || "")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
  }

  function wordDiffOps(original, corrected) {
    var oldW = tokenizeWords(original);
    var newW = tokenizeWords(corrected);
    var n = oldW.length;
    var m = newW.length;
    var dp = [];
    var i;
    var j;
    for (i = 0; i <= n; i++) {
      dp[i] = [];
      for (j = 0; j <= m; j++) dp[i][j] = 0;
    }
    for (i = 1; i <= n; i++) {
      for (j = 1; j <= m; j++) {
        if (oldW[i - 1].toLowerCase() === newW[j - 1].toLowerCase()) {
          dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
        }
      }
    }
    var ops = [];
    i = n;
    j = m;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && oldW[i - 1].toLowerCase() === newW[j - 1].toLowerCase()) {
        ops.unshift({ type: "same", word: newW[j - 1] });
        i--;
        j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        ops.unshift({ type: "add", word: newW[j - 1] });
        j--;
      } else {
        ops.unshift({ type: "del", word: oldW[i - 1] });
        i--;
      }
    }
    return ops;
  }

  function renderCorrectedDiffHtml(original, corrected) {
    var ops = wordDiffOps(original, corrected);
    return ops
      .map(function (op) {
        if (op.type === "same") return escapeHtml(op.word);
        if (op.type === "add") {
          return '<mark class="oe-cor-diff__chg">' + escapeHtml(op.word) + "</mark>";
        }
        return "";
      })
      .join(" ");
  }

  function renderOriginalDiffHtml(original, corrected) {
    var ops = wordDiffOps(original, corrected);
    var out = [];
    var i = 0;
    while (i < ops.length) {
      if (ops[i].type === "del") {
        var chunk = [];
        while (i < ops.length && ops[i].type === "del") {
          chunk.push(ops[i].word);
          i++;
        }
        out.push('<del class="oe-cor-diff__del">' + escapeHtml(chunk.join(" ")) + "</del>");
      } else if (ops[i].type === "same") {
        out.push(escapeHtml(ops[i].word));
        i++;
      } else {
        i++;
      }
    }
    return out.join(" ");
  }

  function countWords(text) {
    return tokenizeWords(text).length;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    return iso.replace("T", " ").slice(0, 16);
  }

  function applyFormLabels() {
    var map = {
      corFormEyebrow: "eyebrow",
      corFormTitle: "formTitle",
      corFormHint: "formHint",
      corModoLabel: "modeLabel",
      corDropLabel: "dropLabel",
      corShortcutHint: "shortcutHint",
      corHistoryTitle: "historyTitle",
      corHistoryRefresh: "historyRefresh",
      corHistoryEmpty: "historyEmpty",
      corHistoryLoginText: "historyLogin",
      corHistoryLoginBtn: "historyLoginBtn",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });

    var input = document.getElementById("textoInput");
    if (input) input.placeholder = t("placeholder");

    var btn = document.getElementById("btnCorrigir");
    if (btn) btn.textContent = t("btnCorrect");

    var modo = document.getElementById("corModo");
    if (modo && modo.options.length >= 4) {
      modo.options[0].textContent = t("modeNormal");
      modo.options[1].textContent = t("modeSpelling");
      modo.options[2].textContent = t("modeFormal");
      modo.options[3].textContent = t("modeSimple");
    }

    var hint = document.getElementById("corProgressHint");
    if (hint) hint.textContent = t("progressHint");
  }

  function updateMeta() {
    var input = document.getElementById("textoInput");
    var meta = document.getElementById("corMeta");
    if (!input || !meta) return;
    meta.textContent = fmt("chars", input.value.length) + " · " + fmt("words", countWords(input.value));
  }

  function isWideLayout() {
    return global.matchMedia && global.matchMedia("(min-width: 901px)").matches;
  }

  function setPlaceholderVisible(visible) {
    var placeholder = document.getElementById("corPlaceholder");
    if (placeholder) placeholder.hidden = !visible;
  }

  function applyPlaceholderLabels() {
    var title = document.getElementById("corPlaceholderTitle");
    var hint = document.getElementById("corPlaceholderHint");
    if (title) title.textContent = t("placeholderTitle");
    if (hint) hint.textContent = t("placeholderHint");
  }

  function scrollToResultIfNeeded(out) {
    if (!isWideLayout() && out) {
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function ensureApiReady() {
    if (!global.OuviescreviAPI || !global.OuviescreviAPI.init) {
      return Promise.reject(new Error("api-missing"));
    }
    if (global.OuviescreviAPI.getToken()) {
      return Promise.resolve();
    }
    return apiInitWithTimeout();
  }

  function hideOutput(out) {
    if (!out) return;
    out.hidden = true;
    out.innerHTML = "";
    out.classList.remove("oe-cor-output--error", "oe-cor-output--comparing", "oe-cor-output--split", "oe-cor-output--loading");
    compareVisible = false;
    setPlaceholderVisible(true);
  }

  function showOutputLoading(out) {
    if (!out) return;
    setPlaceholderVisible(false);
    out.hidden = false;
    out.classList.remove("oe-cor-output--error");
    out.classList.add("oe-cor-output--loading");
    out.innerHTML =
      '<div class="oe-cor-output__loading">' +
      '<div class="oe-cor-output__loading-spinner" aria-hidden="true"></div>' +
      '<p class="oe-cor-output__loading-title">' + escapeHtml(t("loadingTitle")) + "</p>" +
      '<p class="oe-cor-output__loading-phrase" id="loadingPhrase"></p>' +
      "</div>";
  }

  function bindOutputActions(out) {
    var actions = {
      "[data-cor-copy]": function (btn) { copyText(lastCorrected, btn); },
      "[data-cor-apply]": function () {
        var input = document.getElementById("textoInput");
        if (input && lastCorrected) {
          input.value = lastCorrected;
          updateMeta();
          if (global.OuviescreviUI) global.OuviescreviUI.toast(t("applied"), "success");
          input.scrollIntoView({ behavior: "smooth", block: "start" });
          input.focus();
        }
      },
      "[data-cor-compare]": function (btn) { toggleCompare(out, btn); },
      "[data-cor-download]": function () { downloadTxt(); },
      "[data-cor-pdf]": function () { exportPdf(); },
      "[data-cor-whatsapp]": function () { shareWhatsApp(); },
      "[data-cor-recorrect]": function () {
        var input = document.getElementById("textoInput");
        if (input) {
          input.scrollIntoView({ behavior: "smooth", block: "start" });
          input.focus();
        }
      },
    };
    Object.keys(actions).forEach(function (sel) {
      var el = out.querySelector(sel);
      if (el) el.addEventListener("click", function () { actions[sel](el); });
    });
  }

  function toggleCompare(out, btn) {
    var panel = out.querySelector("[data-cor-compare-panel]");
    var singleView = out.querySelector("[data-cor-single-view]");
    if (!panel) return;
    compareVisible = !compareVisible;
    panel.hidden = !compareVisible;
    if (singleView) singleView.hidden = compareVisible;
    out.classList.toggle("oe-cor-output--split", compareVisible);
    out.classList.toggle("oe-cor-output--comparing", compareVisible);
    btn.textContent = compareVisible ? t("hideCompare") : t("compare");
  }

  function copyText(text, btn) {
    if (!text) return;
    var done = function () {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copied"), "success");
      if (btn) {
        var prev = btn.textContent;
        btn.textContent = t("copied");
        setTimeout(function () { btn.textContent = prev; }, 1800);
      }
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done).catch(function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
      });
    } else if (global.OuviescreviUI) {
      global.OuviescreviUI.toast(t("copyFail"), "error");
    }
  }

  function downloadTxt() {
    if (!lastCorrected) return;
    var blob = new Blob([lastCorrected + t("signature")], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "texto-corrigido-ouviescrevi.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportPdf() {
    if (!lastCorrected) return;
    loadJsPdf()
      .then(function () {
        if (!global.jspdf || !global.jspdf.jsPDF) throw new Error("jspdf missing");
        var doc = new global.jspdf.jsPDF();
        var lines = doc.splitTextToSize(lastCorrected + t("signature"), 180);
        doc.setFontSize(11);
        doc.text(lines, 14, 20);
        doc.save("texto-corrigido-ouviescrevi.pdf");
      })
      .catch(function () {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("pdfFail"), "error");
      });
  }

  var jspdfPromise = null;

  function loadJsPdf() {
    if (global.jspdf && global.jspdf.jsPDF) return Promise.resolve();
    if (!jspdfPromise) {
      jspdfPromise = new Promise(function (resolve, reject) {
        var src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
        var existing = document.querySelector('script[src="' + src + '"]');
        if (existing) {
          existing.addEventListener("load", resolve, { once: true });
          return;
        }
        var s = document.createElement("script");
        s.src = src;
        s.async = true;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    return jspdfPromise;
  }

  async function shareWhatsApp() {
    if (!lastCorrected) return;
    var body = lastCorrected + t("signature");
    if (body.length > 3500) body = body.slice(0, 3400) + "\n\n[...]";
    global.open("https://api.whatsapp.com/send?text=" + encodeURIComponent(body), "_blank", "noopener");
    try {
      await global.OuviescreviAPI.init();
      await fetch(global.OuviescreviAPI.getBase() + "/notify-whatsapp-share", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI.authJson({
            page: global.location.href,
            note: "Corretor — partilha texto corrigido",
          })
        ),
      });
    } catch (e) { /* opcional */ }
  }

  function showSuccess(out, original, corrected) {
    lastOriginal = original;
    lastCorrected = corrected;
    compareVisible = true;
    var diffHtml = renderCorrectedDiffHtml(original, corrected);
    var origDiffHtml = renderOriginalDiffHtml(original, corrected);
    setPlaceholderVisible(false);
    out.hidden = false;
    out.classList.remove("oe-cor-output--error", "oe-cor-output--loading");
    out.classList.add("oe-cor-output--split");
    out.innerHTML =
      '<header class="oe-cor-output__head">' +
      '<div class="oe-cor-output__status" aria-hidden="true">✓</div>' +
      '<div class="oe-cor-output__title-wrap">' +
      '<h2 class="oe-cor-output__title">' + escapeHtml(t("resultTitle")) + "</h2>" +
      '<p class="oe-cor-output__subtitle">' + escapeHtml(t("resultSubtitle")) + "</p>" +
      "</div></header>" +
      '<div class="oe-cor-compare oe-cor-compare--live" data-cor-compare-panel>' +
      '<div class="oe-cor-compare__col oe-cor-compare__col--original">' +
      '<p class="oe-cor-compare__label">' + escapeHtml(t("compareOriginal")) + "</p>" +
      '<div class="oe-cor-compare__text oe-cor-compare__text--diff">' + origDiffHtml + "</div></div>" +
      '<div class="oe-cor-compare__col oe-cor-compare__col--fixed">' +
      '<p class="oe-cor-compare__label">' + escapeHtml(t("compareFixed")) + "</p>" +
      '<div class="oe-cor-compare__text oe-cor-compare__text--diff">' + diffHtml + "</div></div></div>" +
      '<div class="oe-cor-output__view" data-cor-single-view hidden>' +
      '<div class="oe-cor-output__body oe-cor-output__body--diff">' + diffHtml + "</div>" +
      "</div>" +
      '<footer class="oe-cor-output__toolbar">' +
      '<div class="oe-cor-output__toolbar-row oe-cor-output__toolbar-row--primary">' +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--cta" data-cor-apply>' + escapeHtml(t("apply")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn" data-cor-copy>' + escapeHtml(t("copy")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--whatsapp" data-cor-whatsapp>' + escapeHtml(t("whatsapp")) + "</button>" +
      "</div>" +
      '<div class="oe-cor-output__toolbar-row oe-cor-output__toolbar-row--secondary">' +
      '<div class="oe-cor-output__group">' +
      '<span class="oe-cor-output__group-label">' + escapeHtml(t("actionsExport")) + "</span>" +
      '<div class="oe-cor-output__group-btns">' +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--compact" data-cor-pdf>' + escapeHtml(t("pdf")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--compact" data-cor-download>' + escapeHtml(t("download")) + "</button>" +
      "</div></div>" +
      '<div class="oe-cor-output__group oe-cor-output__group--end">' +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--compact" data-cor-compare>' + escapeHtml(t("hideCompare")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--ghost" data-cor-recorrect>' + escapeHtml(t("recorrect")) + "</button>" +
      "</div></div></footer>";

    bindOutputActions(out);
    scrollToResultIfNeeded(out);
    setTimeout(function () { loadHistory(); }, 500);
  }

  function showError(out, message) {
    lastOriginal = "";
    lastCorrected = "";
    setPlaceholderVisible(false);
    out.hidden = false;
    out.classList.remove("oe-cor-output--loading");
    out.classList.add("oe-cor-output--error");
    out.innerHTML =
      '<div class="oe-cor-output__head"><h2 class="oe-cor-output__title">' + escapeHtml(t("errorTitle")) + "</h2></div>" +
      '<pre class="oe-cor-output__body">' + escapeHtml(message) + "</pre>";
    scrollToResultIfNeeded(out);
  }

  async function loadHistory() {
    var panel = document.getElementById("corHistoryPanel");
    var list = document.getElementById("corHistoryList");
    var empty = document.getElementById("corHistoryEmpty");
    var login = document.getElementById("corHistoryLogin");
    var logged = document.getElementById("corHistoryLogged");
    if (!panel) return;

    panel.hidden = false;
    if (!isSiteUser()) {
      if (login) login.hidden = false;
      if (logged) logged.hidden = true;
      return;
    }
    if (login) login.hidden = true;
    if (logged) logged.hidden = false;
    if (!list) return;

    list.innerHTML = "<li class='oe-cor-history__loading'>" + escapeHtml(t("historyLoading")) + "</li>";
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/corrections?limit=20", {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error("fail");
      var data = await res.json();
      var items = data.items || [];
      list.innerHTML = "";
      if (!items.length) {
        if (empty) empty.hidden = false;
        return;
      }
      if (empty) empty.hidden = true;
      items.forEach(function (row) {
        var li = document.createElement("li");
        li.className = "oe-cor-history__item";
        var preview = (row.preview || "").trim();
        li.innerHTML =
          '<button type="button" class="oe-cor-history__open">' +
          '<span class="oe-cor-history__preview">' + escapeHtml(preview || "—") + "</span>" +
          '<span class="oe-cor-history__meta">' + escapeHtml(formatDate(row.created_at)) + "</span>" +
          "</button>" +
          '<button type="button" class="oe-cor-history__del" aria-label="Apagar">&times;</button>';
        li.querySelector(".oe-cor-history__open").addEventListener("click", function () {
          openHistoryItem(row.id);
        });
        li.querySelector(".oe-cor-history__del").addEventListener("click", function (e) {
          e.stopPropagation();
          deleteHistoryItem(row.id);
        });
        list.appendChild(li);
      });
    } catch (e) {
      list.innerHTML = "<li class='oe-cor-history__empty'>" + escapeHtml(t("historyFail")) + "</li>";
    }
  }

  async function openHistoryItem(id) {
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/corrections/" + id, {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error("fail");
      var row = await res.json();
      var input = document.getElementById("textoInput");
      var out = document.getElementById("resultado");
      if (input) {
        input.value = row.original_text || "";
        updateMeta();
      }
      if (out && row.original_text && row.corrected_text) {
        showSuccess(out, row.original_text, row.corrected_text);
      }
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("historyLoaded"), "success");
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("historyFail"), "error");
    }
  }

  async function deleteHistoryItem(id) {
    if (!confirm(t("historyDeleteConfirm"))) return;
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/corrections/" + id, {
        method: "DELETE",
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error("fail");
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("historyDeleted"), "success");
      loadHistory();
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("historyFail"), "error");
    }
  }

  function apiInitWithTimeout(ms) {
    ms = ms || 15000;
    if (!global.OuviescreviAPI || !global.OuviescreviAPI.init) {
      return Promise.reject(new Error("api-missing"));
    }
    return Promise.race([
      global.OuviescreviAPI.init(),
      new Promise(function (_, reject) {
        setTimeout(function () { reject(new Error("api-timeout")); }, ms);
      }),
    ]);
  }

  async function correctFromPage() {
    ensureBoot();
    var input = document.getElementById("textoInput");
    var btn = document.getElementById("btnCorrigir");
    var out = document.getElementById("resultado");
    var progress = document.getElementById("corProgress");
    var modo = document.getElementById("corModo");
    if (!input || !btn || !out) return;

    var texto = input.value.trim();
    if (!texto) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      else alert(t("needText"));
      return;
    }

    showOutputLoading(out);
    if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));
    if (progress) progress.hidden = true;

    var phrases = t("phrases");
    var fraseIndex = 0;
    var interval = setInterval(function () {
      var phraseEl = document.getElementById("loadingPhrase");
      if (phraseEl && phrases.length) {
        phraseEl.textContent = phrases[fraseIndex];
        fraseIndex = (fraseIndex + 1) % phrases.length;
      }
    }, 500);

    try {
      await ensureApiReady();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/correct", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI.authJson({
            text: texto,
            mode: modo ? modo.value : "normal",
            lang: config.apiLang || config.lang || "pt",
          })
        ),
      });
      var data = await res.json().catch(function () { return {}; });
      clearInterval(interval);

      if (res.ok && data.corrected) {
        showSuccess(out, texto, data.corrected);
      } else {
        var detail = data.detail || data.error || t("unexpected");
        if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg || d; }).join(" ");
        showError(out, String(detail));
      }
    } catch (err) {
      clearInterval(interval);
      console.error(err);
      showError(out, t("serverError"));
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function init(opts) {
    if (init._done) return;
    init._done = true;
    config = Object.assign({}, config, opts || {});
    applyFormLabels();
    applyPlaceholderLabels();
    updateMeta();

    var input = document.getElementById("textoInput");
    if (input) {
      input.addEventListener("input", updateMeta);
      input.addEventListener("paste", function () {
        setTimeout(updateMeta, 0);
      });
      input.addEventListener("change", updateMeta);
      input.addEventListener("keydown", function (e) {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
          e.preventDefault();
          correctFromPage();
        }
      });
    }

    var btn = document.getElementById("btnCorrigir");
    if (btn) btn.addEventListener("click", correctFromPage);

    var refresh = document.getElementById("corHistoryRefresh");
    if (refresh) refresh.addEventListener("click", loadHistory);

    var loginBtn = document.getElementById("corHistoryLoginBtn");
    if (loginBtn) {
      loginBtn.addEventListener("click", function (e) {
        e.preventDefault();
        if (global.OuviescreviAuth && global.OuviescreviAuth.openModal) {
          global.OuviescreviAuth.openModal("login");
        }
      });
    }

    if (global.CorretorFiles) {
      global.CorretorFiles.setup({
        onText: function () { updateMeta(); },
        strings: {
          filePdfFail: t("filePdfFail"),
          fileDocxFail: t("fileDocxFail"),
          fileUnsupported: t("fileUnsupported"),
        },
      });
    }

    setTimeout(function () { loadHistory(); }, 0);

    if (global.OuviescreviAPI && global.OuviescreviAPI.init) {
      global.OuviescreviAPI.init().catch(function () {});
    }
  }

  var booted = false;
  var bootAttempts = 0;
  var MAX_BOOT_ATTEMPTS = 10;

  function safeBoot() {
    if (booted) return true;
    if (!document.getElementById("btnCorrigir")) return false;
    try {
      var lang = (document.documentElement.lang || "pt").slice(0, 2);
      var apiLang = document.body.getAttribute("data-cor-api-lang") || lang;
      init({ lang: lang, apiLang: apiLang });
      booted = true;
      return true;
    } catch (err) {
      console.error("[CorretorUI] init failed", err);
      return false;
    }
  }

  function ensureBoot() {
    if (!booted) safeBoot();
    return booted;
  }

  function scheduleBoot() {
    if (booted) return;
    bootAttempts += 1;
    if (safeBoot()) return;
    if (bootAttempts < MAX_BOOT_ATTEMPTS) {
      setTimeout(scheduleBoot, 120 * bootAttempts);
    }
  }

  global.corrigirTexto = function (ev) {
    if (ev && ev.preventDefault) ev.preventDefault();
    ensureBoot();
    correctFromPage();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleBoot);
  } else {
    scheduleBoot();
  }
  global.addEventListener("load", scheduleBoot);

  global.CorretorUI = { init: init, correct: correctFromPage, loadHistory: loadHistory, boot: scheduleBoot };
})(window);
