/**
 * Perguntas — renderização em cards + exportação.
 */
(function (global) {
  var STRINGS = {
    pt: {
      eyebrow: "Estudo · Testes · Aulas",
      formTitle: "Texto de origem",
      formHint: "Cola uma aula, artigo ou capítulo. A IA gera perguntas de escolha múltipla com resposta e explicação.",
      placeholder: "Cola aqui o conteúdo a estudar — por exemplo uma aula, artigo ou capítulo...",
      btnGenerate: "🎓 Gerar Perguntas",
      countLabel: "N.º de perguntas",
      countCustom: "Outro",
      countCustomAria: "Número personalizado de perguntas",
      loading: "A gerar perguntas...",
      needText: "Introduz texto para gerar perguntas.",
      errorGenerate: "Erro ao gerar perguntas.",
      resultsReady: "%n perguntas geradas",
      resultsLead: "Revê as perguntas abaixo e escolhe o que fazer a seguir",
      questionsTitle: "As tuas perguntas",
      questionsHint: "Cada pergunta inclui resposta correta e explicação",
      toolsTitle: "O que queres fazer a seguir?",
      shareTitle: "Estudar e partilhar",
      shareHint: "Com respostas e explicações — para rever ou enviar",
      studyPdf: "PDF estudo",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      copy: "Copiar tudo",
      txt: "TXT",
      print: "Imprimir",
      correct: "Resposta correta",
      explanation: "Explicação",
      question: "Pergunta",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      pdfFail: "Não foi possível gerar PDF.",
      signature: "\n\n— Gerado com Ouviescrevi: https://ouviescrevi.pt/perguntas.html",
      empty: "Sem perguntas para exportar.",
      viewGenerated: "Ver perguntas geradas",
    },
    en: {
      eyebrow: "Study · Quizzes · Classes",
      formTitle: "Source text",
      formHint: "Paste a lesson, article or chapter. AI creates multiple-choice questions with answers and explanations.",
      placeholder: "Paste the content to study here — e.g. a lesson, article or chapter...",
      btnGenerate: "🎓 Generate Questions",
      countLabel: "Number of questions",
      countCustom: "Custom",
      countCustomAria: "Custom number of questions",
      loading: "Generating...",
      needText: "Paste some text first.",
      errorGenerate: "Error generating questions.",
      resultsReady: "%n questions generated",
      resultsLead: "Review the questions below and choose your next step",
      questionsTitle: "Your questions",
      questionsHint: "Each question includes the correct answer and an explanation",
      toolsTitle: "What would you like to do next?",
      shareTitle: "Study and share",
      shareHint: "With answers and explanations — to review or send",
      studyPdf: "Study PDF",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      copy: "Copy all",
      txt: "TXT",
      print: "Print",
      correct: "Correct answer",
      explanation: "Explanation",
      question: "Question",
      copied: "Copied!",
      copyFail: "Could not copy.",
      pdfFail: "Could not generate PDF.",
      signature: "\n\n— Generated with Ouviescrevi: https://ouviescrevi.pt/en/perguntas.html",
      empty: "Nothing to export.",
      viewGenerated: "View generated questions",
    },
    es: {
      eyebrow: "Estudio · Exámenes · Clases",
      formTitle: "Texto de origen",
      formHint: "Pega una clase, artículo o capítulo. La IA genera preguntas de opción múltiple con respuesta y explicación.",
      placeholder: "Pega aquí el contenido a estudiar — por ejemplo una clase, artículo o capítulo...",
      btnGenerate: "🎓 Generar preguntas",
      countLabel: "N.º de preguntas",
      countCustom: "Otro",
      countCustomAria: "Número personalizado de preguntas",
      loading: "Generando...",
      needText: "Introduce texto primero.",
      errorGenerate: "Error al generar preguntas.",
      resultsReady: "%n preguntas generadas",
      resultsLead: "Revisa las preguntas abajo y elige el siguiente paso",
      questionsTitle: "Tus preguntas",
      questionsHint: "Cada pregunta incluye respuesta correcta y explicación",
      toolsTitle: "¿Qué quieres hacer ahora?",
      shareTitle: "Estudiar y compartir",
      shareHint: "Con respuestas y explicaciones — para repasar o enviar",
      studyPdf: "PDF estudio",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      copy: "Copiar todo",
      txt: "TXT",
      print: "Imprimir",
      correct: "Respuesta correcta",
      explanation: "Explicación",
      question: "Pregunta",
      copied: "¡Copiado!",
      copyFail: "No se pudo copiar.",
      pdfFail: "No se pudo generar el PDF.",
      signature: "\n\n— Generado con Ouviescrevi: https://ouviescrevi.pt/es/perguntas.html",
      empty: "Nada que exportar.",
      viewGenerated: "Ver preguntas generadas",
    },
    fr: {
      eyebrow: "Révision · Quiz · Cours",
      formTitle: "Texte source",
      formHint: "Collez un cours, un article ou un chapitre. L'IA génère des QCM avec réponses et explications.",
      placeholder: "Collez ici le contenu à étudier — par ex. un cours, article ou chapitre...",
      btnGenerate: "🎓 Générer des questions",
      countLabel: "Nombre de questions",
      countCustom: "Autre",
      countCustomAria: "Nombre personnalisé de questions",
      loading: "Génération...",
      needText: "Collez du texte d'abord.",
      errorGenerate: "Erreur lors de la génération.",
      resultsReady: "%n questions générées",
      resultsLead: "Relisez les questions ci-dessous et choisissez la suite",
      questionsTitle: "Vos questions",
      questionsHint: "Chaque question inclut la bonne réponse et une explication",
      toolsTitle: "Que souhaitez-vous faire ensuite ?",
      shareTitle: "Réviser et partager",
      shareHint: "Avec réponses et explications — pour réviser ou envoyer",
      studyPdf: "PDF révision",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      copy: "Tout copier",
      txt: "TXT",
      print: "Imprimer",
      correct: "Bonne réponse",
      explanation: "Explication",
      question: "Question",
      copied: "Copié !",
      copyFail: "Impossible de copier.",
      pdfFail: "Impossible de générer le PDF.",
      signature: "\n\n— Généré avec Ouviescrevi: https://ouviescrevi.pt/fr/perguntas.html",
      empty: "Rien à exporter.",
      viewGenerated: "Voir les questions générées",
    },
    de: {
      eyebrow: "Lernen · Tests · Unterricht",
      formTitle: "Quelltext",
      formHint: "Füge eine Lektion, einen Artikel oder ein Kapitel ein. Die KI erstellt Multiple-Choice-Fragen mit Antworten und Erklärungen.",
      placeholder: "Füge hier den Lernstoff ein — z. B. eine Lektion, einen Artikel oder ein Kapitel...",
      btnGenerate: "🎓 Fragen generieren",
      countLabel: "Anzahl Fragen",
      countCustom: "Andere",
      countCustomAria: "Benutzerdefinierte Anzahl Fragen",
      loading: "Wird erstellt...",
      needText: "Zuerst Text einfügen.",
      errorGenerate: "Fehler beim Erstellen der Fragen.",
      resultsReady: "%n Fragen erstellt",
      resultsLead: "Sieh dir die Fragen unten an und wähle den nächsten Schritt",
      questionsTitle: "Deine Fragen",
      questionsHint: "Jede Frage enthält die richtige Antwort und eine Erklärung",
      toolsTitle: "Was möchtest du als Nächstes tun?",
      shareTitle: "Lernen und teilen",
      shareHint: "Mit Antworten und Erklärungen — zum Wiederholen oder Senden",
      studyPdf: "Lern-PDF",
      pdf: "PDF",
      whatsapp: "WhatsApp",
      copy: "Alles kopieren",
      txt: "TXT",
      print: "Drucken",
      correct: "Richtige Antwort",
      explanation: "Erklärung",
      question: "Frage",
      copied: "Kopiert!",
      copyFail: "Kopieren fehlgeschlagen.",
      pdfFail: "PDF konnte nicht erstellt werden.",
      signature: "\n\n— Erstellt mit Ouviescrevi: https://ouviescrevi.pt/de/perguntas.html",
      empty: "Nichts zu exportieren.",
      viewGenerated: "Generierte Fragen anzeigen",
    },
  };

  var config = { lang: "pt", numQuestions: 3 };
  var lastPlainText = "";
  var lastQuestions = [];

  function t(key) {
    var pack = STRINGS[config.lang] || STRINGS.pt;
    return pack[key] || STRINGS.pt[key] || key;
  }

  function applyFormLabels() {
    var map = [
      ["quizFormEyebrow", "eyebrow"],
      ["quizFormTitle", "formTitle"],
      ["quizFormHint", "formHint"],
      ["quizCountLabel", "countLabel"],
    ];
    map.forEach(function (pair) {
      var el = document.getElementById(pair[0]);
      if (el) el.textContent = t(pair[1]);
    });
    var textarea = document.getElementById("texto");
    if (textarea) textarea.placeholder = t("placeholder");
    var btn = document.getElementById("btnPerguntas");
    if (btn) btn.textContent = t("btnGenerate");
  }

  var COUNT_PRESETS = [3, 5, 10, 15, 20];
  var COUNT_MAX = 30;

  function readNumQuestions() {
    var sel = document.getElementById("quizNumQuestions");
    if (!sel) return config.numQuestions;
    if (sel.value === "custom") {
      var inp = document.getElementById("quizNumQuestionsCustom");
      var custom = inp ? parseInt(inp.value, 10) : 0;
      if (custom > 0) return Math.min(COUNT_MAX, Math.max(1, custom));
      return config.numQuestions;
    }
    var n = parseInt(sel.value, 10);
    return n > 0 ? n : config.numQuestions;
  }

  function syncCustomCountVisibility() {
    var sel = document.getElementById("quizNumQuestions");
    var inp = document.getElementById("quizNumQuestionsCustom");
    if (!sel || !inp) return;
    var isCustom = sel.value === "custom";
    inp.hidden = !isCustom;
    inp.toggleAttribute("aria-hidden", !isCustom);
    if (isCustom) inp.focus();
  }

  function setupCountSelector() {
    var sel = document.getElementById("quizNumQuestions");
    if (!sel) return;
    var current = config.numQuestions || COUNT_PRESETS[0];
    var isPreset = COUNT_PRESETS.indexOf(current) >= 0;

    sel.innerHTML = "";
    COUNT_PRESETS.forEach(function (n) {
      var opt = document.createElement("option");
      opt.value = String(n);
      opt.textContent = String(n);
      if (n === current) opt.selected = true;
      sel.appendChild(opt);
    });
    var customOpt = document.createElement("option");
    customOpt.value = "custom";
    customOpt.textContent = t("countCustom");
    if (!isPreset) customOpt.selected = true;
    sel.appendChild(customOpt);

    var customInput = document.getElementById("quizNumQuestionsCustom");
    if (customInput) {
      customInput.value = String(isPreset ? 12 : current);
      customInput.setAttribute("aria-label", t("countCustomAria"));
      customInput.max = String(COUNT_MAX);
    }

    syncCustomCountVisibility();
  }

  function stripMd(text) {
    return String(text || "")
      .replace(/\*\*/g, "")
      .replace(/^#{1,6}\s+/gm, "")
      .trim();
  }

  function parseQuestions(raw) {
    var text = String(raw || "").replace(/\r\n/g, "\n").trim();
    if (!text) return [];

    var chunks = text.split(/\n-{3,}\s*\n/);
    if (chunks.length === 1) {
      chunks = text.split(/\n(?=(?:#{1,3}\s*)?\*{0,2}(?:Pergunta|Question|Pregunta|Frage)\s+\d+)/i);
    }
    if (chunks.length === 1 && chunks[0] === text) {
      chunks = [text];
    }

    var items = [];
    chunks.forEach(function (block, idx) {
      var clean = block.trim();
      if (!clean) return;

      var headerMatch = clean.match(
        /^(?:#{1,3}\s*)?\*{0,2}(?:Pergunta|Question|Pregunta|Frage)\s*(\d+)\*{0,2}\s*:?\s*/i
      );
      var number = headerMatch ? headerMatch[1] : String(idx + 1);
      var body = headerMatch ? clean.slice(headerMatch[0].length).trim() : clean;

      var optionRe = /^([A-D])\)\s*(.+)$/gim;
      var options = [];
      var match;
      var firstOptionIndex = -1;
      while ((match = optionRe.exec(body)) !== null) {
        if (firstOptionIndex < 0) firstOptionIndex = match.index;
        options.push({ letter: match[1].toUpperCase(), text: stripMd(match[2]) });
      }

      var prompt = body;
      var answer = "";
      var explanation = "";

      var answerMatch = body.match(
        /(?:\*\*)?(?:Resposta correta|Correct answer|Respuesta correcta|Bonne réponse|Richtige Antwort)(?:\*\*)?\s*:?\s*([A-D])\)?/i
      );
      if (answerMatch) answer = answerMatch[1].toUpperCase();

      var explMatch = body.match(
        /(?:\*\*)?(?:Explicação|Explanation|Explicación|Erklärung)(?:\*\*)?\s*:?\s*([\s\S]+?)(?=\n\s*(?:\*\*)?(?:Resposta|Correct|Respuesta|Bonne|Richtige)|$)/i
      );
      if (explMatch) explanation = stripMd(explMatch[1]);

      if (firstOptionIndex >= 0) {
        prompt = stripMd(body.slice(0, firstOptionIndex));
      } else {
        prompt = stripMd(body.split(/\n(?:\*\*)?(?:Resposta|Correct|Respuesta|Bonne|Richtige)/i)[0]);
      }

      if (!prompt && options.length) {
        prompt = t("question") + " " + number;
      }

      if (prompt || options.length) {
        items.push({
          number: number,
          prompt: prompt,
          options: options,
          answer: answer,
          explanation: explanation,
        });
      }
    });

    return items;
  }

  function toPlainText(questions) {
    if (!questions.length) return lastPlainText || "";
    return questions
      .map(function (q) {
        var lines = [t("question") + " " + q.number + ":", q.prompt, ""];
        q.options.forEach(function (opt) {
          lines.push(opt.letter + ") " + opt.text);
        });
        lines.push("");
        if (q.answer) lines.push(t("correct") + ": " + q.answer + ")");
        if (q.explanation) lines.push(t("explanation") + ": " + q.explanation);
        return lines.join("\n");
      })
      .join("\n\n---\n\n");
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function resultsReadyText(count) {
    return t("resultsReady").replace("%n", String(count));
  }

  function renderQuestions(container, raw) {
    var questions = parseQuestions(raw);
    lastQuestions = questions;
    lastPlainText = questions.length ? toPlainText(questions) : String(raw || "").trim();

    if (!questions.length) {
      container.innerHTML =
        '<pre class="oe-quiz-plain">' + escapeHtml(lastPlainText) + "</pre>";
      container.classList.remove("oe-result--error");
      container.hidden = !lastPlainText;
      return;
    }

    var shareSection =
      '<div class="oe-quiz-panel__section oe-quiz-panel__section--share">' +
      '<h4 class="oe-quiz-panel__section-title">' +
      escapeHtml(t("shareTitle")) +
      "</h3>" +
      '<p class="oe-quiz-panel__section-hint">' +
      escapeHtml(t("shareHint")) +
      "</p>" +
      '<div class="oe-quiz-share__grid">' +
      '<button type="button" class="oe-quiz-share-btn" data-quiz-export="copy">📋 ' +
      escapeHtml(t("copy")) +
      "</button>" +
      '<button type="button" class="oe-quiz-share-btn" data-quiz-export="whatsapp">💬 ' +
      escapeHtml(t("whatsapp")) +
      "</button>" +
      '<button type="button" class="oe-quiz-share-btn" data-quiz-export="txt">📝 ' +
      escapeHtml(t("txt")) +
      "</button>" +
      '<button type="button" class="oe-quiz-share-btn" data-quiz-export="pdf">📄 ' +
      escapeHtml(t("studyPdf")) +
      "</button>" +
      '<button type="button" class="oe-quiz-share-btn" data-quiz-export="print">🖨️ ' +
      escapeHtml(t("print")) +
      "</button>" +
      "</div></div>";

    var builderHtml =
      global.PerguntasTemplates && global.PerguntasTemplates.builderHtml
        ? global.PerguntasTemplates.builderHtml(config.lang, questions.length)
        : "";

    var cards = questions
      .map(function (q) {
        var opts = q.options
          .map(function (opt) {
            var correct = q.answer && opt.letter === q.answer;
            return (
              '<li class="oe-quiz-option' +
              (correct ? " oe-quiz-option--correct" : "") +
              '">' +
              '<span class="oe-quiz-option__letter">' +
              escapeHtml(opt.letter) +
              "</span>" +
              "<span>" +
              escapeHtml(opt.text) +
              "</span></li>"
            );
          })
          .join("");

        var answerHtml = "";
        if (q.answer || q.explanation) {
          answerHtml =
            '<div class="oe-quiz-answer">' +
            (q.answer
              ? '<p class="oe-quiz-answer__row"><span class="oe-quiz-answer__label">' +
                escapeHtml(t("correct")) +
                ":</span> " +
                '<span class="oe-quiz-answer__value">' +
                escapeHtml(q.answer) +
                ")</span></p>"
              : "") +
            (q.explanation
              ? '<p class="oe-quiz-answer__row"><span class="oe-quiz-answer__label">' +
                escapeHtml(t("explanation")) +
                ":</span> " +
                escapeHtml(q.explanation) +
                "</p>"
              : "") +
            "</div>";
        }

        return (
          '<article class="oe-quiz-card">' +
          '<header class="oe-quiz-card__head">' +
          '<span class="oe-quiz-card__num">' +
          escapeHtml(q.number) +
          "</span>" +
          "<h3 class=\"oe-quiz-card__title\">" +
          escapeHtml(q.prompt) +
          "</h3></header>" +
          (opts ? '<ul class="oe-quiz-options">' + opts + "</ul>" : "") +
          '<div class="oe-quiz-card__footer">' +
          answerHtml +
          "</div></article>"
        );
      })
      .join("");

    var questionsSection =
      '<section class="oe-quiz-questions" id="oeQuizQuestions">' +
      '<div class="oe-quiz-questions__head">' +
      "<div>" +
      '<h3 class="oe-quiz-questions__title">' +
      escapeHtml(t("questionsTitle")) +
      "</h3>" +
      '<p class="oe-quiz-questions__hint">' +
      escapeHtml(t("questionsHint")) +
      "</p></div>" +
      '<span class="oe-quiz-questions__badge">' +
      escapeHtml(String(questions.length)) +
      "</span></div>" +
      '<div class="oe-quiz-cards">' +
      cards +
      "</div></section>";

    container.innerHTML =
      '<div class="oe-quiz">' +
      '<section class="oe-quiz-panel">' +
      '<header class="oe-quiz-panel__head">' +
      '<span class="oe-quiz-panel__badge" aria-hidden="true">✓</span>' +
      "<div>" +
      '<h2 class="oe-quiz-panel__title">' +
      escapeHtml(resultsReadyText(questions.length)) +
      "</h2>" +
      '<p class="oe-quiz-panel__lead">' +
      escapeHtml(t("resultsLead")) +
      "</p></div></header>" +
      questionsSection +
      '<div class="oe-quiz-panel__tools">' +
      '<p class="oe-quiz-panel__tools-title">' +
      escapeHtml(t("toolsTitle")) +
      "</p>" +
      shareSection +
      '<div class="oe-quiz-panel__divider" role="separator"></div>' +
      '<div class="oe-quiz-panel__section oe-quiz-panel__section--classroom">' +
      builderHtml +
      "</div></div></section></div>";
    container.classList.remove("oe-result--error");
    container.hidden = false;

    container.querySelectorAll("[data-quiz-export]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        exportQuiz(btn.getAttribute("data-quiz-export"), btn);
      });
    });

    if (global.PerguntasTemplates && global.PerguntasTemplates.mount) {
      global.PerguntasTemplates.mount(
        container.querySelector("#oeTestBuilder"),
        container.querySelector("#oeTestPreview"),
        questions,
        config.lang,
        { question: t("question"), correct: t("correct"), explanation: t("explanation") },
        global.OuviescreviUI ? global.OuviescreviUI.toast.bind(global.OuviescreviUI) : null
      );
    }
  }

  function showError(container, message) {
    container.innerHTML =
      '<pre class="oe-quiz-plain">' + escapeHtml(message) + "</pre>";
    container.classList.add("oe-result--error");
    container.hidden = false;
    lastPlainText = "";
    lastQuestions = [];
  }

  function exportQuiz(kind, btn) {
    var text = lastPlainText;
    if (!text) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("empty"), "error");
      return;
    }

    if (kind === "pdf") {
      if (!global.jspdf || !global.jspdf.jsPDF) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("pdfFail"), "error");
        return;
      }
      try {
        var doc = new global.jspdf.jsPDF();
        var lines = doc.splitTextToSize(text + t("signature"), 180);
        var y = 18;
        doc.setFontSize(11);
        lines.forEach(function (line) {
          if (y > 280) {
            doc.addPage();
            y = 18;
          }
          doc.text(line, 14, y);
          y += 6;
        });
        doc.save("perguntas-ouviescrevi.pdf");
      } catch (e) {
        if (global.OuviescreviUI) global.OuviescreviUI.toast(t("pdfFail"), "error");
      }
      return;
    }

    if (kind === "whatsapp") {
      var waUrl =
        "https://api.whatsapp.com/send?text=" +
        encodeURIComponent(text + t("signature"));
      global.open(waUrl, "_blank", "noopener");
      return;
    }

    if (kind === "copy") {
      var done = function () {
        if (!btn) return;
        var prev = btn.textContent;
        btn.textContent = "✅ " + t("copied");
        setTimeout(function () {
          btn.textContent = prev;
        }, 2000);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text + t("signature")).then(done).catch(function () {
          if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
        });
      } else if (global.OuviescreviUI) {
        global.OuviescreviUI.toast(t("copyFail"), "error");
      }
      return;
    }

    if (kind === "txt") {
      var blob = new Blob([text + t("signature")], {
        type: "text/plain;charset=utf-8",
      });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = "perguntas-ouviescrevi.txt";
      a.click();
      URL.revokeObjectURL(url);
      return;
    }

    if (kind === "print") {
      global.print();
    }
  }

  async function generateFromPage() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnPerguntas");
    var out = document.getElementById("resultado");
    if (!textoEl || !btn || !out) return;

    var texto = textoEl.value.trim();
    if (!texto) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));
    }

    try {
      await global.OuviescreviAPI.init();
      var apiLang = config.lang === "en" ? "en" : "pt";
      var payload = {
        text: texto,
        lang: apiLang,
        num_questions: readNumQuestions(),
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-questions", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (data.questions) {
        renderQuestions(out, data.questions);
        var questionsEl = out.querySelector("#oeQuizQuestions");
        if (questionsEl) {
          questionsEl.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
          out.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      } else {
        showError(out, data.error || data.detail || "Erro inesperado.");
      }
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("errorGenerate"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyFormLabels();
    setupCountSelector();
    var btn = document.getElementById("btnPerguntas");
    if (btn) btn.addEventListener("click", generateFromPage);
    var countSel = document.getElementById("quizNumQuestions");
    if (countSel) {
      countSel.addEventListener("change", function () {
        syncCustomCountVisibility();
        config.numQuestions = readNumQuestions();
      });
    }
    var countCustom = document.getElementById("quizNumQuestionsCustom");
    if (countCustom) {
      countCustom.addEventListener("input", function () {
        config.numQuestions = readNumQuestions();
      });
    }
    global.gerarPerguntas = generateFromPage;
  }

  global.PerguntasUI = {
    init: init,
    render: renderQuestions,
    parse: parseQuestions,
    exportQuiz: exportQuiz,
    generate: generateFromPage,
    getLastQuestions: function () {
      return lastQuestions.slice();
    },
  };
})(window);
