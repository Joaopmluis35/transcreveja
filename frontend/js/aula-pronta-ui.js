/**
 * Aula Pronta — pacote de estudo (resumo + glossário + perguntas).
 */
(function (global) {
  var STORAGE_KEY = "oe_aula_pronta_text";
  var config = { lang: "pt" };
  var lastPack = null;
  var lastQuestions = [];

  var STRINGS = {
    pt: {
      eyebrow: "Professores · Estudantes · Revisão",
      formTitle: "Texto da aula",
      formHint: "Funciona com transcrições do Ouviescrevi, apontamentos ou qualquer texto longo.",
      placeholder: "Cola aqui a transcrição ou apontamentos da aula…",
      langLabel: "Idioma do pacote",
      countLabel: "Perguntas",
      btnGenerate: "📦 Gerar pacote",
      loading: "A gerar pacote…",
      needText: "Introduz texto para gerar o pacote (mín. ~80 caracteres).",
      errorGenerate: "Erro ao gerar o pacote.",
      truncated: "O texto foi truncado para caber no limite — o pacote baseia-se no início do conteúdo.",
      shortSummary: "Resumo rápido",
      studySummary: "Resumo para estudar",
      keyPoints: "Ideias-chave",
      glossary: "Glossário",
      questions: "Perguntas de revisão",
      copy: "Copiar tudo",
      pdfStudy: "PDF estudo",
      pdfTest: "PDF teste (aluno)",
      copied: "Copiado!",
      copyFail: "Não foi possível copiar.",
      pdfFail: "Não foi possível gerar PDF.",
      correct: "Resposta",
      explanation: "Explicação",
      classroomTitle: "Folha para sala de aula",
      signature: "\n\n— Gerado com Ouviescrevi: https://ouviescrevi.pt/aula-pronta.html",
    },
    en: {
      eyebrow: "Teachers · Students · Revision",
      formTitle: "Lesson text",
      formHint: "Works with Ouviescrevi transcripts, notes or any long text.",
      placeholder: "Paste the lesson transcript or notes here…",
      langLabel: "Pack language",
      countLabel: "Questions",
      btnGenerate: "📦 Generate pack",
      loading: "Generating pack…",
      needText: "Paste some text first (min. ~80 characters).",
      errorGenerate: "Error generating pack.",
      truncated: "Text was truncated to fit the limit — the pack is based on the beginning of the content.",
      shortSummary: "Quick summary",
      studySummary: "Study summary",
      keyPoints: "Key points",
      glossary: "Glossary",
      questions: "Revision questions",
      copy: "Copy all",
      pdfStudy: "Study PDF",
      pdfTest: "Student test PDF",
      copied: "Copied!",
      copyFail: "Could not copy.",
      pdfFail: "Could not generate PDF.",
      correct: "Answer",
      explanation: "Explanation",
      classroomTitle: "Classroom sheet",
      signature: "\n\n— Generated with Ouviescrevi: https://ouviescrevi.pt/en/aula-pronta.html",
    },
  };

  function t(key) {
    var loc = STRINGS[config.lang] || STRINGS.pt;
    return loc[key] || STRINGS.pt[key] || key;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function packQuestionsToQuiz(questions) {
    return (questions || []).map(function (q, idx) {
      var options = [];
      var opts = q.options || {};
      ["A", "B", "C", "D"].forEach(function (letter) {
        if (opts[letter]) options.push({ letter: letter, text: String(opts[letter]) });
      });
      return {
        number: String(idx + 1),
        prompt: String(q.prompt || ""),
        options: options,
        answer: String(q.answer || "").toUpperCase(),
        explanation: String(q.explanation || ""),
      };
    });
  }

  function packToPlainText(pack) {
    if (!pack) return "";
    var lines = [];
    if (pack.title) lines.push(pack.title, "");
    if (pack.short_summary) {
      lines.push(t("shortSummary") + ":", pack.short_summary, "");
    }
    if (pack.study_summary) {
      lines.push(t("studySummary") + ":", pack.study_summary, "");
    }
    if (pack.key_points && pack.key_points.length) {
      lines.push(t("keyPoints") + ":");
      pack.key_points.forEach(function (p) {
        lines.push("• " + p);
      });
      lines.push("");
    }
    if (pack.glossary && pack.glossary.length) {
      lines.push(t("glossary") + ":");
      pack.glossary.forEach(function (g) {
        lines.push((g.term || g.word || "") + ": " + (g.definition || ""));
      });
      lines.push("");
    }
    lastQuestions.forEach(function (q) {
      lines.push("Pergunta " + q.number + ":", q.prompt, "");
      q.options.forEach(function (o) {
        lines.push(o.letter + ") " + o.text);
      });
      if (q.answer) lines.push(t("correct") + ": " + q.answer);
      if (q.explanation) lines.push(t("explanation") + ": " + q.explanation);
      lines.push("");
    });
    return lines.join("\n").trim() + t("signature");
  }

  function renderPack(container, pack, truncated) {
    lastPack = pack;
    lastQuestions = packQuestionsToQuiz(pack.questions);

    var glossaryHtml = "";
    if (pack.glossary && pack.glossary.length) {
      glossaryHtml =
        '<dl class="oe-ap-glossary">' +
        pack.glossary
          .map(function (g) {
            return (
              "<dt>" +
              escapeHtml(g.term || g.word || "") +
              "</dt><dd>" +
              escapeHtml(g.definition || "") +
              "</dd>"
            );
          })
          .join("") +
        "</dl>";
    }

    var keyPointsHtml = "";
    if (pack.key_points && pack.key_points.length) {
      keyPointsHtml =
        "<ul>" +
        pack.key_points.map(function (p) {
          return "<li>" + escapeHtml(p) + "</li>";
        }).join("") +
        "</ul>";
    }

    var questionsHtml = lastQuestions
      .map(function (q) {
        var opts =
          "<ul class='oe-ap-question__opts'>" +
          q.options
            .map(function (o) {
              return "<li>" + escapeHtml(o.letter) + ") " + escapeHtml(o.text) + "</li>";
            })
            .join("") +
          "</ul>";
        var ans = q.answer
          ? '<p class="oe-ap-question__answer"><strong>' +
            escapeHtml(t("correct")) +
            ":</strong> " +
            escapeHtml(q.answer) +
            (q.explanation ? " — " + escapeHtml(q.explanation) : "") +
            "</p>"
          : "";
        return (
          '<article class="oe-ap-question">' +
          '<span class="oe-ap-question__num">#' +
          escapeHtml(q.number) +
          "</span>" +
          '<p class="oe-ap-question__prompt">' +
          escapeHtml(q.prompt) +
          "</p>" +
          opts +
          ans +
          "</article>"
        );
      })
      .join("");

    var warn = truncated
      ? '<p class="oe-ap-pack__warn">' + escapeHtml(t("truncated")) + "</p>"
      : "";

    var classroom =
      lastQuestions.length && global.PerguntasTemplates
        ? '<section class="oe-ap-section oe-ap-section--classroom">' +
          "<h3>" +
          escapeHtml(t("classroomTitle")) +
          "</h3>" +
          '<div id="oeApTestBuilder"></div>' +
          '<div id="oeApTestPreview"></div></section>'
        : "";

    container.innerHTML =
      '<div class="oe-ap-pack">' +
      warn +
      '<header class="oe-ap-pack__head">' +
      '<h2 class="oe-ap-pack__title">' +
      escapeHtml(pack.title || "Aula Pronta") +
      "</h2>" +
      '<div class="oe-ap-pack__actions">' +
      '<button type="button" class="oe-ap-pack__btn" data-ap-export="copy">' +
      escapeHtml(t("copy")) +
      "</button>" +
      '<button type="button" class="oe-ap-pack__btn" data-ap-export="pdf-study">' +
      escapeHtml(t("pdfStudy")) +
      "</button>" +
      '<button type="button" class="oe-ap-pack__btn oe-ap-pack__btn--primary" data-ap-export="pdf-test">' +
      escapeHtml(t("pdfTest")) +
      "</button>" +
      "</div></header>" +
      '<div class="oe-ap-pack__body">' +
      '<section class="oe-ap-section"><h3>' +
      escapeHtml(t("shortSummary")) +
      "</h3><p>" +
      escapeHtml(pack.short_summary || "") +
      "</p></section>" +
      '<section class="oe-ap-section"><h3>' +
      escapeHtml(t("studySummary")) +
      "</h3><p>" +
      escapeHtml(pack.study_summary || "").replace(/\n/g, "<br>") +
      "</p></section>" +
      (keyPointsHtml
        ? '<section class="oe-ap-section"><h3>' +
          escapeHtml(t("keyPoints")) +
          "</h3>" +
          keyPointsHtml +
          "</section>"
        : "") +
      (glossaryHtml
        ? '<section class="oe-ap-section"><h3>' +
          escapeHtml(t("glossary")) +
          "</h3>" +
          glossaryHtml +
          "</section>"
        : "") +
      '<section class="oe-ap-section"><h3>' +
      escapeHtml(t("questions")) +
      "</h3>" +
      questionsHtml +
      "</section>" +
      classroom +
      "</div></div>";

    container.hidden = false;

    container.querySelectorAll("[data-ap-export]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        exportPack(btn.getAttribute("data-ap-export"), btn);
      });
    });

    if (lastQuestions.length && global.PerguntasTemplates && global.PerguntasTemplates.mount) {
      var testLang = (document.getElementById("apLang") || {}).value || config.lang;
      global.PerguntasTemplates.mount(
        container.querySelector("#oeApTestBuilder"),
        container.querySelector("#oeApTestPreview"),
        lastQuestions,
        testLang,
        { question: "Pergunta", correct: t("correct"), explanation: t("explanation") },
        global.OuviescreviUI ? global.OuviescreviUI.toast.bind(global.OuviescreviUI) : null
      );
    }

    container.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(container, message) {
    container.innerHTML = '<pre class="oe-ap-section">' + escapeHtml(message) + "</pre>";
    container.hidden = false;
    lastPack = null;
    lastQuestions = [];
  }

  function exportPack(kind) {
    if (!lastPack) return;
    var text = packToPlainText(lastPack);

    if (kind === "copy") {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(
          function () {
            if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copied"), "success");
          },
          function () {
            if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copyFail"), "error");
          }
        );
      }
      return;
    }

    if (!global.jspdf || !global.jspdf.jsPDF) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("pdfFail"), "error");
      return;
    }

    try {
      var doc = new global.jspdf.jsPDF();
      var y = 18;
      var margin = 14;
      var maxW = 180;

      function writeln(line, size) {
        doc.setFontSize(size || 10);
        var lines = doc.splitTextToSize(String(line || ""), maxW);
        lines.forEach(function (ln) {
          if (y > 280) {
            doc.addPage();
            y = 18;
          }
          doc.text(ln, margin, y);
          y += size && size >= 14 ? 8 : 5.5;
        });
      }

      if (kind === "pdf-study") {
        doc.setFontSize(16);
        writeln(lastPack.title || "Aula Pronta", 16);
        y += 4;
        writeln(t("shortSummary") + ": " + (lastPack.short_summary || ""), 11);
        y += 2;
        writeln(t("studySummary") + ":\n" + (lastPack.study_summary || ""), 10);
        if (lastPack.key_points && lastPack.key_points.length) {
          y += 2;
          writeln(t("keyPoints") + ":", 11);
          lastPack.key_points.forEach(function (p) {
            writeln("• " + p, 10);
          });
        }
        if (lastPack.glossary && lastPack.glossary.length) {
          y += 2;
          writeln(t("glossary") + ":", 11);
          lastPack.glossary.forEach(function (g) {
            writeln((g.term || "") + ": " + (g.definition || ""), 10);
          });
        }
        y += 4;
        writeln(t("questions") + ":", 11);
        lastQuestions.forEach(function (q) {
          writeln("Pergunta " + q.number + ": " + q.prompt, 10);
          q.options.forEach(function (o) {
            writeln("  " + o.letter + ") " + o.text, 10);
          });
          if (q.answer) writeln(t("correct") + ": " + q.answer, 10);
          if (q.explanation) writeln(t("explanation") + ": " + q.explanation, 10);
          y += 2;
        });
        doc.save("aula-pronta-estudo.pdf");
        return;
      }

      if (kind === "pdf-test") {
        writeln(lastPack.title || "Teste", 14);
        y += 4;
        lastQuestions.forEach(function (q) {
          writeln(q.number + ". " + q.prompt, 11);
          q.options.forEach(function (o) {
            writeln("   " + o.letter + ") " + o.text, 10);
          });
          y += 3;
        });
        doc.save("aula-pronta-teste.pdf");
      }
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("pdfFail"), "error");
    }
  }

  async function generate() {
    var textoEl = document.getElementById("texto");
    var btn = document.getElementById("btnAulaPronta");
    var out = document.getElementById("resultado");
    if (!textoEl || !btn || !out) return;

    var texto = textoEl.value.trim();
    if (texto.length < 80) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));
    }

    try {
      await global.OuviescreviAPI.init();
      var langEl = document.getElementById("apLang");
      var numEl = document.getElementById("apNumQuestions");
      var payload = {
        text: texto,
        lang: langEl ? langEl.value : config.lang,
        num_questions: numEl ? parseInt(numEl.value, 10) || 10 : 10,
      };
      var res = await fetch(global.OuviescreviAPI.getBase() + "/generate-aula-pronta", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(global.OuviescreviAPI.authJson(payload)),
      });
      var data = await res.json();
      if (!res.ok) {
        showError(out, data.detail || data.error || t("errorGenerate"));
        return;
      }
      if (data.pack) {
        renderPack(out, data.pack, !!data.truncated);
      } else {
        showError(out, data.detail || t("errorGenerate"));
      }
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("errorGenerate"), "error");
      console.error(e);
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function applyStrings() {
    var map = {
      apFormEyebrow: "eyebrow",
      apFormTitle: "formTitle",
      apFormHint: "formHint",
      apLangLabel: "langLabel",
      apCountLabel: "countLabel",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });
    var texto = document.getElementById("texto");
    if (texto) texto.placeholder = t("placeholder");
    var btn = document.getElementById("btnAulaPronta");
    if (btn) btn.textContent = t("btnGenerate");
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyStrings();

    try {
      var saved = sessionStorage.getItem(STORAGE_KEY);
      if (saved) {
        var ta = document.getElementById("texto");
        if (ta && !ta.value.trim()) ta.value = saved;
        sessionStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {}

    var btn = document.getElementById("btnAulaPronta");
    if (btn) btn.addEventListener("click", generate);
  }

  global.AulaProntaUI = { init: init };
})(typeof window !== "undefined" ? window : this);
