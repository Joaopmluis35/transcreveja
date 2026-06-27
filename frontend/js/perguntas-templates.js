/**
 * Modelos de teste para sala de aula — extensão de PerguntasUI.
 */
(function (global) {
  var FIELD_DEFS = [
    { id: "school", labelKey: "fieldSchool", kind: "fill", defaultOn: true },
    { id: "testTitle", labelKey: "fieldTestTitle", kind: "fill", defaultOn: true, valueKey: "defaultTestTitle" },
    { id: "discipline", labelKey: "fieldDiscipline", kind: "fill", defaultOn: true },
    { id: "className", labelKey: "fieldClass", kind: "fill", defaultOn: true },
    { id: "teacher", labelKey: "fieldTeacher", kind: "fill", defaultOn: false },
    { id: "date", labelKey: "fieldDate", kind: "fill", defaultOn: true, autoDate: true },
    { id: "duration", labelKey: "fieldDuration", kind: "fill", defaultOn: true, phKey: "durationPlaceholder" },
    { id: "studentName", labelKey: "fieldStudentName", kind: "line", defaultOn: true },
    { id: "studentNumber", labelKey: "fieldStudentNumber", kind: "line", defaultOn: true },
    { id: "studentAge", labelKey: "fieldStudentAge", kind: "line", defaultOn: false },
    { id: "instructions", labelKey: "fieldInstructions", kind: "instructions", defaultOn: true },
  ];

  var TEMPLATE_STRINGS = {
    pt: {
      classroomTitle: "Modelo para sala de aula",
      classroomHint: "Escolhe o estilo, os dados do teste e imprime ou guarda em PDF para os alunos.",
      templateStyle: "Estilo do teste",
      styleClassic: "Clássico",
      styleModern: "Moderno",
      styleMinimal: "Simples",
      sheetMode: "Versão",
      modeStudent: "Folha do aluno (sem respostas)",
      modeTeacher: "Grelha do professor (com gabarito)",
      fieldsTitle: "Dados no cabeçalho",
      previewTitle: "Pré-visualização",
      btnPrintTest: "Imprimir teste",
      btnPdfTest: "PDF do teste",
      fieldSchool: "Escola",
      fieldTestTitle: "Título do teste",
      fieldDiscipline: "Disciplina",
      fieldClass: "Turma",
      fieldTeacher: "Professor(a)",
      fieldDate: "Data",
      fieldDuration: "Duração",
      fieldStudentName: "Nome do aluno",
      fieldStudentNumber: "N.º / ID",
      fieldStudentAge: "Idade",
      fieldInstructions: "Instruções",
      defaultTestTitle: "Teste de avaliação",
      durationPlaceholder: "ex. 45 min",
      defaultInstructions: "Lê todas as perguntas com atenção. Escolhe apenas uma resposta por pergunta.",
      testPoints: "Cotação",
      testPointsUnit: "valores",
      answerKeyTitle: "Gabarito",
      generatedBy: "Gerado com Ouviescrevi",
      needQuestions: "Gera perguntas primeiro para criar o teste.",
      testPdfFail: "Não foi possível gerar o PDF do teste.",
    },
    en: {
      classroomTitle: "Classroom test template",
      classroomHint: "Pick a style, set test details, then print or save as PDF for your students.",
      templateStyle: "Test style",
      styleClassic: "Classic",
      styleModern: "Modern",
      styleMinimal: "Minimal",
      sheetMode: "Version",
      modeStudent: "Student sheet (no answers)",
      modeTeacher: "Teacher key (with answers)",
      fieldsTitle: "Header fields",
      previewTitle: "Preview",
      btnPrintTest: "Print test",
      btnPdfTest: "Test PDF",
      fieldSchool: "School",
      fieldTestTitle: "Test title",
      fieldDiscipline: "Subject",
      fieldClass: "Class",
      fieldTeacher: "Teacher",
      fieldDate: "Date",
      fieldDuration: "Duration",
      fieldStudentName: "Student name",
      fieldStudentNumber: "No. / ID",
      fieldStudentAge: "Age",
      fieldInstructions: "Instructions",
      defaultTestTitle: "Assessment test",
      durationPlaceholder: "e.g. 45 min",
      defaultInstructions: "Read all questions carefully. Choose only one answer per question.",
      testPoints: "Points",
      testPointsUnit: "pts",
      answerKeyTitle: "Answer key",
      generatedBy: "Generated with Ouviescrevi",
      needQuestions: "Generate questions first to build the test.",
      testPdfFail: "Could not generate test PDF.",
    },
    es: {
      classroomTitle: "Plantilla para el aula",
      classroomHint: "Elige el estilo, los datos del examen e imprime o guarda en PDF para tus alumnos.",
      templateStyle: "Estilo del test",
      styleClassic: "Clásico",
      styleModern: "Moderno",
      styleMinimal: "Simple",
      sheetMode: "Versión",
      modeStudent: "Hoja del alumno (sin respuestas)",
      modeTeacher: "Gabarito del profesor",
      fieldsTitle: "Datos en la cabecera",
      previewTitle: "Vista previa",
      btnPrintTest: "Imprimir test",
      btnPdfTest: "PDF del test",
      fieldSchool: "Colegio",
      fieldTestTitle: "Título del test",
      fieldDiscipline: "Asignatura",
      fieldClass: "Clase",
      fieldTeacher: "Profesor(a)",
      fieldDate: "Fecha",
      fieldDuration: "Duración",
      fieldStudentName: "Nombre del alumno",
      fieldStudentNumber: "N.º / ID",
      fieldStudentAge: "Edad",
      fieldInstructions: "Instrucciones",
      defaultTestTitle: "Prueba de evaluación",
      durationPlaceholder: "ej. 45 min",
      defaultInstructions: "Lee todas las preguntas con atención. Elige solo una respuesta por pregunta.",
      testPoints: "Puntuación",
      testPointsUnit: "pts",
      answerKeyTitle: "Gabarito",
      generatedBy: "Generado con Ouviescrevi",
      needQuestions: "Genera preguntas primero para crear el test.",
      testPdfFail: "No se pudo generar el PDF del test.",
    },
    fr: {
      classroomTitle: "Modèle pour la classe",
      classroomHint: "Choisissez le style, les informations du test, puis imprimez ou enregistrez en PDF.",
      templateStyle: "Style du test",
      styleClassic: "Classique",
      styleModern: "Moderne",
      styleMinimal: "Minimal",
      sheetMode: "Version",
      modeStudent: "Feuille élève (sans réponses)",
      modeTeacher: "Corrigé professeur",
      fieldsTitle: "Champs d'en-tête",
      previewTitle: "Aperçu",
      btnPrintTest: "Imprimer le test",
      btnPdfTest: "PDF du test",
      fieldSchool: "École",
      fieldTestTitle: "Titre du test",
      fieldDiscipline: "Matière",
      fieldClass: "Classe",
      fieldTeacher: "Enseignant(e)",
      fieldDate: "Date",
      fieldDuration: "Durée",
      fieldStudentName: "Nom de l'élève",
      fieldStudentNumber: "N.º / ID",
      fieldStudentAge: "Âge",
      fieldInstructions: "Consignes",
      defaultTestTitle: "Test d'évaluation",
      durationPlaceholder: "ex. 45 min",
      defaultInstructions: "Lisez toutes les questions attentivement. Une seule réponse par question.",
      testPoints: "Barème",
      testPointsUnit: "pts",
      answerKeyTitle: "Corrigé",
      generatedBy: "Généré avec Ouviescrevi",
      needQuestions: "Générez d'abord des questions pour créer le test.",
      testPdfFail: "Impossible de générer le PDF du test.",
    },
    de: {
      classroomTitle: "Klassenzimmer-Vorlage",
      classroomHint: "Wähle Stil und Testdaten, dann drucken oder als PDF für die Klasse speichern.",
      templateStyle: "Test-Stil",
      styleClassic: "Klassisch",
      styleModern: "Modern",
      styleMinimal: "Schlicht",
      sheetMode: "Version",
      modeStudent: "Schülerblatt (ohne Antworten)",
      modeTeacher: "Lehrerlösung (mit Antworten)",
      fieldsTitle: "Kopfzeilenfelder",
      previewTitle: "Vorschau",
      btnPrintTest: "Test drucken",
      btnPdfTest: "Test-PDF",
      fieldSchool: "Schule",
      fieldTestTitle: "Testtitel",
      fieldDiscipline: "Fach",
      fieldClass: "Klasse",
      fieldTeacher: "Lehrer(in)",
      fieldDate: "Datum",
      fieldDuration: "Dauer",
      fieldStudentName: "Name des Schülers",
      fieldStudentNumber: "Nr. / ID",
      fieldStudentAge: "Alter",
      fieldInstructions: "Anweisungen",
      defaultTestTitle: "Klassenarbeit",
      durationPlaceholder: "z. B. 45 Min",
      defaultInstructions: "Lies alle Fragen sorgfältig. Nur eine Antwort pro Frage.",
      testPoints: "Punkte",
      testPointsUnit: "Pkt",
      answerKeyTitle: "Lösung",
      generatedBy: "Erstellt mit Ouviescrevi",
      needQuestions: "Erstelle zuerst Fragen für den Test.",
      testPdfFail: "Test-PDF konnte nicht erstellt werden.",
    },
  };

  function tt(key, lang) {
    var pack = TEMPLATE_STRINGS[lang] || TEMPLATE_STRINGS.pt;
    return pack[key] || TEMPLATE_STRINGS.pt[key] || key;
  }

  function todayIso() {
    var d = new Date();
    var m = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return d.getFullYear() + "-" + m + "-" + day;
  }

  function formatDisplayDate(iso, lang) {
    if (!iso) return "";
    var p = iso.split("-");
    if (p.length !== 3) return iso;
    if (lang === "en") return p[1] + "/" + p[2] + "/" + p[0];
    return p[2] + "/" + p[1] + "/" + p[0];
  }

  function esc(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function readConfig(root, lang) {
    if (!root) return null;
    var style = (root.querySelector('input[name="testStyle"]:checked') || {}).value || "modern";
    var mode = (root.querySelector('input[name="testMode"]:checked') || {}).value || "student";
    var fields = {};
    FIELD_DEFS.forEach(function (def) {
      var on = root.querySelector("#tf_" + def.id);
      var val = root.querySelector("#tv_" + def.id);
      if (!on || !on.checked) return;
      if (def.kind === "instructions") {
        fields[def.id] = val ? val.value.trim() : tt("defaultInstructions", lang);
      } else if (def.kind === "line") {
        fields[def.id] = true;
      } else {
        var v = val ? val.value.trim() : "";
        if (def.autoDate && !v) v = todayIso();
        fields[def.id] = v;
      }
    });
    return { style: style, mode: mode, fields: fields, lang: lang };
  }

  function buildSheetHtml(questions, cfg, questionLabel, correctLabel, explanationLabel) {
    var f = cfg.fields || {};
    var studentMode = cfg.mode !== "teacher";
    var style = cfg.style || "modern";
    var html = '<div class="oe-test-sheet oe-test-sheet--' + esc(style) + '">';

    html += '<header class="oe-test-sheet__header">';
    if (f.school) html += '<p class="oe-test-sheet__school">' + esc(f.school) + "</p>";
    if (f.testTitle) html += '<h2 class="oe-test-sheet__title">' + esc(f.testTitle) + "</h2>";
    html += '<div class="oe-test-sheet__meta">';

    var meta = [];
    if (f.discipline) meta.push('<span><strong>' + esc(tt("fieldDiscipline", cfg.lang)) + ":</strong> " + esc(f.discipline) + "</span>");
    if (f.className) meta.push('<span><strong>' + esc(tt("fieldClass", cfg.lang)) + ":</strong> " + esc(f.className) + "</span>");
    if (f.teacher) meta.push('<span><strong>' + esc(tt("fieldTeacher", cfg.lang)) + ":</strong> " + esc(f.teacher) + "</span>");
    if (f.date) meta.push('<span><strong>' + esc(tt("fieldDate", cfg.lang)) + ":</strong> " + esc(formatDisplayDate(f.date, cfg.lang)) + "</span>");
    if (f.duration) meta.push('<span><strong>' + esc(tt("fieldDuration", cfg.lang)) + ":</strong> " + esc(f.duration) + "</span>");
    html += meta.join("");
    html += "</div></header>";

    var lines = [];
    if (f.studentName) lines.push({ label: tt("fieldStudentName", cfg.lang) });
    if (f.studentNumber) lines.push({ label: tt("fieldStudentNumber", cfg.lang) });
    if (f.studentAge) lines.push({ label: tt("fieldStudentAge", cfg.lang) });
    if (lines.length) {
      html += '<div class="oe-test-sheet__student">';
      lines.forEach(function (line) {
        html +=
          '<div class="oe-test-sheet__line-field"><span>' +
          esc(line.label) +
          '</span><span class="oe-test-sheet__line"></span></div>';
      });
      html += "</div>";
    }

    if (f.instructions) {
      html +=
        '<div class="oe-test-sheet__instructions"><strong>' +
        esc(tt("fieldInstructions", cfg.lang)) +
        ":</strong> " +
        esc(f.instructions) +
        "</div>";
    }

    html += '<ol class="oe-test-sheet__questions">';
    questions.forEach(function (q) {
      html += '<li class="oe-test-sheet__question"><p class="oe-test-sheet__qtext"><span class="oe-test-sheet__qnum">' + esc(q.number) + ".</span> " + esc(q.prompt) + "</p>";
      if (q.options && q.options.length) {
        html += '<ul class="oe-test-sheet__options">';
        q.options.forEach(function (opt) {
          html += "<li><span class=\"oe-test-sheet__bubble\">" + esc(opt.letter) + "</span> " + esc(opt.text) + "</li>";
        });
        html += "</ul>";
      }
      html += '<p class="oe-test-sheet__points">' + esc(tt("testPoints", cfg.lang)) + ": ______ " + esc(tt("testPointsUnit", cfg.lang)) + "</p>";
      html += "</li>";
    });
    html += "</ol>";

    if (!studentMode) {
      html += '<section class="oe-test-sheet__key"><h3>' + esc(tt("answerKeyTitle", cfg.lang)) + "</h3><ul>";
      questions.forEach(function (q) {
        html += "<li><strong>" + esc(questionLabel) + " " + esc(q.number) + ":</strong> " + esc(q.answer || "—");
        if (q.explanation) html += " — <em>" + esc(q.explanation) + "</em>";
        html += "</li>";
      });
      html += "</ul></section>";
    }

    html += '<footer class="oe-test-sheet__footer">' + esc(tt("generatedBy", cfg.lang)) + "</footer>";
    html += "</div>";
    return html;
  }

  function builderHtml(lang) {
    var fieldsHtml = FIELD_DEFS.map(function (def) {
      var checked = def.defaultOn ? " checked" : "";
      var input = "";
      if (def.kind === "line") {
        input = '<span class="oe-test-builder__line-note">—</span>';
      } else if (def.kind === "instructions") {
        input =
          '<textarea id="tv_' + def.id + '" rows="2" class="oe-test-builder__input">' +
          esc(tt("defaultInstructions", lang)) +
          "</textarea>";
      } else {
        var val = def.valueKey ? tt(def.valueKey, lang) : "";
        if (def.autoDate) val = todayIso();
        var ph = def.phKey ? ' placeholder="' + esc(tt(def.phKey, lang)) + '"' : "";
        var type = def.autoDate ? ' type="date"' : ' type="text"';
        input =
          '<input id="tv_' + def.id + '" class="oe-test-builder__input"' + type + ' value="' + esc(val) + '"' + ph + ">";
      }
      return (
        '<label class="oe-test-builder__field"><input type="checkbox" id="tf_' +
        def.id +
        '"' +
        checked +
        "><span>" +
        esc(tt(def.labelKey, lang)) +
        "</span>" +
        input +
        "</label>"
      );
    }).join("");

    return (
      '<section class="oe-test-builder" id="oeTestBuilder">' +
      '<div class="oe-test-builder__head">' +
      "<h3>📋 " + esc(tt("classroomTitle", lang)) + "</h3>" +
      "<p>" + esc(tt("classroomHint", lang)) + "</p>" +
      "</div>" +
      '<div class="oe-test-builder__grid">' +
      '<div class="oe-test-builder__panel">' +
      "<h4>" + esc(tt("templateStyle", lang)) + "</h4>" +
      '<label class="oe-test-builder__radio"><input type="radio" name="testStyle" value="classic"> ' + esc(tt("styleClassic", lang)) + "</label>" +
      '<label class="oe-test-builder__radio"><input type="radio" name="testStyle" value="modern" checked> ' + esc(tt("styleModern", lang)) + "</label>" +
      '<label class="oe-test-builder__radio"><input type="radio" name="testStyle" value="minimal"> ' + esc(tt("styleMinimal", lang)) + "</label>" +
      "<h4>" + esc(tt("sheetMode", lang)) + "</h4>" +
      '<label class="oe-test-builder__radio"><input type="radio" name="testMode" value="student" checked> ' + esc(tt("modeStudent", lang)) + "</label>" +
      '<label class="oe-test-builder__radio"><input type="radio" name="testMode" value="teacher"> ' + esc(tt("modeTeacher", lang)) + "</label>" +
      '<div class="oe-test-builder__actions">' +
      '<button type="button" class="oe-quiz-btn oe-quiz-btn--primary" id="btnPrintTest">🖨️ ' + esc(tt("btnPrintTest", lang)) + "</button>" +
      '<button type="button" class="oe-quiz-btn" id="btnPdfTest">📄 ' + esc(tt("btnPdfTest", lang)) + "</button>" +
      "</div></div>" +
      '<div class="oe-test-builder__panel">' +
      "<h4>" + esc(tt("fieldsTitle", lang)) + "</h4>" +
      '<div class="oe-test-builder__fields">' + fieldsHtml + "</div>" +
      "</div></div>" +
      '<div class="oe-test-builder__preview-wrap">' +
      "<h4>" + esc(tt("previewTitle", lang)) + "</h4>" +
      '<div class="oe-test-builder__preview" id="oeTestPreview"></div>' +
      "</div></section>"
    );
  }

  function sheetToPlain(questions, cfg, questionLabel, correctLabel) {
    var lines = [];
    var f = cfg.fields || {};
    if (f.school) lines.push(f.school);
    if (f.testTitle) lines.push(f.testTitle);
    if (f.discipline) lines.push(tt("fieldDiscipline", cfg.lang) + ": " + f.discipline);
    if (f.className) lines.push(tt("fieldClass", cfg.lang) + ": " + f.className);
    lines.push("");
    if (f.instructions) lines.push(f.instructions, "");
    questions.forEach(function (q) {
      lines.push(q.number + ". " + q.prompt);
      q.options.forEach(function (o) {
        lines.push("  " + o.letter + ") " + o.text);
      });
      lines.push("");
    });
    if (cfg.mode === "teacher") {
      lines.push("--- " + tt("answerKeyTitle", cfg.lang) + " ---");
      questions.forEach(function (q) {
        lines.push(questionLabel + " " + q.number + ": " + (q.answer || "—"));
      });
    }
    return lines.join("\n");
  }

  function exportTestPdf(questions, cfg, questionLabel, toast) {
    if (!global.jspdf || !global.jspdf.jsPDF) {
      if (toast) toast(tt("testPdfFail", cfg.lang), "error");
      return;
    }
    try {
      var doc = new global.jspdf.jsPDF();
      var y = 16;
      var margin = 14;
      var maxW = 182;
      var f = cfg.fields || {};

      function line(text, size, bold) {
        doc.setFontSize(size || 11);
        doc.setFont(undefined, bold ? "bold" : "normal");
        var rows = doc.splitTextToSize(String(text || ""), maxW);
        rows.forEach(function (row) {
          if (y > 285) {
            doc.addPage();
            y = 16;
          }
          doc.text(row, margin, y);
          y += size === 14 ? 7 : 5.5;
        });
      }

      if (f.school) line(f.school, 10, false);
      if (f.testTitle) line(f.testTitle, 14, true);
      y += 2;
      var meta = [];
      if (f.discipline) meta.push(tt("fieldDiscipline", cfg.lang) + ": " + f.discipline);
      if (f.className) meta.push(tt("fieldClass", cfg.lang) + ": " + f.className);
      if (f.date) meta.push(tt("fieldDate", cfg.lang) + ": " + formatDisplayDate(f.date, cfg.lang));
      if (meta.length) line(meta.join("  |  "), 10, false);
      y += 3;

      if (f.studentName) line(tt("fieldStudentName", cfg.lang) + ": _______________________________", 10, false);
      if (f.studentNumber) line(tt("fieldStudentNumber", cfg.lang) + ": _______________", 10, false);
      if (f.studentAge) line(tt("fieldStudentAge", cfg.lang) + ": _______", 10, false);
      if (f.instructions) {
        y += 2;
        line(tt("fieldInstructions", cfg.lang) + ": " + f.instructions, 10, false);
      }
      y += 4;

      questions.forEach(function (q) {
        line(q.number + ". " + q.prompt, 11, true);
        q.options.forEach(function (o) {
          line("   " + o.letter + ") " + o.text, 10, false);
        });
        line(tt("testPoints", cfg.lang) + ": ______", 9, false);
        y += 2;
      });

      if (cfg.mode === "teacher") {
        doc.addPage();
        y = 16;
        line(tt("answerKeyTitle", cfg.lang), 13, true);
        questions.forEach(function (q) {
          line(questionLabel + " " + q.number + ": " + (q.answer || "—"), 10, false);
        });
      }

      doc.save("teste-ouviescrevi.pdf");
    } catch (e) {
      if (toast) toast(tt("testPdfFail", cfg.lang), "error");
    }
  }

  function printTest(questions, cfg, questionLabel) {
    var html = buildSheetHtml(questions, cfg, questionLabel, "", "");
    var cssHref = "/css/perguntas.css?v=3";
    var w = global.open("", "_blank", "noopener");
    if (!w) return;
    w.document.write(
      "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Teste</title>" +
      '<link rel="stylesheet" href="' + cssHref + '">' +
      "<style>body{margin:0;padding:16px;background:#e5e7eb}@media print{body{background:#fff;padding:0}}</style>" +
      "</head><body>" + html + "<script>window.onload=function(){window.print();};<\/script></body></html>"
    );
    w.document.close();
  }

  function mount(builderRoot, previewRoot, questions, lang, labels, toast) {
    if (!builderRoot || !previewRoot || !questions.length) return;

    function refresh() {
      var cfg = readConfig(builderRoot, lang);
      previewRoot.innerHTML = buildSheetHtml(
        questions,
        cfg,
        labels.question,
        labels.correct,
        labels.explanation
      );
    }

    builderRoot.querySelectorAll("input, textarea, select").forEach(function (el) {
      el.addEventListener("change", refresh);
      el.addEventListener("input", refresh);
    });

    var btnPrint = builderRoot.querySelector("#btnPrintTest");
    var btnPdf = builderRoot.querySelector("#btnPdfTest");
    if (btnPrint) {
      btnPrint.addEventListener("click", function () {
        var cfg = readConfig(builderRoot, lang);
        printTest(questions, cfg, labels.question);
      });
    }
    if (btnPdf) {
      btnPdf.addEventListener("click", function () {
        var cfg = readConfig(builderRoot, lang);
        exportTestPdf(questions, cfg, labels.question, toast);
      });
    }

    refresh();
  }

  global.PerguntasTemplates = {
    builderHtml: builderHtml,
    mount: mount,
    buildSheetHtml: buildSheetHtml,
    readConfig: readConfig,
    tt: tt,
  };
})(window);
