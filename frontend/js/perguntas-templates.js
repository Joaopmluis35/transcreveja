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
      classroomSubtitle: "Folha formatada para alunos — sem respostas",
      classroomHint: "Configura estilo, dados e cotação. Depois imprime ou guarda em PDF.",
      templateStyle: "Estilo do teste",
      styleClassic: "Clássico",
      styleModern: "Moderno",
      styleMinimal: "Simples",
      sheetMode: "Versão",
      modeStudent: "Folha do aluno (sem respostas)",
      modeTeacher: "Grelha do professor (com gabarito)",
      fieldsTitle: "Dados no cabeçalho",
      previewTitle: "Pré-visualização",
      btnPrintTest: "Imprimir folha",
      btnPdfTest: "PDF da folha",
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
      tabStyle: "Estilo",
      tabFields: "Dados",
      tabGrading: "Cotação",
      btnPreview: "Pré-visualizar",
      gradingTitle: "Sistema de notas",
      gradingHint: "Define a cotação de cada pergunta. Aparece no teste impresso e no gabarito.",
      gradingTotal: "Total do teste",
      gradingMode: "Distribuição",
      gradingEqual: "Dividir igualmente",
      gradingCustom: "Definir por pergunta",
      gradingBlank: "Linha em branco (aluno preenche)",
      gradingShowTotal: "Mostrar total no cabeçalho",
      gradingQuestion: "Pergunta",
      totalLabel: "Total",
      closePreview: "Fechar",
      testLanguage: "Idioma do teste",
      testLanguageHint: "Aplica-se às perguntas (ao gerar) e à folha impressa/PDF.",
    },
    en: {
      classroomTitle: "Classroom test template",
      classroomSubtitle: "Formatted sheet for students — no answers",
      classroomHint: "Set style, fields and grading, then print or save as PDF.",
      templateStyle: "Test style",
      styleClassic: "Classic",
      styleModern: "Modern",
      styleMinimal: "Minimal",
      sheetMode: "Version",
      modeStudent: "Student sheet (no answers)",
      modeTeacher: "Teacher key (with answers)",
      fieldsTitle: "Header fields",
      previewTitle: "Preview",
      btnPrintTest: "Print sheet",
      btnPdfTest: "Sheet PDF",
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
      tabStyle: "Style",
      tabFields: "Fields",
      tabGrading: "Grading",
      btnPreview: "Preview",
      gradingTitle: "Grading system",
      gradingHint: "Set points per question. Shown on the printed test and answer key.",
      gradingTotal: "Test total",
      gradingMode: "Distribution",
      gradingEqual: "Split equally",
      gradingCustom: "Custom per question",
      gradingBlank: "Blank line (student fills in)",
      gradingShowTotal: "Show total in header",
      gradingQuestion: "Question",
      totalLabel: "Total",
      closePreview: "Close",
      testLanguage: "Test language",
      testLanguageHint: "Applies to questions (when generating) and the printed/PDF sheet.",
    },
    es: {
      classroomTitle: "Plantilla para el aula",
      classroomSubtitle: "Hoja formateada para alumnos — sin respuestas",
      classroomHint: "Configura estilo, datos y puntuación. Luego imprime o guarda en PDF.",
      templateStyle: "Estilo del test",
      styleClassic: "Clásico",
      styleModern: "Moderno",
      styleMinimal: "Simple",
      sheetMode: "Versión",
      modeStudent: "Hoja del alumno (sin respuestas)",
      modeTeacher: "Gabarito del profesor",
      fieldsTitle: "Datos en la cabecera",
      previewTitle: "Vista previa",
      btnPrintTest: "Imprimir hoja",
      btnPdfTest: "PDF de la hoja",
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
      tabStyle: "Estilo",
      tabFields: "Datos",
      tabGrading: "Puntuación",
      btnPreview: "Vista previa",
      gradingTitle: "Sistema de notas",
      gradingHint: "Define la puntuación de cada pregunta. Aparece en el test impreso y en el gabarito.",
      gradingTotal: "Total del test",
      gradingMode: "Distribución",
      gradingEqual: "Dividir igualmente",
      gradingCustom: "Personalizar por pregunta",
      gradingBlank: "Línea en blanco",
      gradingShowTotal: "Mostrar total en la cabecera",
      gradingQuestion: "Pregunta",
      totalLabel: "Total",
      closePreview: "Cerrar",
      testLanguage: "Idioma del test",
      testLanguageHint: "Afecta a las preguntas (al generar) y a la hoja impresa/PDF.",
    },
    fr: {
      classroomTitle: "Modèle pour la classe",
      classroomSubtitle: "Feuille formatée pour les élèves — sans réponses",
      classroomHint: "Configurez le style, les champs et le barème, puis imprimez ou enregistrez en PDF.",
      templateStyle: "Style du test",
      styleClassic: "Classique",
      styleModern: "Moderne",
      styleMinimal: "Minimal",
      sheetMode: "Version",
      modeStudent: "Feuille élève (sans réponses)",
      modeTeacher: "Corrigé professeur",
      fieldsTitle: "Champs d'en-tête",
      previewTitle: "Aperçu",
      btnPrintTest: "Imprimer la feuille",
      btnPdfTest: "PDF de la feuille",
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
      tabStyle: "Style",
      tabFields: "Données",
      tabGrading: "Barème",
      btnPreview: "Aperçu",
      gradingTitle: "Système de notation",
      gradingHint: "Définissez les points par question. Affiché sur le test imprimé et le corrigé.",
      gradingTotal: "Total du test",
      gradingMode: "Répartition",
      gradingEqual: "Répartir également",
      gradingCustom: "Personnaliser par question",
      gradingBlank: "Ligne vide",
      gradingShowTotal: "Afficher le total en en-tête",
      gradingQuestion: "Question",
      totalLabel: "Total",
      closePreview: "Fermer",
      testLanguage: "Langue du test",
      testLanguageHint: "S'applique aux questions (à la génération) et à la feuille imprimée/PDF.",
    },
    de: {
      classroomTitle: "Klassenzimmer-Vorlage",
      classroomSubtitle: "Formatiertes Blatt für Schüler — ohne Antworten",
      classroomHint: "Stil, Felder und Bewertung einstellen, dann drucken oder als PDF speichern.",
      templateStyle: "Test-Stil",
      styleClassic: "Klassisch",
      styleModern: "Modern",
      styleMinimal: "Schlicht",
      sheetMode: "Version",
      modeStudent: "Schülerblatt (ohne Antworten)",
      modeTeacher: "Lehrerlösung (mit Antworten)",
      fieldsTitle: "Kopfzeilenfelder",
      previewTitle: "Vorschau",
      btnPrintTest: "Blatt drucken",
      btnPdfTest: "Blatt-PDF",
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
      tabStyle: "Stil",
      tabFields: "Daten",
      tabGrading: "Bewertung",
      btnPreview: "Vorschau",
      gradingTitle: "Bewertungssystem",
      gradingHint: "Punkte pro Frage festlegen. Erscheint auf dem Test und in der Lösung.",
      gradingTotal: "Test-Gesamtpunktzahl",
      gradingMode: "Verteilung",
      gradingEqual: "Gleich aufteilen",
      gradingCustom: "Pro Frage anpassen",
      gradingBlank: "Leerzeile",
      gradingShowTotal: "Gesamtpunktzahl in der Kopfzeile",
      gradingQuestion: "Frage",
      totalLabel: "Gesamt",
      closePreview: "Schließen",
      testLanguage: "Testsprache",
      testLanguageHint: "Gilt für Fragen (beim Erstellen) und das Druck-/PDF-Blatt.",
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

  function readConfig(root, lang, questionCount) {
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

    var gradingMode = (root.querySelector('input[name="gradingMode"]:checked') || {}).value || "equal";
    var totalEl = root.querySelector("#gradingTotal");
    var totalPoints = totalEl ? parseFloat(String(totalEl.value).replace(",", ".")) : 20;
    if (!totalPoints || totalPoints < 1) totalPoints = 20;
    var showTotal = root.querySelector("#gradingShowTotal");
    var perQuestion = [];
    if (gradingMode === "custom") {
      for (var i = 0; i < questionCount; i++) {
        var inp = root.querySelector('#gradingQ_' + (i + 1));
        var v = inp ? parseFloat(String(inp.value).replace(",", ".")) : 0;
        perQuestion.push(v > 0 ? v : 1);
      }
    } else if (gradingMode === "equal") {
      var each = Math.round((totalPoints / Math.max(questionCount, 1)) * 10) / 10;
      for (var j = 0; j < questionCount; j++) perQuestion.push(each);
    }

    return {
      style: style,
      mode: mode,
      fields: fields,
      lang: lang,
      grading: {
        mode: gradingMode,
        totalPoints: totalPoints,
        showTotal: !showTotal || showTotal.checked,
        perQuestion: perQuestion,
      },
    };
  }

  function formatPoints(n, lang) {
    if (n === null || n === undefined || isNaN(n)) return "";
    var v = Number(n);
    if (Math.abs(v - Math.round(v)) < 0.01) v = Math.round(v);
    return String(v);
  }

  function pointsLine(cfg, qIndex) {
    var g = cfg.grading || {};
    var unit = tt("testPointsUnit", cfg.lang);
    var label = tt("testPoints", cfg.lang);
    if (g.mode === "blank") {
      return label + ": ______ " + unit;
    }
    var pts = g.perQuestion && g.perQuestion[qIndex];
    if (pts) return label + ": " + formatPoints(pts, cfg.lang) + " " + unit;
    return label + ": ______ " + unit;
  }

  function gradingTotalDisplay(cfg) {
    var g = cfg.grading || {};
    if (!g.showTotal || g.mode === "blank") return "";
    var total = g.totalPoints;
    if (g.mode === "custom" && g.perQuestion && g.perQuestion.length) {
      total = g.perQuestion.reduce(function (a, b) {
        return a + b;
      }, 0);
      total = Math.round(total * 10) / 10;
    }
    return (
      '<span class="oe-test-sheet__total"><strong>' +
      esc(tt("totalLabel", cfg.lang)) +
      ":</strong> " +
      esc(formatPoints(total, cfg.lang)) +
      " " +
      esc(tt("testPointsUnit", cfg.lang)) +
      "</span>"
    );
  }

  function buildSheetHtml(questions, cfg, questionLabel) {
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
    var totalHtml = gradingTotalDisplay(cfg);
    if (totalHtml) meta.push(totalHtml);
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
    questions.forEach(function (q, idx) {
      var g = cfg.grading || {};
      var ptsTag = "";
      if (g.mode !== "blank" && g.perQuestion && g.perQuestion[idx]) {
        ptsTag =
          ' <span class="oe-test-sheet__qpts">(' +
          esc(formatPoints(g.perQuestion[idx], cfg.lang) + " " + tt("testPointsUnit", cfg.lang)) +
          ")</span>";
      }
      html +=
        '<li class="oe-test-sheet__question"><p class="oe-test-sheet__qtext"><span class="oe-test-sheet__qnum">' +
        esc(q.number) +
        ".</span> " +
        esc(q.prompt) +
        ptsTag +
        "</p>";
      if (q.options && q.options.length) {
        html += '<ul class="oe-test-sheet__options">';
        q.options.forEach(function (opt) {
          html += "<li><span class=\"oe-test-sheet__bubble\">" + esc(opt.letter) + "</span> " + esc(opt.text) + "</li>";
        });
        html += "</ul>";
      }
      if ((cfg.grading || {}).mode === "blank") {
        html += '<p class="oe-test-sheet__points">' + esc(pointsLine(cfg, idx)) + "</p>";
      }
      html += "</li>";
    });
    html += "</ol>";

    if (!studentMode) {
      html += '<section class="oe-test-sheet__key"><h3>' + esc(tt("answerKeyTitle", cfg.lang)) + "</h3><ul>";
      questions.forEach(function (q, idx) {
        var pts = (cfg.grading || {}).perQuestion && (cfg.grading || {}).perQuestion[idx];
        var ptsTxt = pts && (cfg.grading || {}).mode !== "blank" ? " [" + formatPoints(pts, cfg.lang) + " " + tt("testPointsUnit", cfg.lang) + "]" : "";
        html += "<li><strong>" + esc(questionLabel) + " " + esc(q.number) + ":</strong> " + esc(q.answer || "—") + esc(ptsTxt);
        if (q.explanation) html += " — <em>" + esc(q.explanation) + "</em>";
        html += "</li>";
      });
      html += "</ul></section>";
    }

    html += '<footer class="oe-test-sheet__footer">' + esc(tt("generatedBy", cfg.lang)) + "</footer>";
    html += "</div>";
    return html;
  }

  function chipRadio(name, value, label, checked) {
    return (
      '<label class="oe-test-builder__chip">' +
      '<input type="radio" name="' +
      esc(name) +
      '" value="' +
      esc(value) +
      '"' +
      (checked ? " checked" : "") +
      ">" +
      "<span>" +
      esc(label) +
      "</span></label>"
    );
  }

  function tabBtn(step, id, label, active) {
    return (
      '<button type="button" class="oe-test-builder__tab' +
      (active ? " is-active" : "") +
      '" data-test-tab="' +
      esc(id) +
      '" role="tab">' +
      '<span class="oe-test-builder__tab-step">' +
      esc(step) +
      "</span>" +
      '<span class="oe-test-builder__tab-label">' +
      esc(label) +
      "</span></button>"
    );
  }

  function langOptionsHtml(selected) {
    var langs = [
      { id: "pt", label: "Português" },
      { id: "en", label: "English" },
      { id: "es", label: "Español" },
      { id: "fr", label: "Français" },
      { id: "de", label: "Deutsch" },
    ];
    return langs
      .map(function (l) {
        return (
          '<option value="' +
          l.id +
          '"' +
          (l.id === selected ? " selected" : "") +
          ">" +
          esc(l.label) +
          "</option>"
        );
      })
      .join("");
  }

  function builderHtml(lang, questionCount) {
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
      '<section class="oe-test-builder is-collapsed" id="oeTestBuilder">' +
      '<button type="button" class="oe-test-builder__collapse" id="oeTestBuilderToggle" aria-expanded="false">' +
      '<span class="oe-test-builder__collapse-text">' +
      "<strong>📋 " +
      esc(tt("classroomTitle", lang)) +
      "</strong>" +
      "<small>" +
      esc(tt("classroomSubtitle", lang)) +
      "</small></span>" +
      '<svg class="oe-test-builder__chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>' +
      "</button>" +
      '<div class="oe-test-builder__body" id="oeTestBuilderBody" hidden>' +
      "<p class=\"oe-test-builder__hint\">" +
      esc(tt("classroomHint", lang)) +
      "</p>" +
      '<div class="oe-test-builder__tabs" role="tablist">' +
      tabBtn("1", "style", tt("tabStyle", lang), true) +
      tabBtn("2", "fields", tt("tabFields", lang), false) +
      tabBtn("3", "grading", tt("tabGrading", lang), false) +
      "</div>" +
      '<div class="oe-test-builder__panel-pane is-active" data-test-pane="style">' +
      "<h4>" +
      esc(tt("templateStyle", lang)) +
      "</h4>" +
      '<div class="oe-test-builder__chips">' +
      chipRadio("testStyle", "classic", tt("styleClassic", lang), false) +
      chipRadio("testStyle", "modern", tt("styleModern", lang), true) +
      chipRadio("testStyle", "minimal", tt("styleMinimal", lang), false) +
      "</div>" +
      "<h4>" +
      esc(tt("sheetMode", lang)) +
      "</h4>" +
      '<div class="oe-test-builder__chips oe-test-builder__chips--wide">' +
      chipRadio("testMode", "student", tt("modeStudent", lang), true) +
      chipRadio("testMode", "teacher", tt("modeTeacher", lang), false) +
      "</div>" +
      "<h4>" +
      esc(tt("testLanguage", lang)) +
      "</h4>" +
      '<select id="testSheetLang" class="oe-test-builder__lang-select" aria-label="' +
      esc(tt("testLanguage", lang)) +
      '">' +
      langOptionsHtml(lang) +
      "</select>" +
      "<p class=\"oe-test-builder__subhint\">" +
      esc(tt("testLanguageHint", lang)) +
      "</p></div>" +
      '<div class="oe-test-builder__panel-pane" data-test-pane="fields" hidden>' +
      '<div class="oe-test-builder__fields">' +
      fieldsHtml +
      "</div></div>" +
      '<div class="oe-test-builder__panel-pane" data-test-pane="grading" hidden>' +
      "<h4>" +
      esc(tt("gradingTitle", lang)) +
      "</h4>" +
      "<p class=\"oe-test-builder__subhint\">" +
      esc(tt("gradingHint", lang)) +
      "</p>" +
      '<label class="oe-test-builder__inline"><span>' +
      esc(tt("gradingTotal", lang)) +
      '</span><input type="number" id="gradingTotal" min="1" step="0.5" value="20"></label>' +
      "<h4>" +
      esc(tt("gradingMode", lang)) +
      "</h4>" +
      '<div class="oe-test-builder__chips oe-test-builder__chips--stack">' +
      chipRadio("gradingMode", "equal", tt("gradingEqual", lang), true) +
      chipRadio("gradingMode", "custom", tt("gradingCustom", lang), false) +
      chipRadio("gradingMode", "blank", tt("gradingBlank", lang), false) +
      "</div>" +
      '<label class="oe-test-builder__check"><input type="checkbox" id="gradingShowTotal" checked> ' +
      esc(tt("gradingShowTotal", lang)) +
      "</label>" +
      '<div id="oeGradingQuestions" class="oe-test-builder__grading-grid hidden"></div>' +
      "</div>" +
      '<div class="oe-test-builder__actions">' +
      '<button type="button" class="oe-quiz-btn" id="btnPreviewTest">👁️ ' +
      esc(tt("btnPreview", lang)) +
      "</button>" +
      '<button type="button" class="oe-quiz-btn oe-quiz-btn--primary" id="btnPrintTest">🖨️ ' +
      esc(tt("btnPrintTest", lang)) +
      "</button>" +
      '<button type="button" class="oe-quiz-btn" id="btnPdfTest">📄 ' +
      esc(tt("btnPdfTest", lang)) +
      "</button></div></div>" +
      '<div id="oeTestPreview" hidden></div></section>'
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
      var g = cfg.grading || {};
      if (g.showTotal && g.mode !== "blank") {
        var totalPts = g.totalPoints;
        if (g.mode === "custom" && g.perQuestion && g.perQuestion.length) {
          totalPts = g.perQuestion.reduce(function (a, b) {
            return a + b;
          }, 0);
          totalPts = Math.round(totalPts * 10) / 10;
        }
        line(
          tt("totalLabel", cfg.lang) +
            ": " +
            formatPoints(totalPts, cfg.lang) +
            " " +
            tt("testPointsUnit", cfg.lang),
          10,
          false
        );
      }
      y += 3;

      if (f.studentName) line(tt("fieldStudentName", cfg.lang) + ": _______________________________", 10, false);
      if (f.studentNumber) line(tt("fieldStudentNumber", cfg.lang) + ": _______________", 10, false);
      if (f.studentAge) line(tt("fieldStudentAge", cfg.lang) + ": _______", 10, false);
      if (f.instructions) {
        y += 2;
        line(tt("fieldInstructions", cfg.lang) + ": " + f.instructions, 10, false);
      }
      y += 4;

      questions.forEach(function (q, idx) {
        var ptsSuffix = "";
        if (g.mode !== "blank" && g.perQuestion && g.perQuestion[idx]) {
          ptsSuffix =
            " (" +
            formatPoints(g.perQuestion[idx], cfg.lang) +
            " " +
            tt("testPointsUnit", cfg.lang) +
            ")";
        }
        line(q.number + ". " + q.prompt + ptsSuffix, 11, true);
        q.options.forEach(function (o) {
          line("   " + o.letter + ") " + o.text, 10, false);
        });
        if (g.mode === "blank") {
          line(tt("testPoints", cfg.lang) + ": ______ " + tt("testPointsUnit", cfg.lang), 9, false);
        }
        y += 2;
      });

      if (cfg.mode === "teacher") {
        doc.addPage();
        y = 16;
        line(tt("answerKeyTitle", cfg.lang), 13, true);
        questions.forEach(function (q, idx) {
          var ptsTxt = "";
          if (g.mode !== "blank" && g.perQuestion && g.perQuestion[idx]) {
            ptsTxt =
              " [" +
              formatPoints(g.perQuestion[idx], cfg.lang) +
              " " +
              tt("testPointsUnit", cfg.lang) +
              "]";
          }
          line(questionLabel + " " + q.number + ": " + (q.answer || "—") + ptsTxt, 10, false);
        });
      }

      doc.save("teste-ouviescrevi.pdf");
    } catch (e) {
      if (toast) toast(tt("testPdfFail", cfg.lang), "error");
    }
  }

  function printTest(questions, cfg, questionLabel) {
    var html = buildSheetHtml(questions, cfg, questionLabel, "", "");
    var cssHref = "/css/perguntas.css?v=6";
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
    if (!builderRoot || !questions.length) return;
    var qCount = questions.length;

    function currentSheetLang() {
      var sheetLangSel = builderRoot.querySelector("#testSheetLang");
      if (sheetLangSel) return sheetLangSel.value;
      return lang;
    }

    function getCfg() {
      return readConfig(builderRoot, currentSheetLang(), qCount);
    }

    function buildPreviewHtml() {
      return buildSheetHtml(questions, getCfg(), labels.question);
    }

    var sheetLangSel = builderRoot.querySelector("#testSheetLang");
    if (sheetLangSel) {
      sheetLangSel.value = lang;
      sheetLangSel.addEventListener("change", function () {
        if (global.PerguntasUI && global.PerguntasUI.syncOutputLangSelects) {
          global.PerguntasUI.syncOutputLangSelects(sheetLangSel.value);
        }
      });
    }

    var tabs = builderRoot.querySelectorAll("[data-test-tab]");
    var panes = builderRoot.querySelectorAll("[data-test-pane]");
    tabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        var id = tab.getAttribute("data-test-tab");
        tabs.forEach(function (t) {
          t.classList.toggle("is-active", t === tab);
        });
        panes.forEach(function (p) {
          var match = p.getAttribute("data-test-pane") === id;
          p.classList.toggle("is-active", match);
          p.hidden = !match;
        });
      });
    });

    var toggle = builderRoot.querySelector("#oeTestBuilderToggle");
    var body = builderRoot.querySelector("#oeTestBuilderBody");
    if (toggle && body) {
      toggle.addEventListener("click", function () {
        var open = toggle.getAttribute("aria-expanded") !== "true";
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        body.hidden = !open;
        builderRoot.classList.toggle("is-collapsed", !open);
      });
    }

    var gradingGrid = builderRoot.querySelector("#oeGradingQuestions");

    function readGradingValues() {
      var vals = [];
      for (var i = 0; i < qCount; i++) {
        var inp = builderRoot.querySelector("#gradingQ_" + (i + 1));
        vals.push(inp ? inp.value : "");
      }
      return vals;
    }

    function populateGradingGrid() {
      if (!gradingGrid) return;
      var mode = (builderRoot.querySelector('input[name="gradingMode"]:checked') || {}).value || "equal";
      if (mode !== "custom") {
        gradingGrid.classList.add("hidden");
        gradingGrid.innerHTML = "";
        return;
      }
      gradingGrid.classList.remove("hidden");
      var saved = readGradingValues();
      var totalEl = builderRoot.querySelector("#gradingTotal");
      var total = totalEl ? parseFloat(String(totalEl.value).replace(",", ".")) : 20;
      if (!total || total < 1) total = 20;
      var each = Math.round((total / qCount) * 10) / 10;
      var html = "";
      questions.forEach(function (q, i) {
        var val = saved[i] !== "" ? saved[i] : each;
        html +=
          '<label class="oe-test-builder__grading-row"><span>' +
          esc(tt("gradingQuestion", lang)) +
          " " +
          esc(q.number) +
          '</span><input type="number" id="gradingQ_' +
          (i + 1) +
          '" min="0.5" step="0.5" value="' +
          esc(val) +
          '"></label>';
      });
      gradingGrid.innerHTML = html;
    }

    function onGradingChange() {
      populateGradingGrid();
    }

    builderRoot.querySelectorAll('input[name="gradingMode"]').forEach(function (r) {
      r.addEventListener("change", onGradingChange);
    });
    var totalInput = builderRoot.querySelector("#gradingTotal");
    if (totalInput) {
      totalInput.addEventListener("input", onGradingChange);
    }

    var modalEl = null;
    var escHandler = null;

    function closeModal() {
      if (modalEl) {
        modalEl.remove();
        modalEl = null;
      }
      document.body.classList.remove("oe-test-modal-open");
      if (escHandler) {
        document.removeEventListener("keydown", escHandler);
        escHandler = null;
      }
    }

    function openPreview() {
      closeModal();
      modalEl = document.createElement("div");
      modalEl.className = "oe-test-modal";
      modalEl.innerHTML =
        '<div class="oe-test-modal__backdrop"></div>' +
        '<div class="oe-test-modal__dialog" role="dialog" aria-modal="true" aria-labelledby="oeTestModalTitle">' +
        '<div class="oe-test-modal__head">' +
        '<h3 id="oeTestModalTitle">' +
        esc(tt("previewTitle", lang)) +
        "</h3>" +
        '<button type="button" class="oe-test-modal__close" id="oeTestModalClose">× ' +
        esc(tt("closePreview", lang)) +
        "</button>" +
        "</div>" +
        '<div class="oe-test-modal__body">' +
        buildPreviewHtml() +
        "</div></div>";
      document.body.appendChild(modalEl);
      document.body.classList.add("oe-test-modal-open");
      modalEl.querySelector(".oe-test-modal__backdrop").addEventListener("click", closeModal);
      modalEl.querySelector("#oeTestModalClose").addEventListener("click", closeModal);
      escHandler = function (e) {
        if (e.key === "Escape") closeModal();
      };
      document.addEventListener("keydown", escHandler);
    }

    var btnPreview = builderRoot.querySelector("#btnPreviewTest");
    if (btnPreview) btnPreview.addEventListener("click", openPreview);

    if (previewRoot) previewRoot.hidden = true;

    var btnPrint = builderRoot.querySelector("#btnPrintTest");
    var btnPdf = builderRoot.querySelector("#btnPdfTest");
    if (btnPrint) {
      btnPrint.addEventListener("click", function () {
        printTest(questions, getCfg(), labels.question);
      });
    }
    if (btnPdf) {
      btnPdf.addEventListener("click", function () {
        exportTestPdf(questions, getCfg(), labels.question, toast);
      });
    }

    populateGradingGrid();
  }

  global.PerguntasTemplates = {
    builderHtml: builderHtml,
    mount: mount,
    buildSheetHtml: buildSheetHtml,
    readConfig: readConfig,
    tt: tt,
  };
})(window);
