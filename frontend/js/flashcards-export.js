/**
 * Flashcards — exportar PDF e imprimir com layout personalizado.
 */
(function (global) {
  var jspdfPromise = null;
  var modalEl = null;

  var STRINGS = {
    pt: {
      exportPrint: "Imprimir / PDF",
      exportTitle: "Exportar cartões",
      exportLead: "Escolhe o layout e descarrega PDF ou abre a janela de impressão.",
      layoutLabel: "Cartões por página",
      layout2x2: "2×2 (4 cartões)",
      layout2x3: "2×3 (6 cartões)",
      layout3x3: "3×3 (9 cartões)",
      modeLabel: "Conteúdo",
      modeStudy: "Frente + verso (ficha de estudo)",
      modeFront: "Só frentes (perguntas)",
      modeBack: "Só versos (respostas)",
      modeFold: "Frente | verso lado a lado (cortar/dobrar)",
      modeDuplex: "Frentes e versos separados (impressão frente e verso)",
      fontLabel: "Tamanho do texto",
      fontSm: "Pequeno",
      fontMd: "Médio",
      fontLg: "Grande",
      showTitle: "Título em cada página",
      showNumbers: "Numerar cartões",
      cutLines: "Linhas de corte",
      btnPdf: "📄 Descarregar PDF",
      btnPrint: "🖨️ Imprimir",
      btnClose: "Fechar",
      exportedPdf: "PDF descarregado!",
      pdfFail: "Não foi possível gerar PDF.",
      printHint: "Na impressão, escolhe A4 e margens mínimas.",
      front: "Frente",
      back: "Verso",
      duplexNote: "Modo frente e verso: imprime todas as páginas de frentes, vira a folha e imprime as páginas de versos.",
    },
    en: {
      exportPrint: "Print / PDF",
      exportTitle: "Export cards",
      exportLead: "Pick a layout and download PDF or open the print dialog.",
      layoutLabel: "Cards per page",
      layout2x2: "2×2 (4 cards)",
      layout2x3: "2×3 (6 cards)",
      layout3x3: "3×3 (9 cards)",
      modeLabel: "Content",
      modeStudy: "Front + back (study sheet)",
      modeFront: "Fronts only (questions)",
      modeBack: "Backs only (answers)",
      modeFold: "Front | back side by side (cut/fold)",
      modeDuplex: "Separate fronts and backs (duplex printing)",
      fontLabel: "Text size",
      fontSm: "Small",
      fontMd: "Medium",
      fontLg: "Large",
      showTitle: "Title on each page",
      showNumbers: "Number cards",
      cutLines: "Cut lines",
      btnPdf: "📄 Download PDF",
      btnPrint: "🖨️ Print",
      btnClose: "Close",
      exportedPdf: "PDF downloaded!",
      pdfFail: "Could not generate PDF.",
      printHint: "When printing, choose A4 and minimal margins.",
      front: "Front",
      back: "Back",
      duplexNote: "Duplex mode: print all front pages, flip the stack, then print back pages.",
    },
    es: {
      exportPrint: "Imprimir / PDF",
      exportTitle: "Exportar tarjetas",
      exportLead: "Elige el diseño y descarga PDF o abre la ventana de impresión.",
      layoutLabel: "Tarjetas por página",
      layout2x2: "2×2 (4 tarjetas)",
      layout2x3: "2×3 (6 tarjetas)",
      layout3x3: "3×3 (9 tarjetas)",
      modeLabel: "Contenido",
      modeStudy: "Anverso + reverso (ficha de estudio)",
      modeFront: "Solo anversos (preguntas)",
      modeBack: "Solo reversos (respuestas)",
      modeFold: "Anverso | reverso lado a lado (cortar/doblar)",
      modeDuplex: "Anversos y reversos separados (impresión a doble cara)",
      fontLabel: "Tamaño del texto",
      fontSm: "Pequeño",
      fontMd: "Mediano",
      fontLg: "Grande",
      showTitle: "Título en cada página",
      showNumbers: "Numerar tarjetas",
      cutLines: "Líneas de corte",
      btnPdf: "📄 Descargar PDF",
      btnPrint: "🖨️ Imprimir",
      btnClose: "Cerrar",
      exportedPdf: "¡PDF descargado!",
      pdfFail: "No se pudo generar PDF.",
      printHint: "Al imprimir, elige A4 y márgenes mínimos.",
      front: "Anverso",
      back: "Reverso",
      duplexNote: "Modo doble cara: imprime anversos, gira la pila e imprime reversos.",
    },
    fr: {
      exportPrint: "Imprimer / PDF",
      exportTitle: "Exporter les cartes",
      exportLead: "Choisissez la mise en page et téléchargez le PDF ou imprimez.",
      layoutLabel: "Cartes par page",
      layout2x2: "2×2 (4 cartes)",
      layout2x3: "2×3 (6 cartes)",
      layout3x3: "3×3 (9 cartes)",
      modeLabel: "Contenu",
      modeStudy: "Recto + verso (fiche d'étude)",
      modeFront: "Rectos seulement (questions)",
      modeBack: "Versos seulement (réponses)",
      modeFold: "Recto | verso côte à côte (couper/plier)",
      modeDuplex: "Rectos et versos séparés (impression recto-verso)",
      fontLabel: "Taille du texte",
      fontSm: "Petit",
      fontMd: "Moyen",
      fontLg: "Grand",
      showTitle: "Titre sur chaque page",
      showNumbers: "Numéroter les cartes",
      cutLines: "Lignes de découpe",
      btnPdf: "📄 Télécharger PDF",
      btnPrint: "🖨️ Imprimer",
      btnClose: "Fermer",
      exportedPdf: "PDF téléchargé !",
      pdfFail: "Impossible de générer le PDF.",
      printHint: "À l'impression, choisissez A4 et marges minimales.",
      front: "Recto",
      back: "Verso",
      duplexNote: "Mode recto-verso : imprimez les rectos, retournez la pile, puis les versos.",
    },
    de: {
      exportPrint: "Drucken / PDF",
      exportTitle: "Karten exportieren",
      exportLead: "Layout wählen und PDF herunterladen oder Druckdialog öffnen.",
      layoutLabel: "Karten pro Seite",
      layout2x2: "2×2 (4 Karten)",
      layout2x3: "2×3 (6 Karten)",
      layout3x3: "3×3 (9 Karten)",
      modeLabel: "Inhalt",
      modeStudy: "Vorder- + Rückseite (Lernblatt)",
      modeFront: "Nur Vorderseiten (Fragen)",
      modeBack: "Nur Rückseiten (Antworten)",
      modeFold: "Vorder- | Rückseite nebeneinander (schneiden/falten)",
      modeDuplex: "Getrennte Vorder- und Rückseiten (Duplexdruck)",
      fontLabel: "Textgröße",
      fontSm: "Klein",
      fontMd: "Mittel",
      fontLg: "Groß",
      showTitle: "Titel auf jeder Seite",
      showNumbers: "Karten nummerieren",
      cutLines: "Schnittlinien",
      btnPdf: "📄 PDF herunterladen",
      btnPrint: "🖨️ Drucken",
      btnClose: "Schließen",
      exportedPdf: "PDF heruntergeladen!",
      pdfFail: "PDF konnte nicht erstellt werden.",
      printHint: "Beim Drucken A4 und minimale Ränder wählen.",
      front: "Vorderseite",
      back: "Rückseite",
      duplexNote: "Duplex: zuerst Vorderseiten drucken, stapeln wenden, dann Rückseiten.",
    },
  };

  var GRIDS = {
    "2x2": { cols: 2, rows: 2, perPage: 4 },
    "2x3": { cols: 2, rows: 3, perPage: 6 },
    "3x3": { cols: 3, rows: 3, perPage: 9 },
  };

  var FONT_SIZES = { sm: { body: 9, label: 7, title: 13 }, md: { body: 10.5, label: 8, title: 15 }, lg: { body: 12, label: 9, title: 17 } };

  function t(lang, key) {
    var loc = STRINGS[lang] || STRINGS.pt;
    return loc[key] || STRINGS.pt[key] || key;
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function slugify(title) {
    return (title || "flashcards-ouviescrevi")
      .replace(/[^\w\s-áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇäöüßÄÖÜ]/g, "")
      .trim()
      .replace(/\s+/g, "-")
      .toLowerCase() || "flashcards-ouviescrevi";
  }

  function loadJspdf() {
    if (global.jspdf && global.jspdf.jsPDF) return Promise.resolve();
    if (!jspdfPromise) {
      jspdfPromise = new Promise(function (resolve, reject) {
        var s = document.createElement("script");
        s.src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
        s.onload = function () {
          resolve();
        };
        s.onerror = reject;
        document.head.appendChild(s);
      });
    }
    return jspdfPromise;
  }

  function readOptions(root, lang) {
    return {
      lang: lang,
      grid: (root.querySelector('[name="fcExpGrid"]:checked') || {}).value || "2x3",
      mode: (root.querySelector("#fcExpMode") || {}).value || "study",
      font: (root.querySelector("#fcExpFont") || {}).value || "md",
      showTitle: !!(root.querySelector("#fcExpTitle") || {}).checked,
      showNumbers: !!(root.querySelector("#fcExpNumbers") || {}).checked,
      cutLines: !!(root.querySelector("#fcExpCut") || {}).checked,
    };
  }

  function chunkCards(cards, perPage) {
    var pages = [];
    for (var i = 0; i < cards.length; i += perPage) {
      pages.push(cards.slice(i, i + perPage));
    }
    return pages;
  }

  function cardSides(card, mode, lang) {
    if (mode === "front") return [{ label: t(lang, "front"), text: card.front }];
    if (mode === "back") return [{ label: t(lang, "back"), text: card.back }];
    if (mode === "fold") {
      return [
        { label: t(lang, "front"), text: card.front, half: true },
        { label: t(lang, "back"), text: card.back, half: true },
      ];
    }
    return [
      { label: t(lang, "front"), text: card.front },
      { label: t(lang, "back"), text: card.back },
    ];
  }

  function printCss(opts) {
    var fs = FONT_SIZES[opts.font] || FONT_SIZES.md;
    var border = opts.cutLines ? "1.5px dashed #94a3b8" : "1px solid #cbd5e1";
    return (
      "@page{size:A4;margin:10mm;}" +
      "*{box-sizing:border-box;}" +
      "body{font-family:Segoe UI,system-ui,sans-serif;margin:0;color:#0f172a;}" +
      ".page{page-break-after:always;padding:0;}" +
      ".page:last-child{page-break-after:auto;}" +
      ".page-title{margin:0 0 8px;font-size:" +
      (fs.title + 2) +
      "px;font-weight:700;}" +
      ".page-note{margin:0 0 10px;font-size:10px;color:#64748b;}" +
      ".grid{display:grid;gap:8px;height:267mm;}" +
      ".grid--2x2{grid-template-columns:1fr 1fr;grid-template-rows:repeat(2,1fr);}" +
      ".grid--2x3{grid-template-columns:1fr 1fr;grid-template-rows:repeat(3,1fr);}" +
      ".grid--3x3{grid-template-columns:repeat(3,1fr);grid-template-rows:repeat(3,1fr);}" +
      ".card{border:" +
      border +
      ";border-radius:8px;padding:0;display:flex;flex-direction:column;overflow:hidden;background:#fff;position:relative;min-height:0;}" +
      ".card--single{padding:16px 18px;display:flex;flex-direction:column;gap:14px;}" +
      ".card--study{justify-content:stretch;}" +
      ".card--fold{flex-direction:row;gap:0;padding:0;}" +
      ".card__side{flex:1;min-height:0;padding:16px 18px;display:flex;flex-direction:column;overflow:hidden;gap:14px;}" +
      ".card__side--back{background:#faf5ff;}" +
      ".card__half{flex:1;padding:16px 18px;display:flex;flex-direction:column;gap:14px;border-right:1px dashed #cbd5e1;overflow:hidden;min-height:0;}" +
      ".card__half:last-child{border-right:none;}" +
      ".card__num{position:absolute;top:8px;right:10px;font-size:9px;color:#94a3b8;font-weight:700;z-index:1;}" +
      ".card__label{display:block;flex-shrink:0;font-size:" +
      fs.label +
      "px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#6d28d9;margin:0;padding:0;line-height:1.35;}" +
      ".card__text{font-size:" +
      fs.body +
      "px;line-height:1.5;white-space:pre-wrap;word-break:break-word;flex:1;min-height:0;overflow:hidden;margin:0;padding:0;}" +
      ".card__divider{flex-shrink:0;height:1px;background:#cbd5e1;margin:0;}" +
      "@media print{body{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}"
    );
  }

  function renderPrintCard(card, idx, opts) {
    var mode = opts.mode;
    var num = opts.showNumbers ? '<span class="card__num">' + (idx + 1) + "</span>" : "";

    if (mode === "fold") {
      return (
        '<article class="card card--fold">' +
        num +
        '<div class="card__half"><span class="card__label">' +
        escapeHtml(t(opts.lang, "front")) +
        '</span><div class="card__text">' +
        escapeHtml(card.front) +
        '</div></div><div class="card__half"><span class="card__label">' +
        escapeHtml(t(opts.lang, "back")) +
        '</span><div class="card__text">' +
        escapeHtml(card.back) +
        "</div></div></article>"
      );
    }

    if (mode === "study") {
      return (
        '<article class="card card--study">' +
        num +
        '<div class="card__side card__side--front"><span class="card__label">' +
        escapeHtml(t(opts.lang, "front")) +
        '</span><div class="card__text">' +
        escapeHtml(card.front) +
        '</div></div><div class="card__divider"></div><div class="card__side card__side--back"><span class="card__label">' +
        escapeHtml(t(opts.lang, "back")) +
        '</span><div class="card__text">' +
        escapeHtml(card.back) +
        "</div></div></article>"
      );
    }

    var label = mode === "back" ? t(opts.lang, "back") : t(opts.lang, "front");
    var text = mode === "back" ? card.back : card.front;
    return (
      '<article class="card card--single">' +
      num +
      '<span class="card__label">' +
      escapeHtml(label) +
      '</span><div class="card__text">' +
      escapeHtml(text) +
      "</div></article>"
    );
  }

  function buildPrintHtml(data, opts, autoPrint) {
    var grid = GRIDS[opts.grid] || GRIDS["2x3"];
    var cards = data.cards || [];
    var title = data.title || "Flashcards";
    var sections = [];

    function addSection(pageCards, mode, note) {
      chunkCards(pageCards, grid.perPage).forEach(function (page) {
        sections.push({ page: page, mode: mode, note: note });
      });
    }

    if (opts.mode === "duplex") {
      addSection(cards, "front", t(opts.lang, "duplexNote"));
      addSection(cards, "back", null);
    } else {
      addSection(cards, opts.mode, null);
    }

    var body = sections
      .map(function (sec) {
        var cardsHtml = sec.page
          .map(function (c, i) {
            return renderPrintCard(c, cards.indexOf(c), Object.assign({}, opts, { mode: sec.mode }));
          })
          .join("");
        return (
          '<section class="page">' +
          (opts.showTitle ? '<h1 class="page-title">' + escapeHtml(title) + "</h1>" : "") +
          (sec.note ? '<p class="page-note">' + escapeHtml(sec.note) + "</p>" : "") +
          '<div class="grid grid--' +
          opts.grid +
          '">' +
          cardsHtml +
          "</div></section>"
        );
      })
      .join("");

    var tail =
      autoPrint !== false
        ? '<p class="page-note" style="padding:12px;text-align:center">' +
          escapeHtml(t(opts.lang, "printHint")) +
          '</p><script>window.onload=function(){setTimeout(function(){window.print();},400);};<\/script>'
        : "";
    return (
      "<!DOCTYPE html><html lang=\"" +
      opts.lang +
      '"><head><meta charset="UTF-8"><title>' +
      escapeHtml(title) +
      "</title><style>" +
      printCss(opts) +
      '</style></head><body>' +
      body +
      tail +
      "</body></html>"
    );
  }

  function openPrint(data, opts) {
    var html = buildPrintHtml(data, opts, true);
    var w = global.open("", "_blank", "noopener,noreferrer");
    if (!w) return;
    w.document.open();
    w.document.write(html);
    w.document.close();
  }

  function drawCardPdf(doc, x, y, w, h, card, idx, opts, sideMode) {
    var fs = FONT_SIZES[opts.font] || FONT_SIZES.md;
    var mode = sideMode || opts.mode;
    var pad = 6;
    var borderStyle = opts.cutLines ? "D" : "S";

    doc.setDrawColor(148, 163, 184);
    doc.setLineWidth(0.4);
    if (borderStyle === "D") doc.setLineDashPattern([2, 2], 0);
    doc.roundedRect(x, y, w, h, 3, 3, borderStyle);
    doc.setLineDashPattern([], 0);

    if (opts.showNumbers) {
      doc.setFontSize(7);
      doc.setTextColor(148, 163, 184);
      doc.text(String(idx + 1), x + w - pad, y + pad + 2, { align: "right" });
    }

    var innerX = x + pad;
    var innerY = y + pad + (opts.showNumbers ? 4 : 0);
    var innerW = w - pad * 2;

    if (mode === "fold") {
      var halfW = innerW / 2 - 2;
      doc.setDrawColor(203, 213, 225);
      doc.line(x + w / 2, y + 4, x + w / 2, y + h - 4);
      drawSideBlock(doc, innerX, innerY, halfW, h - pad * 2, t(opts.lang, "front"), card.front, fs);
      drawSideBlock(doc, innerX + halfW + 4, innerY, halfW, h - pad * 2, t(opts.lang, "back"), card.back, fs);
      return;
    }

    if (mode === "study") {
      var gap = 4;
      var halfH = (h - pad * 2 - gap) / 2;
      drawSideBlock(doc, innerX, innerY, innerW, halfH, t(opts.lang, "front"), card.front, fs);
      doc.setDrawColor(203, 213, 225);
      doc.line(innerX, innerY + halfH + gap / 2, innerX + innerW, innerY + halfH + gap / 2);
      drawSideBlock(doc, innerX, innerY + halfH + gap, innerW, halfH, t(opts.lang, "back"), card.back, fs);
      return;
    }

    var sides = cardSides(card, mode, opts.lang);
    var blockH = (h - pad * 2) / sides.length;
    sides.forEach(function (side, i) {
      drawSideBlock(doc, innerX, innerY + i * blockH, innerW, blockH - 4, side.label, side.text, fs);
      if (i < sides.length - 1) {
        doc.setDrawColor(226, 232, 240);
        doc.line(innerX, innerY + (i + 1) * blockH - 2, innerX + innerW, innerY + (i + 1) * blockH - 2);
      }
    });
  }

  function drawSideBlock(doc, x, y, w, h, label, text, fs) {
    var labelSize = fs.label || 8;
    var bodySize = fs.body || 10;
    var labelGap = Math.max(16, bodySize * 1.1);
    var labelY = y + labelSize + 4;
    doc.setFontSize(labelSize);
    doc.setTextColor(109, 40, 217);
    doc.text(String(label || "").toUpperCase(), x, labelY);
    var textStartY = labelY + labelGap;
    doc.setFontSize(bodySize);
    doc.setTextColor(15, 23, 42);
    var lines = doc.splitTextToSize(String(text || ""), w);
    var lineH = bodySize * 0.52;
    var maxLines = Math.max(1, Math.floor((y + h - textStartY) / lineH));
    doc.text(lines.slice(0, maxLines), x, textStartY);
  }

  async function downloadPdf(data, opts) {
    await loadJspdf();
    if (!global.jspdf || !global.jspdf.jsPDF) throw new Error("jspdf");
    var doc = new global.jspdf.jsPDF({ unit: "pt", format: "a4" });
    var grid = GRIDS[opts.grid] || GRIDS["2x3"];
    var pageW = 595.28;
    var pageH = 841.89;
    var margin = 28;
    var titleH = opts.showTitle ? 22 : 0;
    var noteH = opts.mode === "duplex" ? 14 : 0;
    var usableW = pageW - margin * 2;
    var usableH = pageH - margin * 2 - titleH - noteH;
    var cellW = usableW / grid.cols;
    var cellH = usableH / grid.rows;
    var cards = data.cards || [];
    var title = data.title || "Flashcards";

    var pageCount = 0;

    function renderPages(list, sideMode, withNote) {
      chunkCards(list, grid.perPage).forEach(function (page, pi) {
        if (pageCount > 0) doc.addPage();
        pageCount += 1;
        if (opts.showTitle) {
          doc.setFontSize(14);
          doc.setTextColor(15, 23, 42);
          doc.text(title, margin, margin + 12);
        }
        if (withNote && pi === 0) {
          doc.setFontSize(8);
          doc.setTextColor(100, 116, 139);
          doc.text(t(opts.lang, "duplexNote"), margin, margin + titleH + 8, { maxWidth: usableW });
        }
        page.forEach(function (card, i) {
          var col = i % grid.cols;
          var row = Math.floor(i / grid.cols);
          var x = margin + col * cellW + 3;
          var y = margin + titleH + noteH + row * cellH + 3;
          drawCardPdf(doc, x, y, cellW - 6, cellH - 6, card, cards.indexOf(card), opts, sideMode);
        });
      });
    }

    if (opts.mode === "duplex") {
      renderPages(cards, "front", true);
      renderPages(cards, "back", false);
    } else {
      renderPages(cards, opts.mode, false);
    }

    doc.save(slugify(title) + ".pdf");
  }

  function ensureModal(lang) {
    if (modalEl) return modalEl;
    modalEl = document.createElement("div");
    modalEl.id = "fcExportModal";
    modalEl.className = "oe-fc-export";
    modalEl.hidden = true;
    modalEl.innerHTML =
      '<div class="oe-fc-export__backdrop" data-fc-close></div>' +
      '<div class="oe-fc-export__dialog" role="dialog" aria-modal="true" aria-labelledby="fcExportTitle">' +
      '<header class="oe-fc-export__head"><h2 id="fcExportTitle"></h2><p id="fcExportLead"></p></header>' +
      '<div class="oe-fc-export__grid">' +
      '<fieldset class="oe-fc-export__field"><legend id="fcExpLayoutLabel"></legend>' +
      '<label><input type="radio" name="fcExpGrid" value="2x2"> <span data-k="layout2x2"></span></label>' +
      '<label><input type="radio" name="fcExpGrid" value="2x3" checked> <span data-k="layout2x3"></span></label>' +
      '<label><input type="radio" name="fcExpGrid" value="3x3"> <span data-k="layout3x3"></span></label></fieldset>' +
      '<label class="oe-fc-export__field" for="fcExpMode"><span id="fcExpModeLabel"></span>' +
      '<select id="fcExpMode">' +
      '<option value="study"></option><option value="front"></option><option value="back"></option>' +
      '<option value="fold"></option><option value="duplex"></option></select></label>' +
      '<label class="oe-fc-export__field" for="fcExpFont"><span id="fcExpFontLabel"></span>' +
      '<select id="fcExpFont"><option value="sm"></option><option value="md" selected></option><option value="lg"></option></select></label>' +
      '<label class="oe-fc-export__check"><input type="checkbox" id="fcExpTitle" checked> <span data-k="showTitle"></span></label>' +
      '<label class="oe-fc-export__check"><input type="checkbox" id="fcExpNumbers" checked> <span data-k="showNumbers"></span></label>' +
      '<label class="oe-fc-export__check"><input type="checkbox" id="fcExpCut" checked> <span data-k="cutLines"></span></label>' +
      "</div>" +
      '<footer class="oe-fc-export__foot">' +
      '<button type="button" class="oe-fc-export__btn oe-fc-export__btn--ghost" data-fc-close id="fcExpClose"></button>' +
      '<button type="button" class="oe-fc-export__btn" id="fcExpPrint"></button>' +
      '<button type="button" class="oe-fc-export__btn oe-fc-export__btn--primary" id="fcExpPdf"></button>' +
      "</footer></div>";
    document.body.appendChild(modalEl);

    modalEl.querySelectorAll("[data-fc-close]").forEach(function (el) {
      el.addEventListener("click", close);
    });
    modalEl.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });
    return modalEl;
  }

  function applyModalLabels(lang) {
    if (!modalEl) return;
    modalEl.querySelector("#fcExportTitle").textContent = t(lang, "exportTitle");
    modalEl.querySelector("#fcExportLead").textContent = t(lang, "exportLead");
    modalEl.querySelector("#fcExpLayoutLabel").textContent = t(lang, "layoutLabel");
    modalEl.querySelector("#fcExpModeLabel").textContent = t(lang, "modeLabel");
    modalEl.querySelector("#fcExpFontLabel").textContent = t(lang, "fontLabel");
    modalEl.querySelector("#fcExpClose").textContent = t(lang, "btnClose");
    modalEl.querySelector("#fcExpPrint").textContent = t(lang, "btnPrint");
    modalEl.querySelector("#fcExpPdf").textContent = t(lang, "btnPdf");
    modalEl.querySelectorAll("[data-k]").forEach(function (el) {
      el.textContent = t(lang, el.getAttribute("data-k"));
    });
    var modeSel = modalEl.querySelector("#fcExpMode");
    if (modeSel) {
      modeSel.options[0].textContent = t(lang, "modeStudy");
      modeSel.options[1].textContent = t(lang, "modeFront");
      modeSel.options[2].textContent = t(lang, "modeBack");
      modeSel.options[3].textContent = t(lang, "modeFold");
      modeSel.options[4].textContent = t(lang, "modeDuplex");
    }
    var fontSel = modalEl.querySelector("#fcExpFont");
    if (fontSel) {
      fontSel.options[0].textContent = t(lang, "fontSm");
      fontSel.options[1].textContent = t(lang, "fontMd");
      fontSel.options[2].textContent = t(lang, "fontLg");
    }
  }

  var currentData = null;
  var currentLang = "pt";

  function close() {
    if (modalEl) modalEl.hidden = true;
  }

  function open(data, lang) {
    if (!data || !data.cards || !data.cards.length) return;
    currentData = data;
    currentLang = lang || "pt";
    ensureModal(currentLang);
    applyModalLabels(currentLang);
    modalEl.hidden = false;
    var dialog = modalEl.querySelector(".oe-fc-export__dialog");
    if (dialog) dialog.focus();

    var pdfBtn = modalEl.querySelector("#fcExpPdf");
    var printBtn = modalEl.querySelector("#fcExpPrint");
    if (pdfBtn && !pdfBtn._fcBound) {
      pdfBtn._fcBound = true;
      pdfBtn.addEventListener("click", async function () {
        var opts = readOptions(modalEl, currentLang);
        try {
          await downloadPdf(currentData, opts);
          if (global.OuviescreviUI) global.OuviescreviUI.toast(t(currentLang, "exportedPdf"), "success");
        } catch (e) {
          console.error(e);
          if (global.OuviescreviUI) global.OuviescreviUI.toast(t(currentLang, "pdfFail"), "error");
        }
      });
    }
    if (printBtn && !printBtn._fcBound) {
      printBtn._fcBound = true;
      printBtn.addEventListener("click", function () {
        var opts = readOptions(modalEl, currentLang);
        openPrint(currentData, opts);
      });
    }
  }

  global.FlashcardsExport = {
    open: open,
    close: close,
    label: function (lang) {
      return t(lang || "pt", "exportPrint");
    },
  };
})(typeof window !== "undefined" ? window : this);
