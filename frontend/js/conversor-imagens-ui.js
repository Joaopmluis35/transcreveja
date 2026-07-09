/**
 * Conversor de imagens — múltiplos formatos no browser.
 */
(function (global) {
  "use strict";

  var config = { lang: "pt" };
  var files = [];
  var mode = "convert";
  var avifSupported = null;
  var gifencPromise = null;

  var OUTPUT_FORMATS = [
    { id: "image/png", ext: "png", labelKey: "fmtPng", quality: false },
    { id: "image/jpeg", ext: "jpg", labelKey: "fmtJpeg", quality: true, fillWhite: true },
    { id: "image/webp", ext: "webp", labelKey: "fmtWebp", quality: true },
    { id: "image/avif", ext: "avif", labelKey: "fmtAvif", quality: true, requiresAvif: true },
    { id: "image/bmp", ext: "bmp", labelKey: "fmtBmp", quality: false, encoder: "bmp" },
    { id: "image/gif", ext: "gif", labelKey: "fmtGif", quality: false, encoder: "gif" },
  ];

  var STRINGS = {
    pt: {
      dropTitle: "Arrasta imagens aqui",
      dropHint: "ou clica para escolher — JPG, PNG, WebP, GIF, BMP, SVG…",
      formatLabel: "Converter para",
      qualityLabel: "Qualidade (JPEG / WebP / AVIF)",
      btnConvert: "🖼️ Converter e descarregar",
      btnClear: "Limpar",
      needFile: "Escolhe pelo menos uma imagem.",
      converting: "A converter…",
      done: "Conversão concluída!",
      doneMany: "%n imagens convertidas.",
      error: "Não foi possível converter esta imagem.",
      unsupported: "Formato não suportado neste browser.",
      preview: "Pré-visualização",
      selected: "Selecionado:",
      batch: "Ficheiros na fila:",
      fmtPng: "PNG (.png)",
      fmtJpeg: "JPEG (.jpg)",
      fmtWebp: "WebP (.webp)",
      fmtAvif: "AVIF (.avif)",
      fmtBmp: "BMP (.bmp)",
      fmtGif: "GIF (.gif)",
      modeConvert: "Converter",
      modeCompress: "Comprimir",
      modeResize: "Redimensionar",
      modePdf: "Unir em PDF",
      btnCompress: "📦 Comprimir e descarregar",
      btnResize: "📐 Redimensionar e descarregar",
      btnPdf: "📄 Unir em PDF e descarregar",
      pdfHint: "Cada imagem numa página A4 — ideal para documentos ou apresentações.",
      needTwoForPdf: "Escolhe pelo menos uma imagem.",
      donePdf: "PDF criado com %n imagens!",
      compressHint: "Reduz o tamanho do ficheiro mantendo boa qualidade visual.",
      compressFormatLabel: "Formato de saída",
      compressQualityLabel: "Qualidade",
      maxWidthLabel: "Largura máxima (px)",
      customWidthLabel: "Largura personalizada",
      resizeFormatLabel: "Guardar como",
      doneSize: "Feito! %before → %after (−%pct%)",
    },
    en: {
      dropTitle: "Drop images here",
      dropHint: "or click to browse — JPG, PNG, WebP, GIF, BMP, SVG…",
      formatLabel: "Convert to",
      qualityLabel: "Quality (JPEG / WebP / AVIF)",
      btnConvert: "🖼️ Convert and download",
      btnClear: "Clear",
      needFile: "Choose at least one image.",
      converting: "Converting…",
      done: "Conversion complete!",
      doneMany: "%n images converted.",
      error: "Could not convert this image.",
      unsupported: "Format not supported in this browser.",
      preview: "Preview",
      selected: "Selected:",
      batch: "Queued files:",
      fmtPng: "PNG (.png)",
      fmtJpeg: "JPEG (.jpg)",
      fmtWebp: "WebP (.webp)",
      fmtAvif: "AVIF (.avif)",
      fmtBmp: "BMP (.bmp)",
      fmtGif: "GIF (.gif)",
      modeConvert: "Convert",
      modeCompress: "Compress",
      modeResize: "Resize",
      modePdf: "Merge to PDF",
      btnCompress: "📦 Compress and download",
      btnResize: "📐 Resize and download",
      btnPdf: "📄 Merge to PDF and download",
      pdfHint: "One image per A4 page — great for documents or slide decks.",
      needTwoForPdf: "Choose at least one image.",
      donePdf: "PDF created with %n images!",
      compressHint: "Reduce file size while keeping good visual quality.",
      compressFormatLabel: "Output format",
      compressQualityLabel: "Quality",
      maxWidthLabel: "Max width (px)",
      customWidthLabel: "Custom width",
      resizeFormatLabel: "Save as",
      doneSize: "Done! %before → %after (−%pct%)",
    },
  };

  var jspdfPromise = null;

  function t(key) {
    var loc = STRINGS[config.lang] || STRINGS.pt;
    return loc[key] || STRINGS.pt[key] || key;
  }

  function getFormatById(id) {
    for (var i = 0; i < OUTPUT_FORMATS.length; i++) {
      if (OUTPUT_FORMATS[i].id === id) return OUTPUT_FORMATS[i];
    }
    return OUTPUT_FORMATS[0];
  }

  function probeAvif() {
    if (avifSupported !== null) return Promise.resolve(avifSupported);
    return new Promise(function (resolve) {
      if (!global.document || !document.createElement("canvas").toBlob) {
        avifSupported = false;
        resolve(false);
        return;
      }
      var canvas = document.createElement("canvas");
      canvas.width = canvas.height = 1;
      canvas.toBlob(function (blob) {
        avifSupported = !!blob;
        resolve(avifSupported);
      }, "image/avif", 0.5);
    });
  }

  function populateFormatSelect() {
    var sel = document.getElementById("cimgFormat");
    if (!sel) return;
    var current = sel.value;
    sel.innerHTML = "";
    OUTPUT_FORMATS.forEach(function (fmt) {
      if (fmt.requiresAvif && avifSupported === false) return;
      var opt = document.createElement("option");
      opt.value = fmt.id;
      opt.textContent = t(fmt.labelKey);
      sel.appendChild(opt);
    });
    if (current && sel.querySelector('option[value="' + current + '"]')) {
      sel.value = current;
    }
    updateQualityVisibility();
  }

  function updateQualityVisibility() {
    var sel = document.getElementById("cimgFormat");
    var row = document.getElementById("cimgQualityRow");
    if (!sel || !row) return;
    var fmt = getFormatById(sel.value);
    row.hidden = !fmt.quality;
  }

  function setStatus(msg, kind) {
    var el = document.getElementById("cimgStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.className = "oe-cimg-status" + (kind ? " oe-cimg-status--" + kind : "");
  }

  function renderFileList() {
    var meta = document.getElementById("cimgMeta");
    var batch = document.getElementById("cimgBatch");
    var btn = document.getElementById("btnCimgConvert");
    if (!files.length) {
      if (meta) meta.textContent = "";
      if (batch) batch.hidden = true;
      if (btn) btn.disabled = true;
      return;
    }
    if (meta) {
      meta.textContent =
        t("selected") +
        " " +
        files.length +
        (files.length === 1 ? " — " + files[0].name : "");
    }
    if (batch) {
      if (files.length > 1) {
        batch.hidden = false;
        batch.innerHTML =
          "<strong>" +
          t("batch") +
          "</strong><ul>" +
          files
            .map(function (f) {
              return "<li>" + escapeHtml(f.name) + "</li>";
            })
            .join("") +
          "</ul>";
      } else {
        batch.hidden = true;
      }
    }
    if (btn) btn.disabled = false;
    previewFile(files[0]);
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function previewFile(file) {
    var box = document.getElementById("cimgPreview");
    var img = document.getElementById("cimgPreviewImg");
    if (!box || !img) return;
    var url = URL.createObjectURL(file);
    img.onload = function () {
      URL.revokeObjectURL(url);
    };
    img.src = url;
    box.hidden = false;
  }

  function addFiles(fileList) {
    Array.prototype.forEach.call(fileList || [], function (f) {
      if (!f.type || !f.type.startsWith("image/")) return;
      files.push(f);
    });
    renderFileList();
  }

  function loadImageFromFile(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () {
        URL.revokeObjectURL(url);
        resolve(img);
      };
      img.onerror = function () {
        URL.revokeObjectURL(url);
        reject(new Error("load"));
      };
      img.src = url;
    });
  }

  function drawToCanvas(img, fillWhite) {
    var canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    var ctx = canvas.getContext("2d");
    if (fillWhite) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(img, 0, 0);
    return canvas;
  }

  function canvasToBlob(canvas, mime, quality) {
    return new Promise(function (resolve, reject) {
      if (!canvas.toBlob) {
        reject(new Error("unsupported"));
        return;
      }
      canvas.toBlob(
        function (blob) {
          if (blob) resolve(blob);
          else reject(new Error("blob"));
        },
        mime,
        quality
      );
    });
  }

  function encodeBmp(canvas) {
    var w = canvas.width;
    var h = canvas.height;
    var data = canvas.getContext("2d").getImageData(0, 0, w, h).data;
    var rowSize = Math.ceil((w * 3) / 4) * 4;
    var pixelBytes = rowSize * h;
    var buf = new ArrayBuffer(54 + pixelBytes);
    var view = new DataView(buf);
    view.setUint8(0, 0x42);
    view.setUint8(1, 0x4d);
    view.setUint32(2, 54 + pixelBytes, true);
    view.setUint32(6, 0, true);
    view.setUint32(10, 54, true);
    view.setUint32(14, 40, true);
    view.setInt32(18, w, true);
    view.setInt32(22, -h, true);
    view.setUint16(26, 1, true);
    view.setUint16(28, 24, true);
    view.setUint32(30, 0, true);
    view.setUint32(34, pixelBytes, true);
    var offset = 54;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var i = (y * w + x) * 4;
        view.setUint8(offset++, data[i + 2]);
        view.setUint8(offset++, data[i + 1]);
        view.setUint8(offset++, data[i]);
      }
      var pad = rowSize - w * 3;
      for (var p = 0; p < pad; p++) view.setUint8(offset++, 0);
    }
    return new Blob([buf], { type: "image/bmp" });
  }

  function loadGifenc() {
    if (!gifencPromise) {
      gifencPromise = import("https://cdn.jsdelivr.net/npm/gifenc@1.0.3/+esm");
    }
    return gifencPromise;
  }

  async function encodeGif(canvas) {
    var mod = await loadGifenc();
    var GIFEncoder = mod.GIFEncoder;
    var quantize = mod.quantize;
    var applyPalette = mod.applyPalette;
    var ctx = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    var imageData = ctx.getImageData(0, 0, w, h);
    var palette = quantize(imageData.data, 256);
    var index = applyPalette(imageData.data, palette);
    var gif = GIFEncoder();
    gif.writeFrame(index, w, h, { palette: palette, delay: 0 });
    gif.finish();
    return new Blob([gif.bytes()], { type: "image/gif" });
  }

  function loadJspdf() {
    if (!jspdfPromise) {
      jspdfPromise = import("https://cdn.jsdelivr.net/npm/jspdf@2.5.2/+esm");
    }
    return jspdfPromise;
  }

  async function mergeAllToPdf(fileList) {
    var mod = await loadJspdf();
    var jsPDF = mod.jsPDF;
    var pdf = null;
    var pageW = 595.28;
    var pageH = 841.89;
    for (var i = 0; i < fileList.length; i++) {
      var img = await loadImageFromFile(fileList[i]);
      var canvas = drawToCanvas(img, true);
      var dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      var iw = canvas.width;
      var ih = canvas.height;
      var ratio = Math.min(pageW / iw, pageH / ih);
      var nw = iw * ratio;
      var nh = ih * ratio;
      var x = (pageW - nw) / 2;
      var y = (pageH - nh) / 2;
      if (!pdf) {
        pdf = new jsPDF({ unit: "pt", format: "a4" });
      } else {
        pdf.addPage();
      }
      pdf.addImage(dataUrl, "JPEG", x, y, nw, nh);
    }
    var blob = pdf.output("blob");
    downloadBlob(blob, "imagens-ouviescrevi.pdf");
    return fileList.length;
  }

  function downloadBlob(blob, filename) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(url);
    }, 500);
  }

  function outputName(originalName, ext) {
    var base = (originalName || "imagem").replace(/\.[^.]+$/, "");
    return base + "-ouviescrevi." + ext;
  }

  function formatBytes(n) {
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    return (n / (1024 * 1024)).toFixed(2) + " MB";
  }

  function extFromMime(mime) {
    if (mime === "image/jpeg") return "jpg";
    if (mime === "image/webp") return "webp";
    if (mime === "image/png") return "png";
    return "jpg";
  }

  function getMaxWidth() {
    var sel = document.getElementById("cimgMaxWidth");
    if (!sel) return 1920;
    if (sel.value === "custom") {
      var n = parseInt((document.getElementById("cimgCustomWidth") || {}).value, 10);
      return Math.max(100, Math.min(8000, n || 1200));
    }
    return parseInt(sel.value, 10) || 1920;
  }

  function scaleToCanvas(img, maxWidth) {
    var w = img.naturalWidth || img.width;
    var h = img.naturalHeight || img.height;
    if (w <= maxWidth) return drawToCanvas(img, false);
    var nh = Math.round(h * (maxWidth / w));
    var canvas = document.createElement("canvas");
    canvas.width = maxWidth;
    canvas.height = nh;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, maxWidth, nh);
    return canvas;
  }

  async function blobFromCanvas(canvas, fmt, quality) {
    if (fmt.encoder === "bmp") return encodeBmp(canvas);
    if (fmt.encoder === "gif") return encodeGif(canvas);
    return canvasToBlob(canvas, fmt.id, quality);
  }

  async function processFile(file) {
    var img = await loadImageFromFile(file);
    var before = file.size;

    if (mode === "compress") {
      var mime = (document.getElementById("cimgCompressFormat") || {}).value || "image/webp";
      var cq = parseInt((document.getElementById("cimgCompressQuality") || {}).value, 10) || 75;
      var quality = Math.max(0.1, Math.min(1, cq / 100));
      var canvas = drawToCanvas(img, mime === "image/jpeg");
      var blob = await canvasToBlob(canvas, mime, quality);
      downloadBlob(blob, outputName(file.name, extFromMime(mime)));
      return { before: before, after: blob.size };
    }

    if (mode === "resize") {
      var maxW = getMaxWidth();
      var canvas = scaleToCanvas(img, maxW);
      var resizeFmt = (document.getElementById("cimgResizeFormat") || {}).value || "keep";
      var mime =
        resizeFmt === "keep"
          ? file.type && file.type.startsWith("image/") && file.type !== "image/svg+xml"
            ? file.type
            : "image/jpeg"
          : resizeFmt;
      if (mime === "image/jpeg") {
        var c2 = document.createElement("canvas");
        c2.width = canvas.width;
        c2.height = canvas.height;
        var ctx = c2.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, c2.width, c2.height);
        ctx.drawImage(canvas, 0, 0);
        canvas = c2;
      }
      var rq = mime === "image/png" ? undefined : 0.9;
      var blob = await canvasToBlob(canvas, mime, rq);
      downloadBlob(blob, outputName(file.name, extFromMime(mime)));
      return { before: before, after: blob.size };
    }

    var fmtId = (document.getElementById("cimgFormat") || {}).value || "image/png";
    var fmt = getFormatById(fmtId);
    var q = parseInt((document.getElementById("cimgQuality") || {}).value, 10) || 90;
    var quality = Math.max(0.1, Math.min(1, q / 100));
    var canvas = drawToCanvas(img, !!fmt.fillWhite);
    var blob = await blobFromCanvas(canvas, fmt, quality);
    downloadBlob(blob, outputName(file.name, fmt.ext));
    return { before: before, after: blob.size };
  }

  async function convertAll() {
    if (!files.length) {
      setStatus(t("needFile"), "err");
      return;
    }
    var btn = document.getElementById("btnCimgConvert");
    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("converting"));
    }
    setStatus(t("converting"));
    var ok = 0;
    var lastSizes = null;
    try {
      if (mode === "pdf") {
        ok = await mergeAllToPdf(files);
        setStatus(t("donePdf").replace("%n", String(ok)), "ok");
        return;
      }
      for (var i = 0; i < files.length; i++) {
        lastSizes = await processFile(files[i]);
        ok++;
        if (files.length > 1) {
          await new Promise(function (r) {
            setTimeout(r, 350);
          });
        }
      }
      var msg =
        ok === 1 ? t("done") : t("doneMany").replace("%n", String(ok));
      if (ok === 1 && lastSizes && mode !== "convert" && lastSizes.before > lastSizes.after) {
        var pct = Math.round((1 - lastSizes.after / lastSizes.before) * 100);
        msg = t("doneSize")
          .replace("%before", formatBytes(lastSizes.before))
          .replace("%after", formatBytes(lastSizes.after))
          .replace("%pct", String(pct));
      }
      setStatus(msg, "ok");
    } catch (e) {
      console.error(e);
      setStatus(e.message === "unsupported" ? t("unsupported") : t("error"), "err");
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  function setMode(next) {
    mode = next || "convert";
    document.querySelectorAll(".oe-cimg-mode").forEach(function (btn) {
      var active = btn.getAttribute("data-mode") === mode;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    var panels = {
      convert: document.getElementById("cimgPanelConvert"),
      compress: document.getElementById("cimgPanelCompress"),
      resize: document.getElementById("cimgPanelResize"),
      pdf: document.getElementById("cimgPanelPdf"),
    };
    Object.keys(panels).forEach(function (key) {
      if (panels[key]) panels[key].hidden = key !== mode;
    });
    var btn = document.getElementById("btnCimgConvert");
    if (btn) {
      if (mode === "compress") btn.textContent = t("btnCompress");
      else if (mode === "resize") btn.textContent = t("btnResize");
      else if (mode === "pdf") btn.textContent = t("btnPdf");
      else btn.textContent = t("btnConvert");
    }
  }

  function bindModeTabs() {
    document.querySelectorAll(".oe-cimg-mode").forEach(function (btn) {
      btn.addEventListener("click", function () {
        setMode(btn.getAttribute("data-mode"));
      });
    });
    var maxSel = document.getElementById("cimgMaxWidth");
    var customRow = document.getElementById("cimgCustomWidthRow");
    if (maxSel && customRow) {
      maxSel.addEventListener("change", function () {
        customRow.hidden = maxSel.value !== "custom";
      });
    }
    var cq = document.getElementById("cimgCompressQuality");
    var cqOut = document.getElementById("cimgCompressQualityOut");
    if (cq && cqOut) {
      cqOut.textContent = cq.value;
      cq.addEventListener("input", function () {
        cqOut.textContent = cq.value;
      });
    }
  }

  function clearFiles() {
    files = [];
    renderFileList();
    var box = document.getElementById("cimgPreview");
    var img = document.getElementById("cimgPreviewImg");
    if (img) img.removeAttribute("src");
    if (box) box.hidden = true;
    setStatus("");
    var input = document.getElementById("cimgInput");
    if (input) input.value = "";
  }

  function bindDropZone() {
    var zone = document.getElementById("cimgDrop");
    var input = document.getElementById("cimgInput");
    if (!zone || !input) return;

    zone.addEventListener("click", function () {
      input.click();
    });
    zone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        input.click();
      }
    });
    zone.addEventListener("dragover", function (e) {
      e.preventDefault();
      zone.classList.add("is-dragover");
    });
    zone.addEventListener("dragleave", function () {
      zone.classList.remove("is-dragover");
    });
    zone.addEventListener("drop", function (e) {
      e.preventDefault();
      zone.classList.remove("is-dragover");
      addFiles(e.dataTransfer.files);
    });
    input.addEventListener("change", function () {
      addFiles(input.files);
    });
  }

  function applyStrings() {
    var map = {
      cimgDropTitle: "dropTitle",
      cimgDropHint: "dropHint",
      cimgFormatLabel: "formatLabel",
      cimgQualityLabel: "qualityLabel",
      cimgPreviewTitle: "preview",
      cimgModeConvert: "modeConvert",
      cimgModeCompress: "modeCompress",
      cimgModeResize: "modeResize",
      cimgModePdf: "modePdf",
      cimgPdfHint: "pdfHint",
      cimgCompressFormatLabel: "compressFormatLabel",
      cimgCompressQualityLabel: "compressQualityLabel",
      cimgCompressHint: "compressHint",
      cimgMaxWidthLabel: "maxWidthLabel",
      cimgCustomWidthLabel: "customWidthLabel",
      cimgResizeFormatLabel: "resizeFormatLabel",
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });
    populateFormatSelect();
    setMode(mode);
    var clr = document.getElementById("btnCimgClear");
    if (clr) clr.textContent = t("btnClear");
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    bindDropZone();
    bindModeTabs();
    setMode("convert");

    var fmtSel = document.getElementById("cimgFormat");
    if (fmtSel) {
      fmtSel.addEventListener("change", updateQualityVisibility);
    }

    var btn = document.getElementById("btnCimgConvert");
    if (btn) {
      btn.disabled = true;
      btn.addEventListener("click", convertAll);
    }
    var clr = document.getElementById("btnCimgClear");
    if (clr) clr.addEventListener("click", clearFiles);
    var q = document.getElementById("cimgQuality");
    var out = document.getElementById("cimgQualityOut");
    if (q && out) {
      out.textContent = q.value;
      q.addEventListener("input", function () {
        out.textContent = q.value;
      });
    }

    probeAvif().then(function () {
      applyStrings();
    });
  }

  global.ConversorImagensUI = { init: init };
})(typeof window !== "undefined" ? window : this);
