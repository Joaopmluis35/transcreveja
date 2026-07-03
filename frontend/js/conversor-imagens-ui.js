/**
 * Conversor de imagens — PNG, JPEG, WebP no browser.
 */
(function (global) {
  "use strict";

  var config = { lang: "pt" };
  var files = [];

  var STRINGS = {
    pt: {
      dropTitle: "Arrasta imagens aqui",
      dropHint: "ou clica para escolher — JPG, PNG, WebP, GIF, BMP, SVG",
      formatLabel: "Converter para",
      qualityLabel: "Qualidade (JPEG / WebP)",
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
      png: "PNG (.png)",
      jpeg: "JPEG (.jpg)",
      webp: "WebP (.webp)",
    },
    en: {
      dropTitle: "Drop images here",
      dropHint: "or click to browse — JPG, PNG, WebP, GIF, BMP, SVG",
      formatLabel: "Convert to",
      qualityLabel: "Quality (JPEG / WebP)",
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
      png: "PNG (.png)",
      jpeg: "JPEG (.jpg)",
      webp: "WebP (.webp)",
    },
  };

  function t(key) {
    var loc = STRINGS[config.lang] || STRINGS.pt;
    return loc[key] || STRINGS.pt[key] || key;
  }

  function extForFormat(fmt) {
    if (fmt === "image/jpeg") return "jpg";
    if (fmt === "image/webp") return "webp";
    return "png";
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

  function outputName(originalName, mime) {
    var base = (originalName || "imagem").replace(/\.[^.]+$/, "");
    return base + "-ouviescrevi." + extForFormat(mime);
  }

  async function convertOne(file, mime, quality) {
    var img = await loadImageFromFile(file);
    var canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    var ctx = canvas.getContext("2d");
    if (mime === "image/jpeg") {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    ctx.drawImage(img, 0, 0);
    var blob = await canvasToBlob(canvas, mime, quality);
    downloadBlob(blob, outputName(file.name, mime));
  }

  async function convertAll() {
    if (!files.length) {
      setStatus(t("needFile"), "err");
      return;
    }
    var fmt = (document.getElementById("cimgFormat") || {}).value || "image/png";
    var q = parseInt((document.getElementById("cimgQuality") || {}).value, 10) || 90;
    var quality = Math.max(0.1, Math.min(1, q / 100));
    var btn = document.getElementById("btnCimgConvert");
    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("converting"));
    }
    setStatus(t("converting"));
    var ok = 0;
    try {
      for (var i = 0; i < files.length; i++) {
        await convertOne(files[i], fmt, quality);
        ok++;
        if (files.length > 1) {
          await new Promise(function (r) {
            setTimeout(r, 350);
          });
        }
      }
      setStatus(
        ok === 1 ? t("done") : t("doneMany").replace("%n", String(ok)),
        "ok"
      );
    } catch (e) {
      console.error(e);
      setStatus(e.message === "unsupported" ? t("unsupported") : t("error"), "err");
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
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
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.textContent = t(map[id]);
    });
    var fmt = document.getElementById("cimgFormat");
    if (fmt) {
      fmt.options[0].textContent = t("png");
      fmt.options[1].textContent = t("jpeg");
      fmt.options[2].textContent = t("webp");
    }
    var btn = document.getElementById("btnCimgConvert");
    if (btn) btn.textContent = t("btnConvert");
    var clr = document.getElementById("btnCimgClear");
    if (clr) clr.textContent = t("btnClear");
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyStrings();
    bindDropZone();
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
  }

  global.ConversorImagensUI = { init: init };
})(typeof window !== "undefined" ? window : this);
