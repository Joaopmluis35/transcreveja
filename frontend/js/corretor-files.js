/**
 * Corretor — extração de texto de PDF e DOCX (libs carregadas sob demanda).
 */
(function (global) {
  var pdfJsPromise = null;
  var mammothPromise = null;

  function toast(msg, type) {
    if (global.OuviescreviUI) global.OuviescreviUI.toast(msg, type || "error");
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var existing = document.querySelector('script[src="' + src + '"]');
      if (existing) {
        if (existing.dataset.loaded === "true") resolve();
        else existing.addEventListener("load", resolve, { once: true });
        return;
      }
      var s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = function () {
        s.dataset.loaded = "true";
        resolve();
      };
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  function ensurePdfJs() {
    if (global.pdfjsLib) return Promise.resolve();
    if (!pdfJsPromise) {
      pdfJsPromise = loadScript(
        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.min.js"
      );
    }
    return pdfJsPromise;
  }

  function ensureMammoth() {
    if (global.mammoth) return Promise.resolve();
    if (!mammothPromise) {
      mammothPromise = loadScript("https://unpkg.com/mammoth/mammoth.browser.min.js");
    }
    return mammothPromise;
  }

  function extractPdf(file, onDone, onFail) {
    ensurePdfJs()
      .then(function () {
        if (!global.pdfjsLib) throw new Error("pdfjs missing");
        var reader = new FileReader();
        reader.onload = function () {
          var typedarray = new Uint8Array(reader.result);
          global.pdfjsLib
            .getDocument({ data: typedarray })
            .promise.then(function (pdf) {
              var textoFinal = "";
              var total = pdf.numPages;
              var processadas = 0;
              if (total === 0) {
                if (onFail) onFail("empty");
                return;
              }
              for (var i = 1; i <= total; i++) {
                (function (pageNum) {
                  pdf.getPage(pageNum).then(function (page) {
                    page.getTextContent().then(function (content) {
                      textoFinal +=
                        content.items.map(function (item) { return item.str; }).join(" ") + "\n\n";
                      processadas++;
                      if (processadas === total && onDone) onDone(textoFinal.trim());
                    });
                  });
                })(i);
              }
            })
            .catch(function () {
              if (onFail) onFail("pdf");
            });
        };
        reader.onerror = function () {
          if (onFail) onFail("pdf");
        };
        reader.readAsArrayBuffer(file);
      })
      .catch(function () {
        if (onFail) onFail("pdf");
      });
  }

  function extractDocx(file, onDone, onFail) {
    ensureMammoth()
      .then(function () {
        if (!global.mammoth) throw new Error("mammoth missing");
        var reader = new FileReader();
        reader.onload = function () {
          global.mammoth
            .extractRawText({ arrayBuffer: reader.result })
            .then(function (result) {
              if (onDone) onDone((result.value || "").trim());
            })
            .catch(function () {
              if (onFail) onFail("docx");
            });
        };
        reader.onerror = function () {
          if (onFail) onFail("docx");
        };
        reader.readAsArrayBuffer(file);
      })
      .catch(function () {
        if (onFail) onFail("docx");
      });
  }

  function handleFile(file, strings, onText) {
    if (!file) return;
    var name = (file.name || "").toLowerCase();
    if (name.endsWith(".pdf")) {
      extractPdf(file, onText, function () { toast(strings.filePdfFail); });
    } else if (name.endsWith(".docx")) {
      extractDocx(file, onText, function () { toast(strings.fileDocxFail); });
    } else {
      toast(strings.fileUnsupported);
    }
  }

  function setup(opts) {
    var dropZone = document.getElementById(opts.dropZoneId || "corDropZone");
    var fileInput = document.getElementById(opts.fileInputId || "corFileInput");
    var textarea = document.getElementById(opts.textareaId || "textoInput");
    var strings = opts.strings || {};
    if (!dropZone || !fileInput || !textarea) return;

    function applyText(text) {
      textarea.value = text;
      if (opts.onText) opts.onText(text);
    }

    dropZone.addEventListener("click", function () {
      fileInput.click();
    });

    dropZone.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    dropZone.addEventListener("dragover", function (e) {
      e.preventDefault();
      dropZone.classList.add("oe-cor-drop--over");
    });

    dropZone.addEventListener("dragleave", function () {
      dropZone.classList.remove("oe-cor-drop--over");
    });

    dropZone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropZone.classList.remove("oe-cor-drop--over");
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0], strings, applyText);
      }
    });

    fileInput.addEventListener("change", function () {
      if (fileInput.files && fileInput.files[0]) {
        handleFile(fileInput.files[0], strings, applyText);
      }
    });
  }

  global.CorretorFiles = { setup: setup, extractPdf: extractPdf, extractDocx: extractDocx };
})(window);
