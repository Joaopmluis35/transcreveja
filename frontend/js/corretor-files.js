/**
 * Corretor — extração de texto de PDF e DOCX.
 */
(function (global) {
  function toast(msg, type) {
    if (global.OuviescreviUI) global.OuviescreviUI.toast(msg, type || "error");
  }

  function extractPdf(file, onDone, onFail) {
    if (!global.pdfjsLib) {
      if (onFail) onFail("pdf");
      return;
    }
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
                  textoFinal += content.items.map(function (item) { return item.str; }).join(" ") + "\n\n";
                  processadas++;
                  if (processadas === total && onDone) {
                    onDone(textoFinal.trim());
                  }
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
  }

  function extractDocx(file, onDone, onFail) {
    if (!global.mammoth) {
      if (onFail) onFail("docx");
      return;
    }
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
  }

  function handleFile(file, strings, onText) {
    if (!file) return;
    var name = (file.name || "").toLowerCase();
    if (name.endsWith(".pdf")) {
      extractPdf(
        file,
        onText,
        function () { toast(strings.filePdfFail); }
      );
    } else if (name.endsWith(".docx")) {
      extractDocx(
        file,
        onText,
        function () { toast(strings.fileDocxFail); }
      );
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
