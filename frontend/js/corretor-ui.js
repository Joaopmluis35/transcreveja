/**
 * Corretor ortográfico — formulário, progresso e painel de resultado.
 */
(function (global) {
  var config = { lang: "pt" };
  var lastOriginal = "";
  var lastCorrected = "";
  var compareVisible = false;

  var STRINGS = {
    pt: {
      eyebrow: "Ortografia · Gramática · Estilo",
      formTitle: "O teu texto",
      formHint: "Cola ou escreve o texto que queres rever. A IA corrige erros e melhora a clareza.",
      placeholder: "Escreve ou cola aqui o teu texto...",
      modeLabel: "Tipo de correção",
      modeNormal: "Completa (ortografia + gramática)",
      modeSpelling: "Só ortografia e pontuação",
      modeFormal: "Tom mais formal",
      modeSimple: "Linguagem mais simples",
      btnCorrect: "Corrigir texto",
      loading: "A corrigir...",
      needText: "Introduz texto para corrigir.",
      chars: "%n caracteres",
      words: "%n palavras",
      resultTitle: "Texto corrigido",
      resultSubtitle: "Revê o resultado e escolhe o que fazer a seguir",
      errorTitle: "Não foi possível corrigir",
      copy: "Copiar",
      apply: "Aplicar ao texto",
      compare: "Comparar",
      hideCompare: "Ocultar comparação",
      download: "Descarregar TXT",
      recorrect: "Corrigir outra vez",
      copied: "Copiado!",
      applied: "Texto atualizado!",
      copyFail: "Não foi possível copiar.",
      serverError: "Erro ao contactar o servidor.",
      unexpected: "Ocorreu um erro inesperado.",
      compareOriginal: "Original",
      compareFixed: "Corrigido",
      progressHint: "Textos longos podem demorar alguns segundos.",
      phrases: [
        "A ler o texto com atenção...",
        "A identificar erros ortográficos...",
        "A rever gramática e pontuação...",
        "A aplicar correções inteligentes...",
        "A preparar o texto final...",
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

  function countWords(text) {
    var trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }

  function applyFormLabels() {
    var map = {
      corFormEyebrow: "eyebrow",
      corFormTitle: "formTitle",
      corFormHint: "formHint",
      corModoLabel: "modeLabel",
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
    var text = input.value;
    meta.textContent = fmt("chars", text.length) + " · " + fmt("words", countWords(text));
  }

  function hideOutput(out) {
    if (!out) return;
    out.hidden = true;
    out.innerHTML = "";
    out.classList.remove("oe-cor-output--error");
    compareVisible = false;
  }

  function bindOutputActions(out) {
    var copyBtn = out.querySelector("[data-cor-copy]");
    var applyBtn = out.querySelector("[data-cor-apply]");
    var compareBtn = out.querySelector("[data-cor-compare]");
    var downloadBtn = out.querySelector("[data-cor-download]");
    var recorrectBtn = out.querySelector("[data-cor-recorrect]");

    if (copyBtn) {
      copyBtn.addEventListener("click", function () {
        copyText(lastCorrected, copyBtn);
      });
    }
    if (applyBtn) {
      applyBtn.addEventListener("click", function () {
        var input = document.getElementById("textoInput");
        if (input && lastCorrected) {
          input.value = lastCorrected;
          updateMeta();
          if (global.OuviescreviUI) global.OuviescreviUI.toast(t("applied"), "success");
          input.scrollIntoView({ behavior: "smooth", block: "start" });
          input.focus();
        }
      });
    }
    if (compareBtn) {
      compareBtn.addEventListener("click", function () {
        toggleCompare(out, compareBtn);
      });
    }
    if (downloadBtn) {
      downloadBtn.addEventListener("click", downloadTxt);
    }
    if (recorrectBtn) {
      recorrectBtn.addEventListener("click", function () {
        var input = document.getElementById("textoInput");
        if (input) {
          input.scrollIntoView({ behavior: "smooth", block: "start" });
          input.focus();
        }
      });
    }
  }

  function toggleCompare(out, btn) {
    var panel = out.querySelector("[data-cor-compare-panel]");
    if (!panel) return;
    compareVisible = !compareVisible;
    panel.hidden = !compareVisible;
    btn.textContent = compareVisible ? t("hideCompare") : t("compare");
  }

  function copyText(text, btn) {
    if (!text) return;
    var done = function () {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("copied"), "success");
      if (btn) {
        var prev = btn.textContent;
        btn.textContent = t("copied");
        setTimeout(function () {
          btn.textContent = prev;
        }, 1800);
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
    var blob = new Blob([lastCorrected], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "texto-corrigido-ouviescrevi.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  function showSuccess(out, original, corrected) {
    lastOriginal = original;
    lastCorrected = corrected;
    compareVisible = false;
    out.hidden = false;
    out.classList.remove("oe-cor-output--error");
    out.innerHTML =
      '<div class="oe-cor-output__head">' +
      '<div class="oe-cor-output__title-wrap">' +
      '<h2 class="oe-cor-output__title">' + escapeHtml(t("resultTitle")) + "</h2>" +
      '<p class="oe-cor-output__subtitle">' + escapeHtml(t("resultSubtitle")) + "</p>" +
      "</div>" +
      '<div class="oe-cor-output__actions">' +
      '<button type="button" class="oe-cor-output__btn oe-cor-output__btn--primary" data-cor-apply>' + escapeHtml(t("apply")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn" data-cor-copy>' + escapeHtml(t("copy")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn" data-cor-compare>' + escapeHtml(t("compare")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn" data-cor-download>' + escapeHtml(t("download")) + "</button>" +
      '<button type="button" class="oe-cor-output__btn" data-cor-recorrect>' + escapeHtml(t("recorrect")) + "</button>" +
      "</div></div>" +
      '<pre class="oe-cor-output__body">' + escapeHtml(corrected) + "</pre>" +
      '<div class="oe-cor-compare" data-cor-compare-panel hidden>' +
      '<div class="oe-cor-compare__col oe-cor-compare__col--original">' +
      '<p class="oe-cor-compare__label">' + escapeHtml(t("compareOriginal")) + "</p>" +
      '<pre class="oe-cor-compare__text">' + escapeHtml(original) + "</pre></div>" +
      '<div class="oe-cor-compare__col oe-cor-compare__col--fixed">' +
      '<p class="oe-cor-compare__label">' + escapeHtml(t("compareFixed")) + "</p>" +
      '<pre class="oe-cor-compare__text">' + escapeHtml(corrected) + "</pre></div></div>";

    bindOutputActions(out);
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(out, message) {
    lastOriginal = "";
    lastCorrected = "";
    out.hidden = false;
    out.classList.add("oe-cor-output--error");
    out.innerHTML =
      '<div class="oe-cor-output__head">' +
      '<h2 class="oe-cor-output__title">' + escapeHtml(t("errorTitle")) + "</h2>" +
      "</div>" +
      '<pre class="oe-cor-output__body">' + escapeHtml(message) + "</pre>";
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function correctFromPage() {
    var input = document.getElementById("textoInput");
    var btn = document.getElementById("btnCorrigir");
    var out = document.getElementById("resultado");
    var progress = document.getElementById("corProgress");
    var progressBar = document.getElementById("progressBar");
    var loadingPhrase = document.getElementById("loadingPhrase");
    var modo = document.getElementById("corModo");

    if (!input || !btn || !out) return;

    var texto = input.value.trim();
    if (!texto) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(t("needText"), "error");
      return;
    }

    hideOutput(out);
    if (global.OuviescreviUI) {
      global.OuviescreviUI.setButtonLoading(btn, true, t("loading"));
    }
    if (progress) progress.hidden = false;
    if (progressBar) progressBar.style.width = "0%";
    if (loadingPhrase) loadingPhrase.textContent = "";

    var phrases = t("phrases");
    var percent = 0;
    var fraseIndex = 0;
    var interval = setInterval(function () {
      if (percent >= 92) return;
      percent += Math.floor(Math.random() * 4) + 1;
      if (progressBar) progressBar.style.width = percent + "%";
      if (loadingPhrase && phrases && phrases.length) {
        loadingPhrase.textContent = phrases[fraseIndex];
        fraseIndex = (fraseIndex + 1) % phrases.length;
      }
    }, 650);

    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/correct", {
        method: "POST",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI.authJson({
            text: texto,
            mode: modo ? modo.value : "normal",
          })
        ),
      });
      var data = await res.json().catch(function () {
        return {};
      });
      clearInterval(interval);
      if (progressBar) progressBar.style.width = "100%";
      if (loadingPhrase) loadingPhrase.textContent = "";

      if (res.ok && data.corrected) {
        showSuccess(out, texto, data.corrected);
      } else {
        var detail = data.detail || data.error || t("unexpected");
        if (Array.isArray(detail)) detail = detail.map(function (d) { return d.msg || d; }).join(" ");
        showError(out, String(detail));
      }
    } catch (err) {
      clearInterval(interval);
      if (progressBar) progressBar.style.width = "100%";
      if (loadingPhrase) loadingPhrase.textContent = "";
      console.error(err);
      showError(out, t("serverError"));
    } finally {
      if (global.OuviescreviUI) global.OuviescreviUI.setButtonLoading(btn, false);
      setTimeout(function () {
        if (progress) progress.hidden = true;
      }, 600);
    }
  }

  function init(opts) {
    config = Object.assign({}, config, opts || {});
    applyFormLabels();
    updateMeta();

    var input = document.getElementById("textoInput");
    if (input) {
      input.addEventListener("input", updateMeta);
    }

    var btn = document.getElementById("btnCorrigir");
    if (btn) btn.addEventListener("click", correctFromPage);

    global.corrigirTexto = correctFromPage;
  }

  global.CorretorUI = { init: init, correct: correctFromPage };
})(window);
