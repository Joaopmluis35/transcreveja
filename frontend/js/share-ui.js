/**
 * Partilha pública de transcrição (/s/{id} ou /partilha.html?id=…).
 * O botão vive só dentro de #output — nunca como irmão na coluna (grid empilha e sobrepõe o loading).
 */
(function (global) {
  function locale() {
    if (global.OuviescreviI18n) return global.OuviescreviI18n.localeFromPath();
    var m = (global.location && global.location.pathname || "").match(/^\/(en|es|fr|de)(\/|$)/);
    return m ? m[1] : "pt";
  }

  async function createShare(text, title) {
    await global.OuviescreviAPI.init();
    var res = await fetch(global.OuviescreviAPI.getBase() + "/api/share/transcript", {
      method: "POST",
      headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({
        text: text,
        title: title || "Transcrição",
        locale: locale(),
      }),
    });
    var data = await res.json().catch(function () { return {}; });
    if (!res.ok) throw new Error(data.detail || "Não foi possível partilhar.");
    return data;
  }

  async function shareFromOutput() {
    var out = document.getElementById("transcriptionText") || document.getElementById("output");
    var text = "";
    if (out) {
      text = (out.value || out.innerText || out.textContent || "").trim();
    }
    if (text.length < 20) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast("Ainda não há texto para partilhar.", "error");
      return;
    }
    try {
      var data = await createShare(text);
      var url = data.url;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(url);
      }
      if (global.OuviescreviUI) {
        global.OuviescreviUI.toast("Link de partilha copiado.", "success");
      }
      if (navigator.share) {
        try {
          await navigator.share({ title: "Transcrição Ouviescrevi", url: url });
        } catch (e) {}
      }
      return url;
    } catch (e) {
      if (global.OuviescreviUI) global.OuviescreviUI.toast(e.message || "Erro ao partilhar.", "error");
    }
  }

  function removeOrphanButtons() {
    var output = document.getElementById("output");
    document.querySelectorAll("#oeSharePublicBtn").forEach(function (btn) {
      if (!output || !output.contains(btn)) {
        btn.remove();
      }
    });
  }

  function ensureButton() {
    removeOrphanButtons();
    var existing = document.getElementById("oeSharePublicBtn");
    if (existing) return existing;

    var output = document.getElementById("output");
    if (!output) return null;

    var host =
      output.querySelector(".oe-output__actions.btn-group") ||
      output.querySelector(".oe-output__toolbar-actions") ||
      output.querySelector("#downloadButtons") ||
      output.querySelector(".oe-result-actions");
    if (!host) return null;

    var btn = document.createElement("button");
    btn.type = "button";
    btn.id = "oeSharePublicBtn";
    btn.className = "oe-pro-btn oe-pro-btn--ghost oe-share-public-btn hidden";
    btn.textContent = "🔗 Partilhar link";
    btn.setAttribute("aria-hidden", "true");
    btn.addEventListener("click", function () {
      shareFromOutput();
    });
    host.appendChild(btn);
    return btn;
  }

  function syncVisibility() {
    var btn = ensureButton();
    if (!btn) return;
    var output = document.getElementById("output");
    var loading = document.getElementById("loading");
    var textEl = document.getElementById("transcriptionText");
    var hasText = !!(textEl && (textEl.value || "").trim().length >= 20);
    var outputVisible = !!(output && !output.classList.contains("hidden"));
    var loadingActive = !!(loading && !loading.classList.contains("hidden"));
    var show = outputVisible && hasText && !loadingActive;
    btn.classList.toggle("hidden", !show);
    btn.setAttribute("aria-hidden", show ? "false" : "true");
  }

  function init() {
    ensureButton();
    syncVisibility();
    var output = document.getElementById("output");
    if (output && typeof MutationObserver !== "undefined") {
      var mo = new MutationObserver(syncVisibility);
      mo.observe(output, { attributes: true, attributeFilter: ["class"] });
    }
    var loading = document.getElementById("loading");
    if (loading && typeof MutationObserver !== "undefined") {
      var moLoad = new MutationObserver(syncVisibility);
      moLoad.observe(loading, { attributes: true, attributeFilter: ["class"] });
    }
    var textEl = document.getElementById("transcriptionText");
    if (textEl) {
      textEl.addEventListener("input", syncVisibility);
    }
    document.addEventListener("oe-auth-change", syncVisibility);
    setInterval(syncVisibility, 2000);
  }

  global.OuviescreviShare = {
    createShare: createShare,
    shareFromOutput: shareFromOutput,
    init: init,
    syncVisibility: syncVisibility,
  };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
