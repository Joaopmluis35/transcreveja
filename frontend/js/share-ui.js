/**
 * Partilha pública de transcrição (/partilha.html?id=…).
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
    var out = document.getElementById("output");
    var text = (out && (out.innerText || out.textContent) || "").trim();
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

  function ensureButton() {
    if (document.getElementById("oeSharePublicBtn")) return;
    var host =
      document.getElementById("downloadButtons") ||
      document.querySelector(".oe-result-actions") ||
      document.getElementById("output");
    if (!host) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.id = "oeSharePublicBtn";
    btn.className = "oe-pro-btn oe-pro-btn--ghost";
    btn.textContent = "Partilhar link";
    btn.addEventListener("click", function () {
      shareFromOutput();
    });
    if (host.id === "output" && host.parentNode) {
      host.parentNode.insertBefore(btn, host.nextSibling);
    } else {
      host.appendChild(btn);
    }
  }

  function init() {
    ensureButton();
  }

  global.OuviescreviShare = { createShare: createShare, shareFromOutput: shareFromOutput, init: init };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
