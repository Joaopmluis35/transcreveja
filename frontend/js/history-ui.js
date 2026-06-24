/**
 * Histórico de transcrições e indicador de quota diária.
 */
(function (global) {
  function isSiteUser() {
    var role = sessionStorage.getItem("ouviescrevi_site_role");
    return role === "user" && sessionStorage.getItem("ouviescrevi_site_session");
  }

  function formatDate(iso) {
    if (!iso) return "—";
    return iso.replace("T", " ").slice(0, 16);
  }

  function formatQuota(q) {
    if (!q) return "";
    if (q.unlimited) return "";
    var rem = q.remaining;
    var lim = q.limit;
    if (lim <= 0) return "";
    if (q.tier === "registered") {
      return "Conta: " + rem + " de " + lim + " transcrições hoje";
    }
    return "Anónimo: " + rem + " de " + lim + " transcrições hoje — regista-te para mais";
  }

  async function fetchUsage() {
    if (!global.OuviescreviAPI) return null;
    await global.OuviescreviAPI.init();
    var res = await fetch(global.OuviescreviAPI.getBase() + "/api/usage", {
      headers: global.OuviescreviAPI.authHeaders(),
    });
    if (!res.ok) return null;
    return res.json();
  }

  function renderQuota(quota) {
    var el = document.getElementById("quotaBadge");
    if (!el) return;
    var text = formatQuota(quota);
    if (!text) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.textContent = text;
    el.classList.remove("hidden");
    if (quota.remaining === 0) el.classList.add("oe-quota-badge--warn");
    else el.classList.remove("oe-quota-badge--warn");
  }

  async function refreshQuota() {
    try {
      var quota = await fetchUsage();
      renderQuota(quota);
      return quota;
    } catch (e) {
      return null;
    }
  }

  async function loadHistory() {
    var panel = document.getElementById("historyPanel");
    var list = document.getElementById("historyList");
    var empty = document.getElementById("historyEmpty");
    if (!panel || !list) return;

    if (!isSiteUser()) {
      panel.classList.add("hidden");
      return;
    }

    panel.classList.remove("hidden");
    list.innerHTML = "<li class='oe-history-loading'>A carregar…</li>";

    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history?limit=40", {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error();
      var data = await res.json();
      var items = data.items || [];
      list.innerHTML = "";
      if (!items.length) {
        if (empty) empty.classList.remove("hidden");
        return;
      }
      if (empty) empty.classList.add("hidden");
      items.forEach(function (row) {
        var li = document.createElement("li");
        li.className = "oe-history-item";
        var name = row.filename || "Sem nome";
        var preview = (row.preview || "").trim();
        li.innerHTML =
          '<button type="button" class="oe-history-item__open">' +
          '<span class="oe-history-item__name">' + escapeHtml(name) + "</span>" +
          '<span class="oe-history-item__meta">' + formatDate(row.created_at) + "</span>" +
          (preview ? '<span class="oe-history-item__preview">' + escapeHtml(preview) + "…</span>" : "") +
          "</button>" +
          '<button type="button" class="oe-history-item__del" title="Apagar" aria-label="Apagar">✕</button>';
        li.querySelector(".oe-history-item__open").addEventListener("click", function () {
          openHistoryItem(row.id);
        });
        li.querySelector(".oe-history-item__del").addEventListener("click", function (e) {
          e.stopPropagation();
          deleteHistoryItem(row.id, li);
        });
        list.appendChild(li);
      });
    } catch (e) {
      list.innerHTML = "<li class='oe-history-empty'>Não foi possível carregar o histórico.</li>";
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  async function openHistoryItem(id) {
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history/" + id, {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error();
      var row = await res.json();
      var texto = row.formatted || row.transcription || "";
      if (!texto.trim()) return;
      if (typeof global.definirTranscricao === "function") {
        global.definirTranscricao(texto);
      } else {
        var out = document.getElementById("transcricao");
        if (out) out.textContent = texto;
      }
      var output = document.getElementById("output");
      if (output) {
        output.classList.remove("hidden");
        output.scrollIntoView({ behavior: "smooth", block: "start" });
      }
      if (global.OuviescreviUI && global.OuviescreviUI.toast) {
        global.OuviescreviUI.toast("Transcrição carregada do histórico.", "success");
      }
    } catch (e) {
      if (global.OuviescreviUI && global.OuviescreviUI.toast) {
        global.OuviescreviUI.toast("Erro ao abrir transcrição.", "error");
      }
    }
  }

  async function deleteHistoryItem(id, li) {
    if (!confirm("Apagar esta transcrição do histórico?")) return;
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history/" + id, {
        method: "DELETE",
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error();
      if (li && li.parentNode) li.parentNode.removeChild(li);
      var list = document.getElementById("historyList");
      var empty = document.getElementById("historyEmpty");
      if (list && !list.querySelector(".oe-history-item") && empty) {
        empty.classList.remove("hidden");
      }
    } catch (e) {
      if (global.OuviescreviUI && global.OuviescreviUI.toast) {
        global.OuviescreviUI.toast("Erro ao apagar.", "error");
      }
    }
  }

  function refresh() {
    refreshQuota();
    loadHistory();
  }

  function bind() {
    var btn = document.getElementById("btnRefreshHistory");
    if (btn) btn.addEventListener("click", loadHistory);
    document.addEventListener("oe-auth-change", refresh);
  }

  function init() {
    bind();
    refresh();
  }

  global.OuviescreviHistory = {
    init,
    refresh,
    refreshQuota,
    loadHistory,
  };
})(window);
