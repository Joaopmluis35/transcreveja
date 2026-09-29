/**
 * Histórico de transcrições: lista, pesquisa, renomear, partilhar e quota diária.
 */
(function (global) {
  var searchTimer = null;
  var lastQuery = "";

  function isSiteUser() {
    var role = (function () {
      try {
        return localStorage.getItem("ouviescrevi_site_role") || sessionStorage.getItem("ouviescrevi_site_role");
      } catch (e) {
        return sessionStorage.getItem("ouviescrevi_site_role");
      }
    })();
    var sess = (function () {
      try {
        return localStorage.getItem("ouviescrevi_site_session") || sessionStorage.getItem("ouviescrevi_site_session");
      } catch (e) {
        return sessionStorage.getItem("ouviescrevi_site_session");
      }
    })();
    return role === "user" && !!sess;
  }

  function formatDate(iso) {
    if (!iso) return "—";
    return iso.replace("T", " ").slice(0, 16);
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function toast(msg, kind) {
    if (global.OuviescreviUI && global.OuviescreviUI.toast) {
      global.OuviescreviUI.toast(msg, kind || "info");
    }
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
    if (!quota || quota.unlimited) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    var rem = quota.remaining;
    var lim = quota.limit;
    if (lim <= 0) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    if (quota.plan === "pro") {
      el.textContent = "Pro: " + rem + " de " + lim + " transcrições hoje";
    } else if (quota.tier === "registered") {
      el.textContent = "Conta: " + rem + " de " + lim + " transcrições hoje";
    } else {
      el.innerHTML =
        "Anónimo: " +
        rem +
        " de " +
        lim +
        ' transcrições hoje — <a href="#" class="oe-quota-badge__link" data-oe-register>Regista-te</a> para mais';
    }
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

  function currentSearchQuery() {
    var input = document.getElementById("historySearch");
    return input ? String(input.value || "").trim() : "";
  }

  function syncEmptyState(items, query) {
    var empty = document.getElementById("historyEmpty");
    if (!empty) return;
    if (items && items.length) {
      empty.classList.add("hidden");
      return;
    }
    empty.classList.remove("hidden");
    empty.textContent = query
      ? "Nenhuma transcrição corresponde a «" + query + "»."
      : "Ainda não tens transcrições guardadas nesta conta.";
  }

  function setHistoryCount(n) {
    var el = document.getElementById("historyCount");
    if (!el) return;
    if (!n) {
      el.hidden = true;
      el.textContent = "";
      return;
    }
    el.hidden = false;
    el.textContent = n === 1 ? "1 guardada" : n + " guardadas";
  }

  function setRowBusy(li, busy) {
    if (!li) return;
    li.classList.toggle("is-busy", !!busy);
    li.querySelectorAll("button").forEach(function (btn) {
      btn.disabled = !!busy;
    });
  }

  async function fetchHistoryRow(id) {
    await global.OuviescreviAPI.init();
    var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history/" + id, {
      headers: global.OuviescreviAPI.authHeaders(),
    });
    if (!res.ok) throw new Error();
    return res.json();
  }

  function historyText(row) {
    return (row.formatted || row.transcription || "").trim();
  }

  function renderHistoryItems(items) {
    var list = document.getElementById("historyList");
    if (!list) return;
    list.innerHTML = "";
    setHistoryCount(items.length);
    items.forEach(function (row) {
      var li = document.createElement("li");
      li.className = "oe-history-item";
      li.dataset.id = String(row.id);
      var name = row.filename || "Sem nome";
      var preview = (row.preview || "").replace(/\s+/g, " ").trim();
      li.innerHTML =
        '<button type="button" class="oe-history-item__open oe-history-tool">' +
        '<span class="oe-history-item__name">' +
        escapeHtml(name) +
        "</span>" +
        '<span class="oe-history-item__meta">' +
        formatDate(row.created_at) +
        "</span>" +
        (preview
          ? '<span class="oe-history-item__preview">' + escapeHtml(preview) + "</span>"
          : "") +
        "</button>" +
        '<div class="oe-history-item__actions">' +
        '<button type="button" class="oe-history-item__copy oe-history-tool">Copiar</button>' +
        '<button type="button" class="oe-history-item__txt oe-history-tool">TXT</button>' +
        '<button type="button" class="oe-history-item__share oe-history-tool">Partilhar</button>' +
        '<button type="button" class="oe-history-item__rename oe-history-tool">Renomear</button>' +
        '<button type="button" class="oe-history-item__del oe-history-tool">Apagar</button>' +
        "</div>";
      li.querySelector(".oe-history-item__open").addEventListener("click", function () {
        openHistoryItem(row.id, li);
      });
      li.querySelector(".oe-history-item__copy").addEventListener("click", function (e) {
        e.stopPropagation();
        copyHistoryItem(row.id, li);
      });
      li.querySelector(".oe-history-item__txt").addEventListener("click", function (e) {
        e.stopPropagation();
        downloadHistoryItem(row.id, name, li);
      });
      li.querySelector(".oe-history-item__share").addEventListener("click", function (e) {
        e.stopPropagation();
        shareHistoryItem(row.id, name, li);
      });
      li.querySelector(".oe-history-item__rename").addEventListener("click", function (e) {
        e.stopPropagation();
        renameHistoryItem(row.id, name, li);
      });
      li.querySelector(".oe-history-item__del").addEventListener("click", function (e) {
        e.stopPropagation();
        deleteHistoryItem(row.id, li);
      });
      list.appendChild(li);
    });
  }

  async function loadHistory(query) {
    var panel = document.getElementById("historyPanel");
    var list = document.getElementById("historyList");
    var empty = document.getElementById("historyEmpty");
    if (!panel || !list) return;

    if (!isSiteUser()) {
      panel.classList.add("hidden");
      return;
    }

    panel.classList.remove("hidden");
    var q = typeof query === "string" ? query.trim() : currentSearchQuery();
    lastQuery = q;
    list.innerHTML = "<li class='oe-history-loading'><span class='oe-history-loading__spinner' aria-hidden='true'></span><span>A carregar…</span></li>";
    if (empty) empty.classList.add("hidden");

    try {
      await global.OuviescreviAPI.init();
      var histLimit = 100;
      try {
        var meRes = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/me", {
          headers: global.OuviescreviAPI.authHeaders(),
        });
        if (meRes.ok) {
          var me = await meRes.json();
          if (me && (me.plan === "pro" || me.is_pro)) histLimit = 200;
        }
      } catch (eMe) {}
      var url =
        global.OuviescreviAPI.getBase() +
        "/api/auth/history?limit=" +
        histLimit +
        (q ? "&q=" + encodeURIComponent(q) : "");
      var res = await fetch(url, {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error();
      var data = await res.json();
      var items = data.items || [];
      if (!items.length) {
        list.innerHTML = "";
        setHistoryCount(0);
        syncEmptyState(items, q);
        return;
      }
      syncEmptyState(items, q);
      renderHistoryItems(items);
    } catch (e) {
      list.innerHTML = "<li class='oe-history-empty'>Não foi possível carregar o histórico.</li>";
    }
  }

  async function openHistoryItem(id, li) {
    setRowBusy(li, true);
    try {
      var row = await fetchHistoryRow(id);
      var texto = historyText(row);
      if (!texto) return;
      if (typeof global.definirTranscricao === "function") {
        global.definirTranscricao(texto);
      } else {
        var out = document.getElementById("transcricao");
        if (out) out.textContent = texto;
        var ta = document.getElementById("transcriptionText");
        if (ta) ta.value = texto;
      }
      var output = document.getElementById("output");
      if (output) {
        output.classList.remove("hidden");
        if (typeof global.focusTranscriptionResult === "function") {
          global.focusTranscriptionResult();
        } else {
          output.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
      if (global.OuviescreviShare && global.OuviescreviShare.syncVisibility) {
        global.OuviescreviShare.syncVisibility();
      }
      toast("Transcrição carregada do histórico.", "success");
    } catch (e) {
      toast("Erro ao abrir transcrição.", "error");
    } finally {
      setRowBusy(li, false);
    }
  }

  async function copyHistoryItem(id, li) {
    setRowBusy(li, true);
    try {
      var row = await fetchHistoryRow(id);
      var text = historyText(row);
      if (!text) {
        toast("Esta transcrição não tem texto.", "error");
        return;
      }
      if (!navigator.clipboard || !navigator.clipboard.writeText) {
        toast("Não foi possível copiar neste browser.", "error");
        return;
      }
      await navigator.clipboard.writeText(text);
      toast("Texto copiado.", "success");
    } catch (e) {
      toast("Erro ao copiar.", "error");
    } finally {
      setRowBusy(li, false);
    }
  }

  async function downloadHistoryItem(id, fallbackName, li) {
    setRowBusy(li, true);
    try {
      var row = await fetchHistoryRow(id);
      var text = historyText(row);
      if (!text) {
        toast("Esta transcrição não tem texto.", "error");
        return;
      }
      var base = String(row.filename || fallbackName || "transcricao").replace(/\.[^.]+$/, "");
      var blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = base + ".txt";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(a.href);
    } catch (e) {
      toast("Erro ao descarregar.", "error");
    } finally {
      setRowBusy(li, false);
    }
  }

  async function shareHistoryItem(id, fallbackTitle) {
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history/" + id, {
        headers: global.OuviescreviAPI.authHeaders(),
      });
      if (!res.ok) throw new Error();
      var row = await res.json();
      var text = (row.formatted || row.transcription || "").trim();
      if (text.length < 20) {
        toast("Esta transcrição não tem texto suficiente para partilhar.", "error");
        return;
      }
      if (!global.OuviescreviShare || !global.OuviescreviShare.createShare) {
        toast("Partilha indisponível neste momento.", "error");
        return;
      }
      var title = (row.filename || fallbackTitle || "Transcrição").trim();
      var data = await global.OuviescreviShare.createShare(text, title);
      var url = data && data.url;
      if (url && navigator.clipboard && navigator.clipboard.writeText) {
        try {
          await navigator.clipboard.writeText(url);
        } catch (eClip) {}
      }
      toast(url ? "Link de partilha copiado." : "Link de partilha criado.", "success");
      if (url && navigator.share) {
        try {
          await navigator.share({ title: title, url: url });
        } catch (eShare) {}
      }
    } catch (e) {
      toast((e && e.message) || "Erro ao partilhar.", "error");
    }
  }

  async function renameHistoryItem(id, currentName, li) {
    var next = window.prompt("Novo nome da transcrição:", currentName || "");
    if (next === null) return;
    next = String(next).trim();
    if (!next) {
      toast("O nome não pode ficar vazio.", "error");
      return;
    }
    if (next === currentName) return;
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/auth/history/" + id, {
        method: "PATCH",
        headers: global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ filename: next }),
      });
      var data = await res.json().catch(function () {
        return {};
      });
      if (!res.ok) throw new Error(data.detail || "Erro ao renomear.");
      var nameEl = li && li.querySelector(".oe-history-item__name");
      if (nameEl) nameEl.textContent = data.filename || next;
      toast("Nome atualizado.", "success");
    } catch (e) {
      toast((e && e.message) || "Erro ao renomear.", "error");
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
      if (list && !list.querySelector(".oe-history-item")) {
        setHistoryCount(0);
        syncEmptyState([], lastQuery);
      } else if (list) {
        setHistoryCount(list.querySelectorAll(".oe-history-item").length);
      }
      toast("Transcrição apagada.", "success");
    } catch (e) {
      toast("Erro ao apagar.", "error");
    }
  }

  function scheduleSearch() {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(function () {
      loadHistory(currentSearchQuery());
    }, 280);
  }

  function refresh() {
    refreshQuota();
    loadHistory(currentSearchQuery());
  }

  function bind() {
    var btn = document.getElementById("btnRefreshHistory");
    if (btn) btn.addEventListener("click", function () {
      loadHistory(currentSearchQuery());
    });
    var search = document.getElementById("historySearch");
    if (search) {
      search.addEventListener("input", scheduleSearch);
      search.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          e.preventDefault();
          if (searchTimer) clearTimeout(searchTimer);
          loadHistory(currentSearchQuery());
        }
      });
    }
    var clearBtn = document.getElementById("btnClearHistorySearch");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        var input = document.getElementById("historySearch");
        if (input) input.value = "";
        loadHistory("");
      });
    }
    var quotaEl = document.getElementById("quotaBadge");
    if (quotaEl) {
      quotaEl.addEventListener("click", function (e) {
        var link = e.target.closest("[data-oe-register]");
        if (!link) return;
        e.preventDefault();
        if (global.OuviescreviAuth && global.OuviescreviAuth.openModal) {
          global.OuviescreviAuth.openModal("register");
        }
      });
    }
    document.addEventListener("oe-auth-change", refresh);
  }

  function init() {
    bind();
    refresh();
  }

  global.OuviescreviHistory = {
    init: init,
    refresh: refresh,
    refreshQuota: refreshQuota,
    loadHistory: loadHistory,
  };
})(window);
