/**
 * Backoffice Ouviescrevi — painel admin com gráficos (Chart.js)
 */
(function (global) {
  let chartVisitas = null;
  let chartTranscricoes = null;
  let cmsPages = [];
  let cmsContent = {};
  let cmsCurrentPage = null;
  let cmsEditors = {};
  const QUILL_TOOLBAR = [
    [{ header: [2, 3, false] }],
    ["bold", "italic", "underline"],
    [{ list: "ordered" }, { list: "bullet" }],
    ["link"],
    ["clean"],
  ];

  function apiBase() {
    return global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase();
  }

  function formatDay(iso) {
    if (!iso) return "";
    var p = iso.split("-");
    return p[2] + "/" + p[1];
  }

  function buildTable(headers, rows) {
    var table = document.createElement("table");
    table.className = "oe-admin-table";
    var thead = document.createElement("thead");
    var hr = document.createElement("tr");
    headers.forEach(function (t) {
      var th = document.createElement("th");
      th.textContent = t;
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    rows.forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (cell, i) {
        var td = document.createElement("td");
        if (i === 0 && String(cell).indexOf("/") === 0) {
          var code = document.createElement("code");
          code.textContent = cell;
          td.appendChild(code);
        } else {
          td.textContent = cell;
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function setPageTitle(title) {
    var el = document.getElementById("adminPageTitle");
    if (el) el.textContent = title;
  }

  function switchTab(tab) {
    document.querySelectorAll(".oe-admin-nav button").forEach(function (btn) {
      var active = btn.dataset.tab === tab;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll("[data-panel]").forEach(function (panel) {
      panel.classList.toggle("hidden", panel.dataset.panel !== tab);
    });
    var titles = { dashboard: "Painel", conteudo: "Conteúdo do site", transcricoes: "Transcrições" };
    if (tab !== "conteudo") {
      setPageTitle(titles[tab] || "Backoffice");
    } else if (cmsCurrentPage) {
      setPageTitle("Conteúdo — " + cmsCurrentPage.label);
    } else {
      setPageTitle(titles.conteudo);
    }
    document.getElementById("adminSidebar").classList.remove("is-open");
    if (tab === "transcricoes") carregarLogs();
  }

  function atualizarStatusManutencao(on) {
    var pill = document.getElementById("statusPill");
    if (!pill) return;
    pill.textContent = on ? "Em manutenção" : "Operacional";
    pill.className = "oe-admin-status-pill " + (on ? "oe-admin-status-pill--warn" : "oe-admin-status-pill--ok");
  }

  function renderCharts(charts) {
    if (!global.Chart) return;
    var visitas = (charts && charts.visitas_diarias) || [];
    var trans = (charts && charts.transcricoes_diarias) || [];
    var labels = visitas.map(function (d) { return formatDay(d.day); });

    var ctxV = document.getElementById("chartVisitas");
    if (ctxV) {
      if (chartVisitas) chartVisitas.destroy();
      chartVisitas = new Chart(ctxV, {
        type: "line",
        data: {
          labels: labels,
          datasets: [
            {
              label: "Visitas",
              data: visitas.map(function (d) { return d.total; }),
              borderColor: "#7c3aed",
              backgroundColor: "rgba(124, 58, 237, 0.1)",
              fill: true,
              tension: 0.35,
              pointRadius: 3,
            },
            {
              label: "Visitantes únicos",
              data: visitas.map(function (d) { return d.unicos || 0; }),
              borderColor: "#2563eb",
              backgroundColor: "transparent",
              borderDash: [4, 4],
              tension: 0.35,
              pointRadius: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "bottom" } },
          scales: {
            y: { beginAtZero: true, ticks: { precision: 0 } },
          },
        },
      });
    }

    var ctxT = document.getElementById("chartTranscricoes");
    if (ctxT) {
      if (chartTranscricoes) chartTranscricoes.destroy();
      chartTranscricoes = new Chart(ctxT, {
        type: "bar",
        data: {
          labels: trans.map(function (d) { return formatDay(d.day); }),
          datasets: [
            {
              label: "Transcrições",
              data: trans.map(function (d) { return d.total; }),
              backgroundColor: "rgba(5, 150, 105, 0.75)",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, ticks: { precision: 0 } },
          },
        },
      });
    }
  }

  function renderTopPages(pages) {
    var div = document.getElementById("topPaginas");
    if (!div) return;
    if (!pages || !pages.length) {
      div.innerHTML = '<p class="oe-admin-empty">Sem dados de páginas (últimos 30 dias).</p>';
      return;
    }
    div.innerHTML = "";
    div.appendChild(
      buildTable(
        ["Página", "Visitas"],
        pages.map(function (p) { return [p.path || "—", String(p.total)]; })
      )
    );
  }

  function renderVisitasRecentes(rows) {
    var div = document.getElementById("visitasRecentes");
    if (!div) return;
    if (!rows.length) {
      div.innerHTML = '<p class="oe-admin-empty">Ainda sem visitas registadas.</p>';
      return;
    }
    div.innerHTML = "";
    div.appendChild(
      buildTable(
        ["Página", "Dia", "Hora"],
        rows.map(function (r) {
          return [
            r.path || "—",
            r.day || "—",
            (r.created_at || "").replace("T", " ").replace("Z", ""),
          ];
        })
      )
    );
  }

  async function carregarDashboard() {
    try {
      var res = await fetch(apiBase() + "/api/admin/dashboard", {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      var data = await res.json();
      var v = data.visitas || {};

      document.getElementById("statVisitasHoje").textContent = v.visitas_hoje ?? "0";
      document.getElementById("statUnicosHoje").textContent = v.visitantes_unicos_hoje ?? "0";
      document.getElementById("statVisitas7").textContent = v.visitas_7_dias ?? "0";
      document.getElementById("statVisitas30").textContent = v.visitas_30_dias ?? "0";
      document.getElementById("statTransHoje").textContent = data.transcricoes_hoje ?? "0";
      document.getElementById("statTransTotal").textContent = data.transcricoes_total ?? "0";

      var toggle = document.getElementById("manutencaoToggle");
      if (toggle) toggle.checked = !!data.manutencao;
      atualizarStatusManutencao(!!data.manutencao);

      renderCharts(data.charts);
      renderTopPages(data.top_paginas);
      renderVisitasRecentes(data.visitas_recentes || []);
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao carregar painel.", "error");
    }
  }

  function destroyCmsEditors() {
    Object.keys(cmsEditors).forEach(function (key) {
      delete cmsEditors[key];
    });
    var fields = document.getElementById("contentFields");
    if (fields) fields.innerHTML = "";
  }

  function isEmptyQuillHtml(html) {
    var t = (html || "").replace(/\s/g, "");
    return !t || t === "<p><br></p>" || t === "<p></p>";
  }

  function setQuillHtml(quill, html) {
    quill.setText("");
    if (html) quill.clipboard.dangerouslyPasteHTML(html);
  }

  function renderCmsFields(page) {
    destroyCmsEditors();
    var container = document.getElementById("contentFields");
    if (!container || !page) return;

    page.fields.forEach(function (field) {
      var wrap = document.createElement("div");
      wrap.className = "oe-admin-field";
      var label = document.createElement("label");
      label.textContent = field.label;
      wrap.appendChild(label);

      if (field.type === "rich") {
        var editorWrap = document.createElement("div");
        editorWrap.className = "oe-admin-rich-editor";
        if (field.key.indexOf("faq") !== -1 || field.key.indexOf("seo") !== -1) {
          editorWrap.classList.add("oe-admin-rich-editor--tall");
        }
        var editorEl = document.createElement("div");
        editorEl.id = "cmsEditor_" + field.key;
        editorWrap.appendChild(editorEl);
        wrap.appendChild(editorWrap);
        container.appendChild(wrap);

        if (global.Quill) {
          var quill = new Quill(editorEl, {
            theme: "snow",
            modules: { toolbar: QUILL_TOOLBAR },
          });
          var value = cmsContent[field.key] || "";
          setQuillHtml(quill, value);
          cmsEditors[field.key] = quill;
        }
      } else if (field.type === "lines") {
        var ta = document.createElement("textarea");
        ta.name = field.key;
        ta.rows = 7;
        ta.value = cmsContent[field.key] || "";
        wrap.appendChild(ta);
        container.appendChild(wrap);
      } else {
        var input = document.createElement("input");
        input.type = "text";
        input.name = field.key;
        input.value = cmsContent[field.key] || "";
        wrap.appendChild(input);
        container.appendChild(wrap);
      }
    });
  }

  function selectCmsPage(pageId) {
    var page = cmsPages.find(function (p) { return p.id === pageId; });
    if (!page) return;
    cmsCurrentPage = page;
    var select = document.getElementById("cmsPageSelect");
    if (select) select.value = pageId;
    var preview = document.getElementById("cmsPagePreview");
    if (preview) preview.href = page.path.replace(/^\//, "");
    renderCmsFields(page);
    setPageTitle("Conteúdo — " + page.label);
  }

  function populateCmsPageSelect() {
    var select = document.getElementById("cmsPageSelect");
    if (!select) return;
    select.innerHTML = "";
    cmsPages.forEach(function (page) {
      var opt = document.createElement("option");
      opt.value = page.id;
      opt.textContent = page.label + (page.lang === "en" ? " (EN)" : "");
      select.appendChild(opt);
    });
    if (cmsPages.length) selectCmsPage(cmsPages[0].id);
  }

  async function carregarConteudo() {
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      var data = await res.json();
      cmsContent = data.content || {};
      cmsPages = data.pages || [];
      populateCmsPageSelect();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao carregar conteúdo.", "error");
    }
  }

  function collectCmsUpdates() {
    var updates = {};
    if (!cmsCurrentPage) return updates;
    cmsCurrentPage.fields.forEach(function (field) {
      if (field.type === "rich") {
        var quill = cmsEditors[field.key];
        if (quill) {
          var html = quill.root.innerHTML;
          updates[field.key] = isEmptyQuillHtml(html) ? "" : html;
        }
      } else {
        var el = document.querySelector('#contentFields [name="' + field.key + '"]');
        if (el) updates[field.key] = el.value;
      }
    });
    return updates;
  }

  async function guardarConteudo(e) {
    e.preventDefault();
    var updates = collectCmsUpdates();
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        method: "PUT",
        headers: global.OuviescreviAPI.adminAuthHeaders(),
        body: JSON.stringify({ updates: updates }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      var data = await res.json();
      cmsContent = data.content || cmsContent;
      global.OuviescreviUI.toast("Conteúdo guardado.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  async function reporConteudo() {
    if (!cmsCurrentPage) return;
    if (!confirm("Repor os textos de «" + cmsCurrentPage.label + "» aos valores originais?")) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content/reset", {
        method: "POST",
        headers: global.OuviescreviAPI.adminAuthHeaders(),
        body: JSON.stringify({ page: cmsCurrentPage.id }),
      });
      var data = await res.json();
      cmsContent = data.content || {};
      renderCmsFields(cmsCurrentPage);
      global.OuviescreviUI.toast("Textos repostos.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao repor.", "error");
    }
  }

  async function carregarLogs() {
    var div = document.getElementById("tabelaLogs");
    if (!div) return;
    div.innerHTML = '<p class="oe-admin-empty">A carregar...</p>';
    try {
      var res = await fetch(apiBase() + "/api/logs", {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      var data = await res.json();
      var logs = Array.isArray(data) ? data : data.logs || [];
      if (!logs.length) {
        div.innerHTML = '<p class="oe-admin-empty">Sem transcrições registadas.</p>';
        return;
      }
      div.innerHTML = "";
      div.appendChild(
        buildTable(
          ["Ficheiro", "Data"],
          logs
            .slice()
            .reverse()
            .slice(0, 100)
            .map(function (row) {
              return [row.ficheiro || row.file || "—", row.data || row.date || "—"];
            })
        )
      );
    } catch (e) {
      div.innerHTML = '<p class="oe-admin-empty">Erro ao carregar.</p>';
      global.OuviescreviUI.toast("Erro ao carregar transcrições.", "error");
    }
  }

  function mostrarApp() {
    document.getElementById("loginScreen").classList.add("hidden");
    document.getElementById("adminApp").classList.remove("hidden");
    carregarDashboard();
    carregarConteudo();
  }

  function logout() {
    global.OuviescreviAPI.adminLogout();
    document.getElementById("adminApp").classList.add("hidden");
    document.getElementById("loginScreen").classList.remove("hidden");
    document.getElementById("password").value = "";
    if (chartVisitas) { chartVisitas.destroy(); chartVisitas = null; }
    if (chartTranscricoes) { chartTranscricoes.destroy(); chartTranscricoes = null; }
    global.OuviescreviUI.toast("Sessão terminada.");
  }

  function init() {
    document.getElementById("loginForm").addEventListener("submit", async function (e) {
      e.preventDefault();
      try {
        await global.OuviescreviAPI.adminLogin(document.getElementById("password").value);
        global.OuviescreviUI.toast("Sessão iniciada.", "success");
        mostrarApp();
      } catch (err) {
        global.OuviescreviUI.toast("Palavra-chave incorreta.", "error");
      }
    });

    document.querySelectorAll(".oe-admin-nav button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchTab(btn.dataset.tab);
      });
    });

    document.getElementById("manutencaoToggle").addEventListener("change", async function () {
      var toggle = document.getElementById("manutencaoToggle");
      try {
        var res = await fetch(apiBase() + "/api/status", {
          method: "POST",
          headers: global.OuviescreviAPI.adminAuthHeaders(),
          body: JSON.stringify({ manutencao: toggle.checked }),
        });
        var data = await res.json();
        atualizarStatusManutencao(!!data.manutencao);
        global.OuviescreviUI.toast(toggle.checked ? "Manutenção ativada." : "Site operacional.", "success");
      } catch (e) {
        toggle.checked = !toggle.checked;
        global.OuviescreviUI.toast("Erro ao atualizar.", "error");
      }
    });

    document.getElementById("contentForm").addEventListener("submit", guardarConteudo);
    document.getElementById("btnReporConteudo").addEventListener("click", reporConteudo);
    document.getElementById("cmsPageSelect").addEventListener("change", function () {
      selectCmsPage(this.value);
    });
    document.getElementById("btnRefresh").addEventListener("click", carregarDashboard);
    document.getElementById("btnLogoutSide").addEventListener("click", logout);
    document.getElementById("btnLogoutTop").addEventListener("click", logout);
    document.getElementById("adminBurger").addEventListener("click", function () {
      document.getElementById("adminSidebar").classList.toggle("is-open");
    });

    if (global.OuviescreviAPI.isAdminSession()) {
      global.OuviescreviAPI.init().then(mostrarApp);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  global.OuviescreviAdmin = {
    carregarDashboard: carregarDashboard,
    carregarLogs: carregarLogs,
  };
})(window);
