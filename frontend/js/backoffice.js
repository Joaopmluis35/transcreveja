/**
 * Backoffice Ouviescrevi — painel admin com gráficos (Chart.js)
 */
(function (global) {
  let chartVisitas = null;
  let chartTranscricoes = null;
  let chartPeakHours = null;
  let cmsPages = [];
  let cmsAllPages = [];
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

  const ROLE_LEVEL = { viewer: 1, editor: 2, admin: 3 };
  let transOffset = 0;
  const TRANS_PAGE = 50;
  let transTotal = 0;

  function getAdminRole() {
    return sessionStorage.getItem("ouviescrevi_admin_role") || "admin";
  }

  function roleAtLeast(minimum) {
    var role = getAdminRole();
    return (ROLE_LEVEL[role] || 0) >= (ROLE_LEVEL[minimum] || 99);
  }

  function applyRoleUI() {
    var role = getAdminRole();
    document.querySelectorAll("[data-min-role]").forEach(function (el) {
      var min = el.getAttribute("data-min-role") || "viewer";
      el.classList.toggle("hidden", !roleAtLeast(min));
    });
    document.querySelectorAll("[data-admin-only]").forEach(function (el) {
      el.classList.toggle("hidden", role !== "admin");
    });
    var label = document.getElementById("adminUserLabel");
    if (label) {
      var user = sessionStorage.getItem("ouviescrevi_admin_username") || "admin";
      label.textContent = user + " · " + role;
      label.classList.remove("hidden");
    }
  }

  function updateSugestoesBadge(count) {
    var badge = document.getElementById("navBadgeSugestoes");
    if (!badge) return;
    var n = Number(count) || 0;
    if (n > 0) {
      badge.textContent = n > 99 ? "99+" : String(n);
      badge.classList.remove("hidden");
    } else {
      badge.classList.add("hidden");
    }
  }

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
    var titles = { dashboard: "Painel", conteudo: "Conteúdo do site", seo: "SEO", transcricoes: "Transcrições", sugestoes: "Sugestões", sistema: "Sistema" };
    if (tab !== "conteudo") {
      setPageTitle(titles[tab] || "Backoffice");
    } else if (cmsCurrentPage) {
      setPageTitle("Conteúdo — " + cmsCurrentPage.label);
    } else {
      setPageTitle(titles.conteudo);
    }
    document.getElementById("adminSidebar").classList.remove("is-open");
    if (tab === "transcricoes") carregarLogs();
    if (global.OuviescreviAdminExt) global.OuviescreviAdminExt.onTab(tab);
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

  function renderPeakHoursChart(peakRows) {
    if (!global.Chart) return;
    var ctx = document.getElementById("chartPeakHours");
    if (!ctx) return;
    var hours = [];
    for (var h = 0; h < 24; h++) hours.push(String(h).padStart(2, "0"));
    var map = {};
    (peakRows || []).forEach(function (r) {
      map[r.hora] = r.total;
    });
    if (chartPeakHours) chartPeakHours.destroy();
    chartPeakHours = new Chart(ctx, {
      type: "bar",
      data: {
        labels: hours.map(function (h) { return h + "h"; }),
        datasets: [{
          label: "Visitas",
          data: hours.map(function (h) { return map[h] || 0; }),
          backgroundColor: "rgba(37, 99, 235, 0.7)",
          borderRadius: 3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
      },
    });
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
      renderPeakHoursChart((data.charts && data.charts.horas_pico) || []);
      renderTopPages(data.top_paginas);
      updateSugestoesBadge(data.sugestoes_nao_lidas);
      renderVisitasRecentes(data.visitas_recentes || []);
      if (global.OuviescreviAdminExt) global.OuviescreviAdminExt.renderReferrersAndDevices(data);
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

  function pageOptionLabel(page) {
    if (/\((PT|EN|ES|FR|DE)\)/i.test(page.label || "")) return page.label;
    if (page.lang && page.lang !== "pt") {
      return page.label + " (" + String(page.lang).toUpperCase() + ")";
    }
    return page.label;
  }

  function cmsLangOrder(lang) {
    var order = { pt: 0, en: 1, es: 2, fr: 3, de: 4 };
    return order[lang] != null ? order[lang] : 9;
  }

  function filteredCmsPages() {
    var langEl = document.getElementById("cmsLangFilter");
    var lang = langEl ? String(langEl.value || "").trim().toLowerCase() : "";
    var list = cmsAllPages.slice();
    if (lang) {
      list = list.filter(function (p) {
        return String(p.lang || "").toLowerCase() === lang;
      });
    }
    list.sort(function (a, b) {
      var la = cmsLangOrder(a.lang);
      var lb = cmsLangOrder(b.lang);
      if (la !== lb) return la - lb;
      return String(a.label).localeCompare(String(b.label), "pt");
    });
    return list;
  }

  function populateCmsPageSelect() {
    var select = document.getElementById("cmsPageSelect");
    if (!select) return;
    cmsPages = filteredCmsPages();
    var countEl = document.getElementById("cmsPageCount");
    if (countEl) {
      var lang = (document.getElementById("cmsLangFilter") || {}).value || "";
      var hasLocales = cmsAllPages.some(function (p) {
        var l = String(p.lang || "").toLowerCase();
        return l === "es" || l === "fr" || l === "de";
      });
      countEl.textContent = cmsPages.length + " página(s)" + (lang ? " em " + lang.toUpperCase() : "") + ".";
      if (!hasLocales && global.OuviescreviCmsLocales) {
        countEl.textContent += " (páginas ES/FR/DE em modo local — confirma deploy da API no Render para guardar.)";
      } else if (!hasLocales) {
        countEl.textContent += " API no Render sem locales — faz redeploy.";
      }
    }
    select.innerHTML = "";
    if (!cmsPages.length) {
      var empty = document.createElement("option");
      empty.textContent = "Nenhuma página neste idioma";
      empty.value = "";
      select.appendChild(empty);
      destroyCmsEditors();
      return;
    }
    cmsPages.forEach(function (page) {
      var opt = document.createElement("option");
      opt.value = page.id;
      opt.textContent = pageOptionLabel(page);
      select.appendChild(opt);
    });
    var keepId = cmsCurrentPage && cmsPages.some(function (p) { return p.id === cmsCurrentPage.id; })
      ? cmsCurrentPage.id
      : cmsPages[0].id;
    selectCmsPage(keepId);
  }

  async function carregarConteudo() {
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      var data = await res.json();
      cmsContent = data.content || {};
      var fromApi = (data.pages || []).filter(function (p) { return p.category !== "seo"; });
      cmsAllPages = global.OuviescreviCmsLocales
        ? global.OuviescreviCmsLocales.mergeLocaleCmsPages(fromApi)
        : fromApi;
      populateCmsPageSelect();
      if (global.OuviescreviAdminExt) {
        global.OuviescreviAdminExt.setupSeo(data.pages || [], cmsContent);
      }
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

  function formatTransDate(iso) {
    if (!iso) return "—";
    return String(iso).replace("T", " ").replace("Z", "").slice(0, 19);
  }

  function statusBadge(status) {
    var s = (status || "ok").toLowerCase();
    var cls = s === "ok" ? "oe-admin-badge--ok" : "oe-admin-badge--err";
    return '<span class="oe-admin-badge ' + cls + '">' + s + "</span>";
  }

  function transQueryParams() {
    var q = (document.getElementById("transSearch") || {}).value || "";
    var status = (document.getElementById("transStatus") || {}).value || "";
    var dayFrom = (document.getElementById("transDayFrom") || {}).value || "";
    var dayTo = (document.getElementById("transDayTo") || {}).value || "";
    var qs = "?limit=" + TRANS_PAGE + "&offset=" + transOffset;
    if (q) qs += "&q=" + encodeURIComponent(q);
    if (status) qs += "&status=" + encodeURIComponent(status);
    if (dayFrom) qs += "&day_from=" + encodeURIComponent(dayFrom);
    if (dayTo) qs += "&day_to=" + encodeURIComponent(dayTo);
    return qs;
  }

  function renderTransSummary(stats, total) {
    var box = document.getElementById("transSummary");
    if (!box || !stats) return;
    box.innerHTML =
      '<div class="oe-admin-mini-stat"><span>Total</span><strong>' + (total || 0) + "</strong></div>" +
      '<div class="oe-admin-mini-stat"><span>Falhas</span><strong>' + (stats.falhas || 0) + "</strong></div>" +
      '<div class="oe-admin-mini-stat"><span>Proc. médio</span><strong>' + (stats.media_proc_s || 0) + " s</strong></div>" +
      '<div class="oe-admin-mini-stat"><span>Duração média</span><strong>' + (stats.media_dur_s || 0) + " s</strong></div>";
  }

  function renderTransTable(logs, append) {
    var div = document.getElementById("tabelaLogs");
    if (!div) return;
    if (!append) div.innerHTML = "";
    if (!logs.length && !append) {
      div.innerHTML = '<p class="oe-admin-empty">Sem transcrições registadas.</p>';
      return;
    }
    var table = append ? div.querySelector("table") : null;
    if (!table) {
      table = buildTable(
        ["Ficheiro", "Idioma", "MB", "Duração", "Processamento", "Estado", "Erro", "Data"],
        []
      );
      if (!append) div.appendChild(table);
    }
    var tbody = table.querySelector("tbody");
    logs.forEach(function (row) {
      var name = row.ficheiro || "—";
      var short = name.length > 42 ? name.slice(0, 39) + "…" : name;
      var err = row.error_message ? String(row.error_message).slice(0, 60) : "—";
      var tr = document.createElement("tr");
      [
        short,
        row.language || "auto",
        row.size_bytes ? (row.size_bytes / 1048576).toFixed(1) : "—",
        row.duration_sec != null ? Math.round(row.duration_sec) + " s" : "—",
        row.processing_sec != null ? row.processing_sec + " s" : "—",
        row.status || "ok",
        err,
        formatTransDate(row.data),
      ].forEach(function (cell, i) {
        var td = document.createElement("td");
        if (i === 5) {
          td.innerHTML = statusBadge(row.status);
        } else if (i === 6 && row.error_message) {
          td.title = row.error_message;
          td.textContent = err;
          td.className = "oe-admin-cell--err";
        } else {
          td.textContent = cell;
        }
        if (i === 0 && row.ficheiro) td.title = row.ficheiro;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  async function carregarLogs(append) {
    if (append === undefined) append = false;
    var div = document.getElementById("tabelaLogs");
    if (!div) return;
    if (!append) {
      transOffset = 0;
      div.innerHTML = '<p class="oe-admin-empty">A carregar...</p>';
    }
    try {
      var res = await fetch(apiBase() + "/api/admin/transcricoes" + transQueryParams(), {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      var data = await res.json();
      var logs = data.items || [];
      transTotal = data.total || 0;
      renderTransSummary(data.stats, transTotal);
      renderTransTable(logs, !!append);
      var moreBtn = document.getElementById("btnTransMore");
      if (moreBtn) {
        var hasMore = transOffset + logs.length < transTotal;
        moreBtn.classList.toggle("hidden", !hasMore);
      }
    } catch (e) {
      div.innerHTML = '<p class="oe-admin-empty">Erro ao carregar.</p>';
    }
  }

  function carregarMaisLogs() {
    transOffset += TRANS_PAGE;
    carregarLogs(true);
  }

  function mostrarApp() {
    document.getElementById("loginScreen").classList.add("hidden");
    document.getElementById("adminApp").classList.remove("hidden");
    applyRoleUI();
    carregarDashboard();
    if (roleAtLeast("editor")) carregarConteudo();
  }

  function logout() {
    global.OuviescreviAPI.adminLogout();
    document.getElementById("adminApp").classList.add("hidden");
    document.getElementById("loginScreen").classList.remove("hidden");
    document.getElementById("password").value = "";
    if (chartVisitas) { chartVisitas.destroy(); chartVisitas = null; }
    if (chartTranscricoes) { chartTranscricoes.destroy(); chartTranscricoes = null; }
    if (chartPeakHours) { chartPeakHours.destroy(); chartPeakHours = null; }
    global.OuviescreviUI.toast("Sessão terminada.");
  }

  function init() {
    document.getElementById("loginForm").addEventListener("submit", async function (e) {
      e.preventDefault();
      try {
        await global.OuviescreviAPI.adminLogin(
          document.getElementById("password").value,
          (document.getElementById("adminUsername") || {}).value
        );
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
      if (!roleAtLeast("admin")) {
        toggle.checked = !toggle.checked;
        global.OuviescreviUI.toast("Sem permissão.", "error");
        return;
      }
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
    var cmsLang = document.getElementById("cmsLangFilter");
    if (cmsLang) {
      cmsLang.addEventListener("change", function () {
        cmsCurrentPage = null;
        populateCmsPageSelect();
      });
    }
    document.getElementById("btnTransFilter").addEventListener("click", function () { carregarLogs(false); });
    var btnMore = document.getElementById("btnTransMore");
    if (btnMore) btnMore.addEventListener("click", carregarMaisLogs);

    document.getElementById("btnRefresh").addEventListener("click", function () {
      carregarDashboard();
      var active = document.querySelector(".oe-admin-nav button.is-active");
      if (active && active.dataset.tab === "transcricoes") carregarLogs();
      if (active && active.dataset.tab === "sistema" && global.OuviescreviAdminExt) {
        global.OuviescreviAdminExt.loadSystem();
      }
    });
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
    buildTable: buildTable,
    roleAtLeast: roleAtLeast,
    updateSugestoesBadge: updateSugestoesBadge,
    applyRoleUI: applyRoleUI,
  };
})(window);
