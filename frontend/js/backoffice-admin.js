/**
 * Backoffice — funcionalidades avançadas (SEO, sistema, sugestões, export).
 */
(function (global) {
  function apiBase() {
    return global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase();
  }

  function authHeaders() {
    return global.OuviescreviAPI.adminAuthHeaders();
  }

  var sugestoesCache = [];
  var aiInsightsCache = [];
  var aiInsightsSummaryText = "";

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function apiErrorMessage(body, fallback) {
    var detail = body && body.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length) {
      var first = detail[0];
      if (first && typeof first.msg === "string") return first.msg;
    }
    if (body && typeof body.error === "string" && body.error.trim()) return body.error;
    return fallback || "Ocorreu um erro.";
  }

  function roleLabel(role) {
    if (role === "admin") return "Admin";
    if (role === "editor") return "Editor";
    if (role === "viewer") return "Viewer";
    return role || "—";
  }

  function formatSugestaoDate(iso) {
    if (!iso) return "—";
    return String(iso).replace("T", " ").replace("Z", "").slice(0, 19);
  }

  function buildTable(headers, rows) {
    if (global.OuviescreviAdmin && global.OuviescreviAdmin.buildTable) {
      return global.OuviescreviAdmin.buildTable(headers, rows);
    }
    var table = document.createElement("table");
    table.className = "oe-admin-table";
    return table;
  }

  var seoPages = [];
  var seoAllPages = [];
  var seoContent = {};
  var seoCurrentPage = null;
  var chartCloudflare = null;
  var serverLogTimer = null;
  var lastServerLogText = "";

  function safeToast(message, type) {
    if (global.OuviescreviUI && global.OuviescreviUI.toast) {
      global.OuviescreviUI.toast(message, type);
    }
  }

  function downloadBlob(blob, name) {
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.rel = "noopener";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      if (a.parentNode) a.parentNode.removeChild(a);
      URL.revokeObjectURL(url);
    }, 2000);
  }

  function fetchWithTimeout(url, options, timeoutMs) {
    var ctrl = new AbortController();
    var timer = setTimeout(function () {
      ctrl.abort();
    }, timeoutMs || 12000);
    var opts = Object.assign({}, options || {}, { signal: ctrl.signal });
    return fetch(url, opts).finally(function () {
      clearTimeout(timer);
    });
  }

  function dashboardCacheKey() {
    return "oe_admin_dashboard_v1";
  }

  function buildVisitReportFromDashboard(data, days) {
    var nDays = Math.max(1, Math.min(parseInt(days, 10) || 7, 90));
    var charts = (data && data.charts) || {};
    var series = charts.visitas_diarias || [];
    var slice = series.slice(-nDays);
    var byDay = {};
    var totalPv = 0;
    var totalHumans = 0;
    var totalBots = 0;
    var totalOwner = 0;
    slice.forEach(function (row) {
      if (!row || !row.day) return;
      var pv = row.total || 0;
      var humans = row.outros != null ? row.outros : pv;
      var bots = row.bots || 0;
      var owner = row.tuas || 0;
      byDay[row.day] = {
        pageviews: pv,
        unicos: row.unicos || 0,
        human_pageviews: humans,
        bot_pageviews: bots,
        owner_pageviews: owner,
        legacy_pageviews: null,
      };
      totalPv += pv;
      totalHumans += humans;
      totalBots += bots;
      totalOwner += owner;
    });
    var dayFrom = slice.length ? slice[0].day : null;
    var dayTo = slice.length ? slice[slice.length - 1].day : null;
    return {
      exported_at: new Date().toISOString(),
      purpose: "Análise últimos " + nDays + " dia(s) — gerado a partir do painel (fallback)",
      source: { note: "dashboard-cache-fallback", database_backend: "unknown" },
      range: {
        days: nDays,
        from: dayFrom,
        to: dayTo,
        hoje: dayTo,
        ontem: slice.length > 1 ? slice[slice.length - 2].day : null,
      },
      by_day: byDay,
      totals: {
        pageviews: totalPv,
        unicos: null,
        human_unicos_approx: null,
        human_pageviews: totalHumans,
        bot_pageviews: totalBots,
        owner_pageviews: totalOwner,
      },
      totals_2d: {
        pageviews: totalPv,
        human_pageviews: totalHumans,
        bot_pageviews: totalBots,
        owner_pageviews: totalOwner,
      },
      visitas: data.visitas || {},
      trafego_hoje: data.visitas_trafego || {},
      conversao_hoje: data.conversao || {},
      conversao_por_idioma_14d: data.conversao_por_idioma || [],
      series_14d: series,
      visitantes_distintos: data.visitantes_distintos || [],
      visitas_recentes: data.visitas_recentes || [],
      top_pages_30d: data.top_paginas || [],
      top_referrers: data.top_referrers || [],
      devices: data.devices || [],
      owner_ip_labels: data.owner_ip_labels || [],
      owner_uids_count: (data.owner_visitor_uids || []).length,
      fallback: true,
    };
  }

  function loadDashboardCache() {
    try {
      var raw = sessionStorage.getItem(dashboardCacheKey());
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function selectedVisitReportDays() {
    var sel = document.getElementById("visitReportDays");
    var n = sel ? parseInt(sel.value, 10) : 7;
    if (isNaN(n) || n < 1) n = 7;
    return Math.min(n, 90);
  }

  function updateVisitReportHint() {
    var hint = document.getElementById("visitReportHint");
    if (!hint) return;
    var n = selectedVisitReportDays();
    hint.textContent =
      n === 2
        ? "Ontem + hoje — partilhar no chat para analisar"
        : "Últimos " + n + " dias — partilhar no chat para analisar";
  }

  function fetchVisitReport() {
    var days = selectedVisitReportDays();
    var base = apiBase();
    return fetchWithTimeout(
      base + "/api/admin/export/visit-report?days=" + encodeURIComponent(days),
      { headers: authHeaders() },
      15000
    )
      .then(function (r) {
        if (r.ok) return r.json();
        return r.text().then(function (body) {
          var detail = "";
          try {
            var j = JSON.parse(body);
            detail = typeof j.detail === "string" ? j.detail : "";
          } catch (e) {}
          var err = new Error(detail || ("HTTP " + r.status));
          err.status = r.status;
          throw err;
        });
      })
      .catch(function (err) {
        var cached = loadDashboardCache();
        if (cached && (cached.visitas || cached.charts)) {
          var fromCache = buildVisitReportFromDashboard(cached, days);
          fromCache.fallback = true;
          fromCache.fallback_reason = (err && err.message) || "api_error";
          return fromCache;
        }
        throw err;
      });
  }

  function copyTextFallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;left:-9999px;top:0;opacity:0;";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    var ok = false;
    try {
      ok = document.execCommand("copy");
    } catch (e) {
      ok = false;
    }
    document.body.removeChild(ta);
    if (!ok) throw new Error("clipboard_unavailable");
  }

  function copyTextToClipboard(text) {
    try {
      if (global.focus) global.focus();
    } catch (e) {}
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      return navigator.clipboard.writeText(text).catch(function () {
        copyTextFallback(text);
      });
    }
    return Promise.resolve().then(function () {
      copyTextFallback(text);
    });
  }

  function deliverVisitReport(data, mode) {
    var text = JSON.stringify(data, null, 2);
    var rng = data.range || {};
    var day = rng.to || rng.hoje || "hoje";
    var nDays = rng.days || selectedVisitReportDays();
    if (mode === "copy") {
      return copyTextToClipboard(text)
        .then(function () {
          var msg = data.fallback
            ? "JSON (painel) copiado — cola no chat Cursor."
            : "JSON copiado — cola no chat Cursor.";
          safeToast(msg, "success");
        })
        .catch(function () {
          downloadBlob(
            new Blob([text], { type: "application/json" }),
            "ouviescrevi-visitas-" + nDays + "d-" + day + ".json"
          );
          safeToast("Clipboard indisponível — JSON descarregado em alternativa.", "success");
        });
    }
    downloadBlob(
      new Blob([text], { type: "application/json" }),
      "ouviescrevi-visitas-" + nDays + "d-" + day + ".json"
    );
    var dlMsg = data.fallback
      ? "Relatório (painel) descarregado — cola no chat."
      : "Relatório descarregado — cola no chat para analisar.";
    safeToast(dlMsg, "success");
    return Promise.resolve();
  }

  function runVisitReportExport(mode) {
    safeToast("A preparar exportação…", "success");
    return fetchVisitReport()
      .then(function (data) {
        return deliverVisitReport(data, mode);
      })
      .catch(function (err) {
        var msg = (err && err.name === "AbortError")
          ? "Tempo esgotado — tenta «Copiar JSON» ou atualiza o painel."
          : ((err && err.message) || "desconhecido");
        safeToast("Erro ao exportar: " + msg, "error");
      });
  }

  function initVisitReportExport() {
    var btnDl = document.getElementById("btnExportVisitReport");
    var btnCopy = document.getElementById("btnCopyVisitReport");
    var linkLegacy = document.getElementById("exportVisitReport");
    var daysSel = document.getElementById("visitReportDays");

    if (daysSel && daysSel.dataset.oeVisitDaysBound !== "1") {
      daysSel.dataset.oeVisitDaysBound = "1";
      daysSel.addEventListener("change", updateVisitReportHint);
      updateVisitReportHint();
    }

    function bindOnce(el, mode) {
      if (!el || el.dataset.oeVisitExportBound === "1") return;
      el.dataset.oeVisitExportBound = "1";
      el.addEventListener("click", function (e) {
        if (e && e.preventDefault) e.preventDefault();
        if (el.disabled) return;
        el.disabled = true;
        var prev = el.textContent;
        if (mode === "download") el.textContent = "A exportar…";
        runVisitReportExport(mode).finally(function () {
          el.disabled = false;
          if (mode === "download" && prev) el.textContent = prev;
        });
      });
    }

    bindOnce(btnDl, "download");
    bindOnce(btnCopy, "copy");
    bindOnce(linkLegacy, "download");
  }

  function initExports() {
    var base = apiBase();
    ["exportVisitas", "exportTranscricoes", "exportBackup"].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      var table = id === "exportVisitas" ? "visitas" : id === "exportTranscricoes" ? "transcricoes" : "backup";
      var path = table === "backup" ? "/api/admin/backup" : "/api/admin/export/" + table;
      el.href = base + path + (table === "backup" ? "" : "");
      el.onclick = function (e) {
        e.preventDefault();
        fetch(base + path, { headers: authHeaders() })
          .then(function (r) { return table === "backup" ? r.json() : r.text(); })
          .then(function (data) {
            var blob;
            var name;
            if (table === "backup") {
              blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
              name = "ouviescrevi-backup.json";
            } else {
              blob = new Blob([data], { type: "text/csv" });
              name = table + ".csv";
            }
            downloadBlob(blob, name);
          })
          .catch(function () {
            global.OuviescreviUI.toast("Erro ao exportar.", "error");
          });
      };
    });
    initVisitReportExport();
  }

  function renderReferrersAndDevices(data) {
    var refDiv = document.getElementById("topReferrers");
    var devDiv = document.getElementById("deviceStats");
    if (refDiv) {
      refDiv.innerHTML = "";
      var refs = data.top_referrers || [];
      if (!refs.length) refDiv.innerHTML = '<p class="oe-admin-empty">Sem referrers.</p>';
      else refDiv.appendChild(buildTable(["Origem", "Visitas"], refs.map(function (r) { return [r.referrer, String(r.total)]; })));
      var utms = data.top_utm || [];
      if (utms.length) {
        var utmWrap = document.createElement("div");
        utmWrap.style.marginTop = "12px";
        utmWrap.appendChild(document.createElement("div")).textContent = "Campanhas UTM (30d)";
        utmWrap.firstChild.style.cssText = "font-size:0.75rem;color:var(--bo-muted);margin-bottom:6px";
        utmWrap.appendChild(
          buildTable(
            ["Source", "Medium", "Campaign", "Visitas"],
            utms.map(function (u) {
              return [u.utm_source, u.utm_medium, u.utm_campaign, String(u.total)];
            })
          )
        );
        refDiv.appendChild(utmWrap);
      }
    }
    if (devDiv) {
      devDiv.innerHTML = "";
      var devs = data.devices || [];
      if (!devs.length) devDiv.innerHTML = '<p class="oe-admin-empty">Sem dados.</p>';
      else devDiv.appendChild(buildTable(["Dispositivo", "Visitas"], devs.map(function (d) { return [d.device, String(d.total)]; })));
    }
    var conv = data.conversao || {};
    var el = document.getElementById("statConversao");
    if (el) el.textContent = (conv.taxa_conversao_pct || 0) + "%";
    var cost = data.custos_openai || {};
    var cEl = document.getElementById("statCusto");
    if (cEl) {
      var usd = Number(cost.custo_estimado_usd);
      cEl.textContent = "$" + (isNaN(usd) ? "0.00" : usd.toFixed(usd < 0.01 && usd > 0 ? 4 : 2));
    }
    var cSub = document.getElementById("statCustoSub");
    if (cSub) {
      var mins = Number(cost.minutos_audio_total);
      var rate = Number(cost.taxa_por_minuto_usd);
      if (!isNaN(mins) && mins > 0) {
        cSub.textContent =
          mins.toFixed(1) + " min áudio · $" + (isNaN(rate) ? "0.006" : rate.toFixed(3)) + "/min";
      } else {
        cSub.textContent = "Sem minutos de áudio registados";
      }
    }
    var msg = document.getElementById("maintenanceMessage");
    if (msg && data.maintenance_message) msg.value = data.maintenance_message.replace(/<[^>]+>/g, function (t) {
      return t === "<br>" || t === "<br/>" ? "\n" : "";
    }).replace(/<p>/gi, "").replace(/<\/p>/gi, "\n").replace(/<[^>]+>/g, "").trim();
    renderDashboardAlerts(data);
    renderCloudflare(data.cloudflare);
  }

  function renderDashboardAlerts(data) {
    var box = document.getElementById("dashboardAlerts");
    if (!box) return;
    var alerts = [];
    var unread = data.sugestoes_nao_lidas || 0;
    if (unread > 0) {
      alerts.push({
        type: "warn",
        html: "<strong>" + unread + " sugestão" + (unread === 1 ? "" : "ões") + " nova" + (unread === 1 ? "" : "s") + "</strong> — revê no separador Sugestões.",
      });
    }
    var jobsAtivos = data.jobs_ativos || 0;
    if (jobsAtivos > 0) {
      alerts.push({
        type: "info",
        html:
          "<strong>" +
          jobsAtivos +
          " tarefa" +
          (jobsAtivos === 1 ? "" : "s") +
          " em curso</strong> — vê o progresso no separador <a href=\"#\" data-oe-goto-logs>Logs</a>.",
      });
    }
    var apiErr = data.api_errors_24h || 0;
    if (apiErr > 0) {
      alerts.push({
        type: "warn",
        html:
          "<strong>" +
          apiErr +
          " erro" +
          (apiErr === 1 ? "" : "s") +
          " de API (24h)</strong> — detalhe no separador <a href=\"#\" data-oe-goto-logs>Logs</a>.",
      });
    }
    var tErr = data.transcricoes_erros_hoje || 0;
    if (tErr > 0) {
      alerts.push({
        type: "warn",
        html:
          "<strong>" +
          tErr +
          " transcrição" +
          (tErr === 1 ? "" : "ões") +
          " com erro hoje</strong> — revê no separador Transcrições.",
      });
    }
    var tLimit = data.alert_transcriptions_daily || 0;
    var tHoje = data.transcricoes_hoje || 0;
    if (tLimit > 0 && tHoje >= tLimit) {
      alerts.push({
        type: "warn",
        html: "<strong>Limite diário de transcrições atingido</strong> — " + tHoje + " hoje (limite " + tLimit + ").",
      });
    }
    var vLimit = data.alert_visits_daily || 0;
    var vHoje = (data.visitas && data.visitas.visitas_hoje) || 0;
    if (vLimit > 0 && vHoje >= vLimit) {
      alerts.push({
        type: "warn",
        html: "<strong>Limite diário de visitas atingido</strong> — " + vHoje + " hoje (limite " + vLimit + ").",
      });
    }
    if (data.database_persistent === false) {
      alerts.push({
        type: "warn",
        html:
          "<strong>⚠ Base de dados NÃO persistente (SQLite local)</strong> — visitas e transcrições perdem-se em cada deploy. " +
          "Configura <code>TURSO_DATABASE_URL</code> e <code>TURSO_AUTH_TOKEN</code> no Render → Environment.",
      });
    }
    var vStats = data.visitas || {};
    var vTotal = vStats.visitas_total ?? data.visitas_total ?? 0;
    var v30 = vStats.visitas_30_dias ?? 0;
    if (vTotal > 0 && !v30 && !vStats.visitas_hoje) {
      alerts.push({
        type: "warn",
        html:
          "<strong>Visitas antigas na base (" + vTotal + ")</strong> — fora da janela de 30 dias. " +
          "Os contadores recentes ficam a zero até haver tráfego novo.",
      });
    }
    if (vTotal === 0 && (data.transcricoes_total || 0) === 0) {
      alerts.push({
        type: "info",
        html:
          "<strong>Sem dados ainda</strong> — visita o site público (noutro separador) e faz uma transcrição de teste. " +
          "Depois clica <em>Atualizar</em> no painel.",
      });
    }
    if (!alerts.length) {
      box.classList.add("hidden");
      box.innerHTML = "";
      return;
    }
    box.classList.remove("hidden");
    box.innerHTML = alerts.map(function (a) {
      return '<div class="oe-admin-alert oe-admin-alert--' + a.type + '">' + a.html + "</div>";
    }).join("");
    box.querySelectorAll("[data-oe-goto-logs]").forEach(function (link) {
      link.addEventListener("click", function (e) {
        e.preventDefault();
        var nav = document.querySelector('.oe-admin-nav button[data-tab="logs"]');
        if (nav) nav.click();
      });
    });
    var sugEl = document.getElementById("statSugestoesNovas");
    if (sugEl) sugEl.textContent = String(unread);
    var card = document.getElementById("cardSugestoes");
    if (card) card.classList.toggle("oe-admin-card--pulse", unread > 0);
    var logsBadge = document.getElementById("navBadgeLogs");
    if (logsBadge) {
      var badgeN = jobsAtivos || apiErr;
      if (badgeN > 0) {
        logsBadge.textContent = String(badgeN);
        logsBadge.classList.remove("hidden");
      } else {
        logsBadge.classList.add("hidden");
      }
    }
  }

  function parseCloudflareRows(raw) {
    if (!raw || raw.errors) return [];
    try {
      var zones = (((raw.data || {}).viewer || {}).zones || []);
      var groups = (zones[0] || {}).httpRequests1dGroups || [];
      return groups.map(function (g) {
        return {
          date: (g.dimensions || {}).date || "—",
          requests: ((g.sum || {}).requests) || 0,
        };
      }).sort(function (a, b) { return String(a.date).localeCompare(String(b.date)); });
    } catch (e) {
      return [];
    }
  }

  function renderCloudflare(raw) {
    var div = document.getElementById("cloudflareStats");
    if (!div) return;
    if (!raw) {
      if (chartCloudflare) { chartCloudflare.destroy(); chartCloudflare = null; }
      div.innerHTML = '<p class="oe-admin-empty">Configura Zone ID e API Token em Sistema para ver tráfego.</p>';
      return;
    }
    if (raw.errors && raw.errors.length) {
      if (chartCloudflare) { chartCloudflare.destroy(); chartCloudflare = null; }
      div.innerHTML = '<p class="oe-admin-empty">Erro Cloudflare: ' + String((raw.errors[0] || {}).message || "token inválido") + "</p>";
      return;
    }
    var rows = parseCloudflareRows(raw);
    if (!rows.length) {
      if (chartCloudflare) { chartCloudflare.destroy(); chartCloudflare = null; }
      div.innerHTML = '<p class="oe-admin-empty">Sem dados de tráfego nos últimos 7 dias.</p>';
      return;
    }
    var total = rows.reduce(function (sum, r) { return sum + r.requests; }, 0);
    div.innerHTML = "";
    var summary = document.createElement("p");
    summary.className = "oe-admin-cms-hint";
    summary.textContent = "Total pedidos (7 dias): " + total.toLocaleString("pt-PT");
    div.appendChild(summary);
    div.appendChild(buildTable(
      ["Data", "Pedidos"],
      rows.map(function (r) { return [r.date, String(r.requests)]; })
    ));
    var ctx = document.getElementById("chartCloudflare");
    if (ctx && global.Chart) {
      if (chartCloudflare) chartCloudflare.destroy();
      chartCloudflare = new Chart(ctx, {
        type: "bar",
        data: {
          labels: rows.map(function (r) { return r.date; }),
          datasets: [{
            label: "Pedidos HTTP",
            data: rows.map(function (r) { return r.requests; }),
            backgroundColor: "rgba(37, 99, 235, 0.75)",
            borderRadius: 4,
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
  }

  function destroyCharts() {
    if (chartCloudflare) { chartCloudflare.destroy(); chartCloudflare = null; }
  }

  function apiErrorDetail(data) {
    var d = data && data.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d.map(function (x) { return (x && x.msg) || x; }).join(" ");
    }
    return (data && data.error) || "Falha no envio";
  }

  async function testAlertEmail() {
    try {
      var res = await fetch(apiBase() + "/api/admin/test-alert-email", {
        method: "POST",
        headers: authHeaders(),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorDetail(data));
      global.OuviescreviUI.toast("Email de teste enviado para " + (data.to || "destinatário") + ".", "success");
      loadEmails();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro ao enviar email de teste.", "error");
    }
  }

  async function testActivityEmail() {
    try {
      var res = await fetch(apiBase() + "/api/admin/test-activity-email", {
        method: "POST",
        headers: authHeaders(),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorDetail(data));
      global.OuviescreviUI.toast("Notificação de atividade enviada para " + (data.to || "destinatário") + ".", "success");
      loadEmails();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro ao enviar notificação de teste.", "error");
    }
  }

  function emailKindLabel(kind) {
    var map = {
      activity: "Atividade",
      activity_test: "Teste atividade",
      alert: "Alerta",
      alert_test: "Teste alerta",
    };
    return map[kind] || kind || "—";
  }

  function emailStatusLabel(status) {
    if (status === "sent") return "✅ Enviado";
    if (status === "failed") return "❌ Falhou";
    if (status === "skipped") return "⏭️ Omitido";
    return status || "—";
  }

  function renderEmailStatusCards(status) {
    var grid = document.getElementById("emailStatusCards");
    if (!grid) return;
    var ready = status.provider_ready;
    var cards = [
      {
        cls: ready ? "oe-admin-card--green" : "oe-admin-card--amber",
        label: "Envio configurado",
        value: ready ? "Sim" : "Não",
        sub: ready ? "Pronto a enviar" : "Falta RESEND ou SMTP no Render",
      },
      {
        cls: status.resend_configured ? "oe-admin-card--green" : "oe-admin-card--purple",
        label: "Resend (HTTPS)",
        value: status.resend_configured ? "Ativo" : "Inativo",
        sub: "Recomendado no Render",
      },
      {
        cls: status.smtp_configured ? "oe-admin-card--blue" : "oe-admin-card--purple",
        label: "SMTP",
        value: status.smtp_configured ? (status.smtp_fallback ? "Ativo" : "Ignorado") : "Inativo",
        sub: status.smtp_fallback ? "Pode estar bloqueado no Render" : "Só Resend (fallback desativado)",
      },
      {
        cls: "oe-admin-card--blue",
        label: "Destinatário",
        value: status.alert_email_to || status.default_to || "—",
        sub: "Notificações e alertas",
      },
    ];
    grid.innerHTML = "";
    cards.forEach(function (c) {
      var card = document.createElement("div");
      card.className = "oe-admin-card " + c.cls;
      card.innerHTML =
        '<div class="oe-admin-card__label">' + c.label + "</div>" +
        '<div class="oe-admin-card__value" style="font-size:1rem;word-break:break-all">' + c.value + "</div>" +
        '<div class="oe-admin-card__sub">' + c.sub + "</div>";
      grid.appendChild(card);
    });
    var hint = document.getElementById("emailEnvHint");
    if (hint && status.render_hint) {
      hint.textContent = status.render_hint;
    }
    var failBox = document.getElementById("emailLastFailure");
    if (failBox) {
      var lf = status.last_failure;
      if (lf && lf.status === "failed") {
        failBox.hidden = false;
        failBox.innerHTML =
          "<strong>Último envio falhou</strong> (" + (lf.created_at || "—") + "): " +
          (lf.detail || "sem detalhe") +
          (lf.recipient ? " → " + lf.recipient : "");
      } else {
        failBox.hidden = true;
        failBox.textContent = "";
      }
    }
  }

  async function loadEmailLogs() {
    var box = document.getElementById("emailLogTable");
    if (!box) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/email/logs?limit=40", { headers: authHeaders() });
      if (!res.ok) throw new Error();
      var data = await res.json();
      var items = data.items || [];
      if (!items.length) {
        box.innerHTML = "<p class='oe-admin-empty'>Ainda sem envios registados.</p>";
        return;
      }
      box.innerHTML = "";
      box.appendChild(
        buildTable(
          ["Quando", "Tipo", "Destino", "Assunto", "Estado", "Detalhe"],
          items.map(function (row) {
            return [
              (row.created_at || "").replace("T", " ").slice(0, 19),
              emailKindLabel(row.kind),
              row.recipient || "—",
              row.subject || "—",
              emailStatusLabel(row.status),
              (row.detail || row.actor || "—").toString().slice(0, 200),
            ];
          })
        )
      );
    } catch (e) {
      box.innerHTML = "<p class='oe-admin-empty'>Erro ao carregar histórico.</p>";
    }
  }

  async function loadEmails() {
    try {
      var sres = await fetch(apiBase() + "/api/admin/email/status", { headers: authHeaders() });
      if (!sres.ok) throw new Error("status");
      var status = await sres.json();
      renderEmailStatusCards(status);
      setField("emailCfgTo", status.alert_email_to || status.default_to || "");
      setChecked("emailCfgActivity", status.notify_activity_enabled !== false);
      setChecked("emailCfgAlerts", !!status.alert_email_enabled);
      setField("emailCfgTransLimit", String(status.alert_transcriptions_daily || ""));
      setField("emailCfgVisitLimit", String(status.alert_visits_daily || ""));
      var mHint = document.getElementById("marketingOptInHint");
      if (mHint) {
        mHint.textContent =
          "Opt-in marketing: " +
          (status.marketing_opt_in_count != null ? status.marketing_opt_in_count : "—") +
          " · Search Console: submete https://www.ouviescrevi.pt/sitemap.xml";
      }
    } catch (e) {
      var grid = document.getElementById("emailStatusCards");
      if (grid) grid.innerHTML = "<p class='oe-admin-empty'>Erro ao carregar estado de email.</p>";
    }
    await loadEmailLogs();
  }

  async function sendLifecycleEmail(kind) {
    try {
      var res = await fetch(apiBase() + "/api/admin/marketing/send-lifecycle", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ kind: kind, limit: 50 }),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(data.detail || "Falha");
      global.OuviescreviUI.toast(
        "Enviado: " + (data.sent || 0) + " / " + (data.recipients || 0) + " (falhas: " + (data.failed || 0) + ")",
        "success"
      );
      loadEmailLogs();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro ao enviar.", "error");
    }
  }

  async function saveEmailConfig(e) {
    e.preventDefault();
    var updates = {
      alert_email_to: document.getElementById("emailCfgTo").value.trim().replace(/\s+/g, ""),
      notify_activity_enabled: document.getElementById("emailCfgActivity").checked ? "1" : "0",
      alert_email_enabled: document.getElementById("emailCfgAlerts").checked ? "1" : "0",
      alert_transcriptions_daily: document.getElementById("emailCfgTransLimit").value,
      alert_visits_daily: document.getElementById("emailCfgVisitLimit").value,
    };
    try {
      var res = await fetch(apiBase() + "/api/admin/config", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ updates: updates }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Configuração de email guardada.", "success");
      loadEmails();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  async function saveMaintenanceMessage() {
    var msg = document.getElementById("maintenanceMessage");
    var text = msg ? "<p>" + msg.value.replace(/\n/g, "<br>") + "</p>" : "";
    try {
      var res = await fetch(apiBase() + "/api/admin/maintenance", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({
          manutencao: document.getElementById("manutencaoToggle").checked,
          maintenance_message: text,
        }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Mensagem guardada.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  function seoLangOrder(lang) {
    var order = { pt: 0, en: 1, es: 2, fr: 3, de: 4 };
    return order[lang] != null ? order[lang] : 9;
  }

  function filteredSeoPages() {
    var langEl = document.getElementById("seoLangFilter");
    var lang = langEl ? String(langEl.value || "").trim().toLowerCase() : "";
    var list = seoAllPages.slice();
    if (lang) {
      list = list.filter(function (p) {
        return String(p.lang || "").toLowerCase() === lang;
      });
    }
    list.sort(function (a, b) {
      var la = seoLangOrder(a.lang);
      var lb = seoLangOrder(b.lang);
      if (la !== lb) return la - lb;
      return String(a.label).localeCompare(String(b.label), "pt");
    });
    return list;
  }

  function populateSeoPageSelect() {
    var sel = document.getElementById("seoPageSelect");
    if (!sel) return;
    seoPages = filteredSeoPages();
    sel.innerHTML = "";
    if (!seoPages.length) {
      var empty = document.createElement("option");
      empty.textContent = "Nenhuma página neste idioma";
      empty.value = "";
      sel.appendChild(empty);
      var box = document.getElementById("seoFields");
      if (box) box.innerHTML = "";
      return;
    }
    seoPages.forEach(function (p) {
      var opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      sel.appendChild(opt);
    });
    var keepId = seoCurrentPage && seoPages.some(function (p) { return p.id === seoCurrentPage.id; })
      ? seoCurrentPage.id
      : seoPages[0].id;
    var page = seoPages.find(function (p) { return p.id === keepId; });
    if (page) {
      seoCurrentPage = page;
      sel.value = page.id;
      renderSeoFields(page);
      updateSeoPreview(page);
    }
  }

  function updateSeoPreview(page) {
    var link = document.getElementById("seoPagePreview");
    if (!link || !page || !page.path) return;
    link.href = page.path.replace(/^\//, "");
  }

  function setupSeo(pages, content) {
    var fromApi = (pages || []).filter(function (p) { return p.category === "seo"; });
    seoAllPages = global.OuviescreviCmsLocales && global.OuviescreviCmsLocales.mergeLocaleSeoPages
      ? global.OuviescreviCmsLocales.mergeLocaleSeoPages(fromApi)
      : fromApi;
    seoContent = content || {};
    if (global.OuviescreviCmsLocales && global.OuviescreviCmsLocales.mergeLocaleSeoContent) {
      seoContent = global.OuviescreviCmsLocales.mergeLocaleSeoContent(seoContent);
    }
    var langEl = document.getElementById("seoLangFilter");
    if (langEl && !langEl.dataset.bound) {
      langEl.dataset.bound = "1";
      langEl.addEventListener("change", populateSeoPageSelect);
    }
    var sel = document.getElementById("seoPageSelect");
    if (sel && !sel.dataset.bound) {
      sel.dataset.bound = "1";
      sel.onchange = function () {
        var page = seoPages.find(function (p) { return p.id === sel.value; });
        if (page) {
          seoCurrentPage = page;
          renderSeoFields(page);
          updateSeoPreview(page);
        }
      };
    }
    populateSeoPageSelect();
  }

  function seoCharLimit(key) {
    if (key.indexOf("title") !== -1) return 60;
    if (key.indexOf("description") !== -1) return 160;
    return 0;
  }

  function attachSeoCounter(input, limit) {
    var counter = document.createElement("small");
    counter.className = "oe-admin-char-count";
    function refresh() {
      var len = (input.value || "").length;
      counter.textContent = len + " / " + limit;
      counter.classList.toggle("oe-admin-char-count--warn", len > limit);
    }
    input.addEventListener("input", refresh);
    refresh();
    input.parentNode.appendChild(counter);
  }

  function renderSeoFields(page) {
    var box = document.getElementById("seoFields");
    if (!box) return;
    box.innerHTML = "";
    page.fields.forEach(function (field) {
      var wrap = document.createElement("div");
      wrap.className = "oe-admin-field";
      var label = document.createElement("label");
      label.textContent = field.label;
      wrap.appendChild(label);
      var input = document.createElement("input");
      input.type = "text";
      input.name = field.key;
      input.value = seoContent[field.key] || "";
      wrap.appendChild(input);
      var limit = seoCharLimit(field.key);
      if (limit) attachSeoCounter(input, limit);
      box.appendChild(wrap);
    });
    updateSeoPreview(page);
  }

  async function saveSeo(e) {
    e.preventDefault();
    var updates = {};
    document.querySelectorAll("#seoFields [name]").forEach(function (el) {
      updates[el.name] = el.value;
    });
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ updates: updates }),
      });
      if (!res.ok) throw new Error();
      var data = await res.json();
      seoContent = data.content || seoContent;
      global.OuviescreviUI.toast("SEO guardado.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar SEO.", "error");
    }
  }

  async function loadSugestoes() {
    var div = document.getElementById("tabelaSugestoes");
    if (!div) return;
    var unreadOnly = !!(document.getElementById("sugUnreadOnly") || {}).checked;
    var lang = (document.getElementById("sugLangFilter") || {}).value || "";
    try {
      var qs = [];
      if (unreadOnly) qs.push("unread_only=true");
      if (lang) qs.push("lang=" + encodeURIComponent(lang));
      var res = await fetch(
        apiBase() + "/api/admin/sugestoes" + (qs.length ? "?" + qs.join("&") : ""),
        { headers: authHeaders() }
      );
      var data = await res.json();
      var items = data.items || [];
      sugestoesCache = items;
      if (!items.length) {
        div.innerHTML = '<p class="oe-admin-empty">Sem sugestões.</p>';
        return;
      }
      div.innerHTML = "";
      var table = buildTable(
        ["Nome", "Mensagem", "Idioma", "Data", "Estado", "Ações"],
        items.map(function (s) {
          var preview = (s.mensagem || "").replace(/\s+/g, " ").trim();
          if (preview.length > 80) preview = preview.slice(0, 80) + "…";
          return [
            s.nome || "—",
            preview || "—",
            (s.lang || "pt").toUpperCase(),
            formatSugestaoDate(s.created_at),
            s.lida ? "Lida" : "Nova",
            "",
          ];
        })
      );
      table.querySelectorAll("tbody tr").forEach(function (tr, i) {
        var s = items[i];
        if (!s) return;
        tr.classList.add("oe-admin-row--clickable");
        tr.title = "Clique para ver detalhe";
        if (!s.lida) {
          tr.cells[4].innerHTML = '<span class="oe-admin-badge oe-admin-badge--warn">Nova</span>';
        }
        var td = tr.cells[5];
        if (!td) return;
        var viewBtn = document.createElement("button");
        viewBtn.type = "button";
        viewBtn.className = "oe-admin-btn oe-admin-btn--secondary oe-admin-btn--sm";
        viewBtn.textContent = "Ver";
        viewBtn.addEventListener("click", function (e) {
          e.stopPropagation();
          showSugestaoDetail(s);
        });
        td.appendChild(viewBtn);
        tr.addEventListener("click", function () {
          showSugestaoDetail(s);
        });
      });
      div.appendChild(table);
    } catch (e) {
      div.innerHTML = '<p class="oe-admin-empty">Erro ao carregar.</p>';
    }
  }

  function showSugestaoDetail(s) {
    var modal = document.getElementById("sugDetailModal");
    var body = document.getElementById("sugDetailBody");
    var title = document.getElementById("sugDetailTitle");
    var actions = document.getElementById("sugDetailActions");
    if (!modal || !body || !s) return;
    if (title) {
      title.textContent = "Sugestão #" + (s.id || "") + (s.nome ? " — " + s.nome : "");
    }
    var lines = [
      ["ID", s.id != null ? String(s.id) : "—"],
      ["Nome", s.nome || "Anónimo"],
      ["Idioma", (s.lang || "pt").toUpperCase()],
      ["Data", formatSugestaoDate(s.created_at)],
      ["Estado", s.lida ? "Lida" : "Nova"],
    ];
    body.innerHTML =
      '<dl class="oe-admin-dl">' +
      lines.map(function (pair) {
        return "<dt>" + pair[0] + "</dt><dd>" + escapeHtml(pair[1]) + "</dd>";
      }).join("") +
      '</dl><div class="oe-admin-message-box">' + escapeHtml(s.mensagem || "") + "</div>";
    if (actions) {
      actions.innerHTML = "";
      if (!s.lida) {
        var readBtn = document.createElement("button");
        readBtn.type = "button";
        readBtn.className = "oe-admin-btn oe-admin-btn--primary";
        readBtn.textContent = "Marcar como lida";
        readBtn.addEventListener("click", function () {
          markSugestaoRead(s.id, true);
        });
        actions.appendChild(readBtn);
      }
      var delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "oe-admin-btn oe-admin-btn--danger";
      delBtn.textContent = "Apagar";
      delBtn.addEventListener("click", function () {
        deleteSugestao(s.id, true);
      });
      actions.appendChild(delBtn);
      var closeBtn = document.createElement("button");
      closeBtn.type = "button";
      closeBtn.className = "oe-admin-btn oe-admin-btn--secondary";
      closeBtn.textContent = "Fechar";
      closeBtn.addEventListener("click", closeSugestaoDetail);
      actions.appendChild(closeBtn);
    }
    modal.classList.remove("hidden");
  }

  function closeSugestaoDetail() {
    var modal = document.getElementById("sugDetailModal");
    if (modal) modal.classList.add("hidden");
  }

  function initSugestaoModal() {
    var modal = document.getElementById("sugDetailModal");
    if (!modal) return;
    var closeBtn = document.getElementById("sugDetailClose");
    if (closeBtn) closeBtn.addEventListener("click", closeSugestaoDetail);
    modal.querySelectorAll("[data-close-sug-modal]").forEach(function (el) {
      el.addEventListener("click", closeSugestaoDetail);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && modal && !modal.classList.contains("hidden")) {
        closeSugestaoDetail();
      }
    });
  }

  async function deleteSugestao(id, fromModal) {
    if (!confirm("Apagar esta sugestão?")) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/sugestoes/" + id, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Sugestão apagada.", "success");
      if (fromModal) closeSugestaoDetail();
      loadSugestoes();
      if (global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
        global.OuviescreviAdmin.carregarDashboard();
      }
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao apagar.", "error");
    }
  }

  async function markSugestaoRead(id, fromModal) {
    try {
      var res = await fetch(apiBase() + "/api/admin/sugestoes/read", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ id: id }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Sugestão marcada como lida.", "success");
      if (fromModal) closeSugestaoDetail();
      loadSugestoes();
      if (global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
        global.OuviescreviAdmin.carregarDashboard();
      }
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao atualizar.", "error");
    }
  }

  function aiInsightStatusLabel(status) {
    if (status === "saved") return "Guardada";
    if (status === "done") return "Feita";
    if (status === "dismissed") return "Descartada";
    return "Nova";
  }

  function aiInsightPriorityBadge(priority) {
    var p = (priority || "media").toLowerCase();
    var cls = "oe-admin-badge";
    if (p === "alta") cls += " oe-admin-badge--err";
    else if (p === "baixa") cls += " oe-admin-badge--ok";
    else cls += " oe-admin-badge--warn";
    return '<span class="' + cls + '">' + escapeHtml(p) + "</span>";
  }

  function buildAiInsightPrompt(item) {
    var prompt =
      (item && item.cursor_prompt) ||
      ("Implementa no Ouviescrevi: " + ((item && item.title) || ""));
    if (item && item.detail) prompt += "\n\nContexto: " + item.detail;
    if (item && item.evidence) prompt += "\nEvidência: " + item.evidence;
    return prompt;
  }

  function buildAllAiInsightsPrompt(items, summary) {
    var list = (items || []).filter(function (item) {
      return item && (item.status === "new" || item.status === "saved");
    });
    if (!list.length) list = items || [];
    var lines = [
      "Quero implementar no Ouviescrevi (site PT de transcrição) estas sugestões AI do backoffice.",
      "Trata por ordem de prioridade (alta → média → baixa). Se alguma for grande, propõe um plano curto e começa pela mais importante.",
      "",
    ];
    if (summary) {
      lines.push("Resumo da AI: " + summary);
      lines.push("");
    }
    var order = { alta: 0, media: 1, baixa: 2 };
    list = list.slice().sort(function (a, b) {
      return (order[a.priority] != null ? order[a.priority] : 9) -
        (order[b.priority] != null ? order[b.priority] : 9);
    });
    list.forEach(function (item, i) {
      lines.push(
        (i + 1) + ". [" + (item.priority || "media").toUpperCase() + "] " +
        (item.category || "produto") + " — " + (item.title || "Sem título") +
        " (" + aiInsightStatusLabel(item.status) + ")"
      );
      lines.push(buildAiInsightPrompt(item));
      lines.push("");
    });
    return lines.join("\n").trim();
  }

  function copyTextToClipboard(text, okMsg) {
    if (!text) {
      global.OuviescreviUI.toast("Nada para copiar.", "error");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        global.OuviescreviUI.toast(okMsg || "Copiado — cola no Cursor.", "success");
      }).catch(function () {
        global.OuviescreviUI.toast("Não foi possível copiar.", "error");
      });
    } else {
      global.OuviescreviUI.toast("Clipboard indisponível.", "error");
    }
  }

  function copyAllAiInsightsToCursor() {
    if (!aiInsightsCache.length) {
      global.OuviescreviUI.toast("Não há sugestões para copiar.", "error");
      return;
    }
    copyTextToClipboard(
      buildAllAiInsightsPrompt(aiInsightsCache, aiInsightsSummaryText),
      "Todas copiadas — cola no chat do Cursor."
    );
  }

  async function loadAiInsights() {
    var div = document.getElementById("tabelaAiInsights");
    if (!div) return;
    var status = (document.getElementById("aiInsightStatusFilter") || {}).value || "";
    try {
      var qs = status ? "?status=" + encodeURIComponent(status) : "";
      var res = await fetch(apiBase() + "/api/admin/ai-insights" + qs, { headers: authHeaders() });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorMessage(data, "Erro ao carregar."));
      var items = data.items || [];
      aiInsightsCache = items;
      if (!items.length) {
        div.innerHTML = '<p class="oe-admin-empty">Sem sugestões AI. Clica em «Gerar com AI».</p>';
        return;
      }
      var html = items.map(function (item) {
        return (
          '<article class="oe-admin-ai-card" data-id="' + item.id + '">' +
          '<div class="oe-admin-ai-card__head">' +
          "<h4>" + escapeHtml(item.title || "") + "</h4>" +
          '<div class="oe-admin-ai-card__meta">' +
          aiInsightPriorityBadge(item.priority) +
          ' <span class="oe-admin-badge">' + escapeHtml(item.category || "produto") + "</span>" +
          ' <span class="oe-admin-badge">' + escapeHtml(aiInsightStatusLabel(item.status)) + "</span>" +
          "</div></div>" +
          '<p class="oe-admin-ai-card__detail">' + escapeHtml(item.detail || "") + "</p>" +
          (item.evidence
            ? '<p class="oe-admin-hint"><strong>Evidência:</strong> ' + escapeHtml(item.evidence) + "</p>"
            : "") +
          '<div class="oe-admin-ai-card__actions">' +
          '<button type="button" class="oe-admin-btn oe-admin-btn--primary oe-admin-btn--sm" data-ai-copy>Copiar para Cursor</button>' +
          (item.status !== "saved"
            ? '<button type="button" class="oe-admin-btn oe-admin-btn--secondary oe-admin-btn--sm" data-ai-status="saved">Guardar</button>'
            : "") +
          (item.status !== "done"
            ? '<button type="button" class="oe-admin-btn oe-admin-btn--secondary oe-admin-btn--sm" data-ai-status="done">Feita</button>'
            : "") +
          (item.status !== "dismissed"
            ? '<button type="button" class="oe-admin-btn oe-admin-btn--secondary oe-admin-btn--sm" data-ai-status="dismissed">Descartar</button>'
            : "") +
          '<button type="button" class="oe-admin-btn oe-admin-btn--danger oe-admin-btn--sm" data-ai-delete>Apagar</button>' +
          "</div></article>"
        );
      }).join("");
      div.innerHTML = html;
      div.querySelectorAll(".oe-admin-ai-card").forEach(function (card) {
        var id = Number(card.getAttribute("data-id"));
        var item = items.find(function (x) { return Number(x.id) === id; });
        var copyBtn = card.querySelector("[data-ai-copy]");
        if (copyBtn) {
          copyBtn.addEventListener("click", function () {
            copyTextToClipboard(buildAiInsightPrompt(item), "Prompt copiado — cola no Cursor.");
          });
        }
        card.querySelectorAll("[data-ai-status]").forEach(function (btn) {
          btn.addEventListener("click", function () {
            patchAiInsightStatus(id, btn.getAttribute("data-ai-status"));
          });
        });
        var delBtn = card.querySelector("[data-ai-delete]");
        if (delBtn) {
          delBtn.addEventListener("click", function () {
            if (!window.confirm("Apagar esta sugestão AI?")) return;
            deleteAiInsight(id);
          });
        }
      });
    } catch (e) {
      aiInsightsCache = [];
      div.innerHTML = '<p class="oe-admin-empty">' + escapeHtml(e.message || "Erro ao carregar.") + "</p>";
    }
  }

  async function generateAiInsights() {
    var btn = document.getElementById("btnGenerateAiInsights");
    var summaryEl = document.getElementById("aiInsightsSummary");
    var days = (document.getElementById("aiInsightDays") || {}).value || "7";
    if (btn) btn.disabled = true;
    try {
      var res = await fetch(
        apiBase() + "/api/admin/ai-insights/generate?days=" + encodeURIComponent(days) + "&save=true",
        { method: "POST", headers: authHeaders() }
      );
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorMessage(data, "Falha ao gerar."));
      if (summaryEl) {
        if (data.summary) {
          summaryEl.hidden = false;
          summaryEl.textContent = data.summary;
          aiInsightsSummaryText = data.summary;
        } else {
          summaryEl.hidden = true;
          aiInsightsSummaryText = "";
        }
      }
      global.OuviescreviUI.toast(
        "Geradas " + (data.count || 0) + " sugestões AI.",
        "success"
      );
      loadAiInsights();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro ao gerar.", "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async function patchAiInsightStatus(id, status) {
    try {
      var res = await fetch(apiBase() + "/api/admin/ai-insights/" + id, {
        method: "PATCH",
        headers: authHeaders(),
        body: JSON.stringify({ status: status }),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorMessage(data, "Erro ao atualizar."));
      global.OuviescreviUI.toast("Estado atualizado.", "success");
      loadAiInsights();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro.", "error");
    }
  }

  async function deleteAiInsight(id) {
    try {
      var res = await fetch(apiBase() + "/api/admin/ai-insights/" + id, {
        method: "DELETE",
        headers: authHeaders(),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(apiErrorMessage(data, "Erro ao apagar."));
      global.OuviescreviUI.toast("Sugestão apagada.", "success");
      loadAiInsights();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro.", "error");
    }
  }

  async function updateUserRole(userId, role, username) {
    try {
      var res = await fetch(apiBase() + "/api/admin/users/" + userId, {
        method: "PATCH",
        headers: authHeaders(),
        body: JSON.stringify({ role: role }),
      });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(data.detail || "Erro ao atualizar papel.");
      global.OuviescreviUI.toast(
        "Papel de «" + username + "» atualizado para " + roleLabel(role) + ".",
        "success"
      );
      var currentUser = sessionStorage.getItem("ouviescrevi_admin_username") || "";
      if (username === currentUser) {
        sessionStorage.setItem("ouviescrevi_admin_role", role);
        if (global.OuviescreviAdmin && global.OuviescreviAdmin.applyRoleUI) {
          global.OuviescreviAdmin.applyRoleUI();
        }
      }
      loadSystem();
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Erro ao atualizar papel.", "error");
      loadSystem();
    }
  }

  async function deleteUser(userId, username) {
    if (!confirm("Remover utilizador «" + username + "»?")) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/users/" + userId, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Utilizador removido.", "success");
      loadSystem();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao remover.", "error");
    }
  }

  function statusClass(value) {
    var v = String(value || "").toLowerCase();
    if (v === "ok") return "oe-admin-health-card--ok";
    if (v.indexOf("erro") !== -1) return "oe-admin-health-card--err";
    if (v === "warn" || v === "unknown") return "oe-admin-health-card--warn";
    return "oe-admin-health-card--warn";
  }

  function formatBytes(bytes) {
    if (bytes == null || isNaN(bytes)) return "—";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(2) + " MB";
  }

  function formatDateTime(iso) {
    if (!iso) return "—";
    return String(iso).replace("T", " ").replace("Z", "").slice(0, 19);
  }

  function shortDbHost(path) {
    if (!path) return "—";
    if (path.indexOf("libsql://") === 0) {
      return path.replace("libsql://", "").split("/")[0];
    }
    return path;
  }

  function renderHealthCards(h) {
    var grid = document.getElementById("systemHealthCards");
    if (!grid) return;
    var dbLabel = h.database_backend === "turso" ? "Turso Cloud" : "SQLite local";
    var persistLabel = h.database_persistent ? "Persistente" : "Efémera";
    var persistStatus = h.database_persistent ? "ok" : "warn";
    var apiStatus = h.api || "unknown";
    var openaiStatus = h.openai || "unknown";
    var cards = [
      { title: "API", value: apiStatus === "ok" ? "Operacional" : apiStatus, status: apiStatus },
      { title: "Base de dados", value: dbLabel, status: h.database || "unknown" },
      { title: "OpenAI", value: openaiStatus === "ok" ? "Ligada" : openaiStatus, status: openaiStatus },
      { title: "Persistência", value: persistLabel, status: persistStatus },
    ];
    grid.innerHTML = cards.map(function (c) {
      return (
        '<div class="oe-admin-health-card ' + statusClass(c.status) + '">' +
        '<div class="oe-admin-health-card__title">' + c.title + "</div>" +
        '<div class="oe-admin-health-card__value">' + c.value + "</div>" +
        "</div>"
      );
    }).join("");
  }

  function renderSystemDetails(h) {
    var health = document.getElementById("systemHealth");
    if (!health) return;

    health.innerHTML =
      '<div class="oe-admin-kv-grid">' +
      kvRow("Ambiente", h.app_env || "—") +
      kvRow("API pública", h.public_api_base || "—") +
      kvRow("Backend BD", h.database_backend === "turso" ? "Turso (libSQL)" : "Ficheiro SQLite") +
      kvRow(
        "Turso URL (env)",
        h.turso_url_set
          ? (h.turso_url_valid ? "✓ definida (libsql://…)" : "⚠ definida mas formato inválido — usa libsql://…")
          : "✗ não definida no Render"
      ) +
      kvRow(
        "Turso token (env)",
        h.turso_token_set ? "✓ definido" : "✗ não definido no Render"
      ) +
      kvRow("Host / ficheiro", '<code class="oe-admin-code">' + shortDbHost(h.database_path) + "</code>") +
      kvRow("Latência BD", h.database_latency_ms != null ? h.database_latency_ms + " ms" : "—") +
      kvRow("Tamanho ficheiro", formatBytes(h.database_bytes)) +
      kvRow("Disco livre (servidor)", h.disk_free_mb != null ? h.disk_free_mb + " MB" : "—") +
      kvRow("Última transcrição", formatDateTime(h.last_transcription_at)) +
      kvRow("Verificado às", formatDateTime(h.checked_at)) +
      "</div>" +
      '<div class="oe-admin-alert ' + (h.database_persistent ? "oe-admin-alert--ok" : "oe-admin-alert--warn") + '" style="margin-top:16px">' +
      "<strong>" + (h.database_persistent ? "✓ Dados persistentes" : "⚠ Risco de perda de dados") + "</strong><br>" +
      (h.persistence_note || "") +
      "</div>" +
      (!h.database_persistent && !h.turso_url_set && !h.turso_token_set
        ? '<div class="oe-admin-alert oe-admin-alert--warn" style="margin-top:12px">' +
          "<strong>Como ativar Turso no Render</strong><br>" +
          "1. Abre <a href=\"https://dashboard.render.com\" target=\"_blank\" rel=\"noopener\">dashboard.render.com</a> → serviço <code class=\"oe-admin-code\">api-ouviescrevi</code><br>" +
          "2. Menu <strong>Environment</strong> → adiciona <code class=\"oe-admin-code\">TURSO_DATABASE_URL</code> (libsql://…) e <code class=\"oe-admin-code\">TURSO_AUTH_TOKEN</code><br>" +
          "3. Clica <strong>Save, rebuild, and deploy</strong> (obrigatório — só guardar não chega)<br>" +
          "4. Volta aqui e clica <strong>Atualizar</strong> — deve aparecer Turso Cloud + Persistente" +
          "</div>"
        : "") +
      '<div class="oe-admin-alert ' + (h.cms_locales_ready ? "oe-admin-alert--ok" : "oe-admin-alert--warn") + '" style="margin-top:12px">' +
      "<strong>CMS multi-idioma (API)</strong><br>" +
      (h.cms_locales_ready ? "✓ " : "⚠ ") + (h.cms_locales_note || "—") +
      (h.cms_locale_pages != null
        ? "<br><small>Páginas locale: " + h.cms_locale_pages + " · SEO locale: " + (h.cms_locale_seo_pages || 0) +
          " · Chaves: " + (h.cms_locale_keys || 0) + "</small>"
        : "") +
      "</div>";

    var statsDiv = document.getElementById("databaseStats");
    if (statsDiv) {
      var counts = h.table_counts || {};
      var labels = {
        transcricoes: "Transcrições",
        visitas: "Visitas",
        site_content: "Blocos CMS",
        sugestoes: "Sugestões",
        ai_insights: "Sugestões AI",
        admin_users: "Utilizadores",
        audit_log: "Auditoria",
      };
      var rows = Object.keys(labels).map(function (key) {
        return [labels[key], counts[key] != null ? String(counts[key]) : "—"];
      });
      statsDiv.innerHTML = "";
      statsDiv.appendChild(buildTable(["Tabela", "Registos"], rows));
    }
  }

  function kvRow(label, value) {
    return (
      '<div class="oe-admin-kv">' +
      '<span class="oe-admin-kv__label">' + label + "</span>" +
      '<span class="oe-admin-kv__value">' + value + "</span>" +
      "</div>"
    );
  }

  function setField(id, value) {
    var el = document.getElementById(id);
    if (el) el.value = value != null ? value : "";
  }

  function setChecked(id, checked) {
    var el = document.getElementById(id);
    if (el) el.checked = !!checked;
  }

  function serverLogQueryParams() {
    var filter = document.getElementById("serverLogFilter");
    var val = filter ? filter.value : "";
    var params = new URLSearchParams({ limit: "400" });
    if (val === "ERROR") {
      params.set("level", "ERROR");
    } else if (val) {
      params.set("q", val);
    }
    return "?" + params.toString();
  }

  function renderProcessingJobs(items) {
    var box = document.getElementById("processingJobsBox");
    var tableHost = document.getElementById("processingJobsTable");
    var statJobs = document.getElementById("statLogsJobs");
    var badge = document.getElementById("navBadgeLogs");
    var list = items || [];
    var active = list.filter(function (j) {
      return j.status === "processing";
    });
    if (statJobs) statJobs.textContent = String(active.length);
    if (badge) {
      if (active.length > 0) {
        badge.textContent = String(active.length);
        badge.classList.remove("hidden");
      } else {
        badge.classList.add("hidden");
      }
    }
    if (box) {
      if (!active.length) {
        box.classList.add("hidden");
        box.innerHTML = "";
      } else {
        box.classList.remove("hidden");
        box.innerHTML =
          "<p class='oe-admin-jobs-box__title'>Em curso agora</p>" +
          active
            .map(function (j) {
              var pct = j.progress != null ? j.progress + "%" : "—";
              var kind = j.kind === "video-subs" ? "Legendas" : "Transcrição";
              var msg = j.message || j.status || "";
              var file = j.filename ? " · " + j.filename : "";
              return (
                "<div class='oe-admin-jobs-box__item'><strong>" +
                kind +
                " " +
                pct +
                "</strong> — " +
                msg +
                file +
                "</div>"
              );
            })
            .join("");
      }
    }
    if (tableHost) {
      if (!list.length) {
        tableHost.innerHTML = '<p class="oe-admin-empty">Sem tarefas recentes na memória do servidor.</p>';
      } else {
        tableHost.innerHTML = "";
        tableHost.appendChild(
          buildTable(
            ["Tipo", "Estado", "Progresso", "Ficheiro", "Mensagem", "Job"],
            list.slice(0, 40).map(function (j) {
              return [
                j.kind === "video-subs" ? "Legendas" : "Transcrição",
                j.status || "—",
                j.progress != null ? j.progress + "%" : "—",
                (j.filename || "—").toString().slice(0, 40),
                (j.message || "—").toString().slice(0, 80),
                (j.job_id || "—").toString().slice(0, 10),
              ];
            })
          )
        );
      }
    }
  }

  async function loadAuditAndErrors() {
    var role = sessionStorage.getItem("ouviescrevi_admin_role") || "admin";
    var auditDiv = document.getElementById("auditLog");
    var errDiv = document.getElementById("errorLog");
    var statAudit = document.getElementById("statLogsAudit");
    var statErr = document.getElementById("statLogsApiErrors");
    if (role !== "admin") {
      if (auditDiv) auditDiv.innerHTML = '<p class="oe-admin-empty">Apenas administradores.</p>';
      if (errDiv) errDiv.innerHTML = '<p class="oe-admin-empty">Apenas administradores.</p>';
      return;
    }
    try {
      var ares = await fetch(apiBase() + "/api/admin/audit?limit=40", { headers: authHeaders() });
      var adata = await ares.json();
      var logs = adata.items || [];
      if (statAudit) statAudit.textContent = String(logs.length);
      if (auditDiv) {
        auditDiv.innerHTML = "";
        if (!logs.length) {
          auditDiv.innerHTML = '<p class="oe-admin-empty">Sem eventos de auditoria.</p>';
        } else {
          auditDiv.appendChild(
            buildTable(
              ["Quem", "Ação", "Detalhe", "Quando"],
              logs.map(function (l) {
                return [
                  l.actor || "—",
                  l.action || "—",
                  (l.detail || "—").toString().slice(0, 60),
                  (l.created_at || "").replace("T", " ").slice(0, 19),
                ];
              })
            )
          );
        }
      }
    } catch (e) {
      if (auditDiv) auditDiv.innerHTML = '<p class="oe-admin-empty">Erro ao carregar auditoria.</p>';
    }
    try {
      var eres = await fetch(apiBase() + "/api/admin/errors?limit=40", { headers: authHeaders() });
      var edata = await eres.json();
      var errors = edata.items || [];
      if (statErr) statErr.textContent = String(errors.length);
      if (errDiv) {
        errDiv.innerHTML = "";
        if (!errors.length) {
          errDiv.innerHTML = '<p class="oe-admin-empty">Sem erros recentes.</p>';
        } else {
          errDiv.appendChild(
            buildTable(
              ["Path", "Status", "Mensagem", "Quando"],
              errors.map(function (e) {
                return [
                  (e.path || "—").toString().slice(0, 40),
                  String(e.status_code || "—"),
                  (e.message || e.detail || "—").toString().slice(0, 80),
                  (e.created_at || "").replace("T", " ").slice(0, 19),
                ];
              })
            )
          );
        }
      }
    } catch (e2) {
      if (errDiv) errDiv.innerHTML = '<p class="oe-admin-empty">Erro ao carregar erros API.</p>';
    }
  }

  async function loadServerLogs() {
    var view = document.getElementById("serverLogView");
    if (!view) return;
    if ((sessionStorage.getItem("ouviescrevi_admin_role") || "admin") !== "admin") {
      view.textContent = "Apenas administradores podem ver os logs do servidor.";
      return;
    }
    try {
      var res = await fetch(apiBase() + "/api/admin/server-logs" + serverLogQueryParams(), {
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      var data = await res.json();
      lastServerLogText = data.text || "(sem registos)";
      view.textContent = lastServerLogText;
      view.scrollTop = view.scrollHeight;
    } catch (e) {
      view.textContent = "Erro ao carregar logs. Clica Atualizar ou faz logout/login.";
    }
    try {
      var jres = await fetch(apiBase() + "/api/admin/processing-jobs", { headers: authHeaders() });
      if (jres.ok) {
        var jdata = await jres.json();
        renderProcessingJobs(jdata.items || []);
      }
    } catch (e2) {
      /* opcional */
    }
    var updated = document.getElementById("statLogsUpdated");
    if (updated) {
      var now = new Date();
      updated.textContent =
        String(now.getHours()).padStart(2, "0") +
        ":" +
        String(now.getMinutes()).padStart(2, "0") +
        ":" +
        String(now.getSeconds()).padStart(2, "0");
    }
  }

  async function loadLogs() {
    await loadServerLogs();
    await loadAuditAndErrors();
    scheduleServerLogRefresh();
  }

  function scheduleServerLogRefresh() {
    if (serverLogTimer) {
      clearInterval(serverLogTimer);
      serverLogTimer = null;
    }
    var auto = document.getElementById("serverLogAutoRefresh");
    if (auto && auto.checked) {
      serverLogTimer = setInterval(loadServerLogs, 10000);
    }
  }

  function copyServerLogs() {
    if (!lastServerLogText) {
      global.OuviescreviUI.toast("Nada para copiar.", "error");
      return;
    }
    copyTextToClipboard(lastServerLogText).then(
      function () {
        global.OuviescreviUI.toast("Logs copiados.", "success");
      },
      function () {
        downloadServerLogs();
        global.OuviescreviUI.toast("Clipboard indisponível — logs descarregados.", "success");
      }
    );
  }

  function downloadServerLogs() {
    var blob = new Blob([lastServerLogText || ""], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = "ouviescrevi-logs-" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  async function loadSystem() {
    var health = document.getElementById("systemHealth");
    var grid = document.getElementById("systemHealthCards");
    if (grid) grid.innerHTML = '<p class="oe-admin-empty">A verificar serviços...</p>';
    try {
      var hres = await fetch(apiBase() + "/api/admin/health", { headers: authHeaders() });
      if (hres.status === 403) {
        if (grid) grid.innerHTML = "";
        if (health) {
          health.innerHTML =
            '<div class="oe-admin-alert oe-admin-alert--warn">' +
            "<strong>Sessão expirada</strong><br>A base Turso é nova — faz <strong>Sair</strong> e volta a entrar com a palavra-chave." +
            "</div>";
        }
        return;
      }
      if (!hres.ok) throw new Error("health " + hres.status);
      var h = await hres.json();
      renderHealthCards(h);
      renderSystemDetails(h);
    } catch (e) {
      if (grid) grid.innerHTML = "";
      if (health) {
        health.innerHTML =
          '<p class="oe-admin-empty">Erro ao carregar estado do sistema. Clica <strong>Atualizar</strong> ou faz logout/login.</p>';
      }
      var statsDiv = document.getElementById("databaseStats");
      if (statsDiv) statsDiv.innerHTML = '<p class="oe-admin-empty">—</p>';
      return;
    }
    try {
      var cres = await fetch(apiBase() + "/api/admin/config", { headers: authHeaders() });
      if (cres.ok) {
        var cdata = await cres.json();
        var cfg = cdata.config || {};
        setField("cfgMaxMb", cfg.max_file_size_mb || "");
        setField("cfgQuotaAnon", cfg.quota_anonymous_daily || "3");
        setField("cfgQuotaReg", cfg.quota_registered_daily || "20");
        setField("cfgWhisperCost", cfg.whisper_cost_per_minute_usd || "0.006");
        setField("cfgCfZone", cfg.cloudflare_zone_id || "");
        setField("cfgCfToken", "");
      }
      var bres = await fetch(apiBase() + "/api/admin/banners", { headers: authHeaders() });
      if (bres.ok) {
        var bdata = await bres.json();
        var banner = (bdata.items || [])[0];
        if (banner) {
          setField("bannerTexto", banner.texto || "");
          setField("bannerLink", banner.link || "");
          setChecked("bannerAtivo", !!banner.ativo);
        }
      }
      var ul = document.getElementById("usersList");
      if (ul && (sessionStorage.getItem("ouviescrevi_admin_role") || "admin") === "admin") {
        try {
          var ures = await fetch(apiBase() + "/api/admin/users", { headers: authHeaders() });
          var udata = await ures.json();
          var users = udata.items || [];
          if (!users.length) {
            ul.innerHTML = "<p class='oe-admin-empty'>Só o admin por defeito.</p>";
          } else {
            ul.innerHTML = "";
            var utable = buildTable(
              ["Utilizador", "Papel", "Criado", "Ações"],
              users.map(function (u) {
                return [
                  u.username,
                  u.role,
                  (u.created_at || "").replace("T", " ").slice(0, 10),
                  "",
                ];
              })
            );
            var currentUser = sessionStorage.getItem("ouviescrevi_admin_username") || "";
            utable.querySelectorAll("tbody tr").forEach(function (tr, i) {
              var u = users[i];
              if (!u) return;
              var roleTd = tr.cells[1];
              if (roleTd) {
                roleTd.innerHTML = "";
                var roleSelect = document.createElement("select");
                roleSelect.className = "oe-admin-select-inline oe-admin-role-select";
                ["viewer", "editor", "admin"].forEach(function (r) {
                  var opt = document.createElement("option");
                  opt.value = r;
                  opt.textContent = roleLabel(r);
                  if (u.role === r) opt.selected = true;
                  roleSelect.appendChild(opt);
                });
                if (u.username === currentUser) {
                  roleSelect.disabled = true;
                  roleSelect.title = "Não podes alterar o teu próprio papel aqui.";
                } else {
                  roleSelect.addEventListener("change", function () {
                    updateUserRole(u.id, roleSelect.value, u.username);
                  });
                }
                roleTd.appendChild(roleSelect);
              }
              var td = tr.cells[3];
              if (!td) return;
              var btn = document.createElement("button");
              btn.type = "button";
              btn.className = "oe-admin-btn oe-admin-btn--danger oe-admin-btn--sm";
              btn.textContent = "Remover";
              if (u.username === currentUser) {
                btn.disabled = true;
                btn.title = "Não podes remover a tua própria conta.";
              } else {
                btn.addEventListener("click", function () {
                  deleteUser(u.id, u.username);
                });
              }
              td.appendChild(btn);
            });
            ul.appendChild(utable);
          }
        } catch (e) {
          ul.innerHTML = "<p class='oe-admin-empty'>Sem permissão.</p>";
        }
      }
    } catch (e) {
      /* Config/utilizadores são opcionais — o painel de saúde já foi renderizado. */
    }
  }

  async function saveConfig(e) {
    e.preventDefault();
    var updates = {
      max_file_size_mb: document.getElementById("cfgMaxMb").value,
      quota_anonymous_daily: document.getElementById("cfgQuotaAnon").value,
      quota_registered_daily: document.getElementById("cfgQuotaReg").value,
      whisper_cost_per_minute_usd: document.getElementById("cfgWhisperCost").value,
      cloudflare_zone_id: document.getElementById("cfgCfZone").value,
    };
    var tok = document.getElementById("cfgCfToken").value;
    if (tok) updates.cloudflare_api_token = tok;
    try {
      var res = await fetch(apiBase() + "/api/admin/config", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ updates: updates }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Configurações guardadas.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  async function saveBanner(e) {
    e.preventDefault();
    try {
      var res = await fetch(apiBase() + "/api/admin/banners", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({
          texto: document.getElementById("bannerTexto").value,
          link: document.getElementById("bannerLink").value,
          ativo: document.getElementById("bannerAtivo").checked,
        }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Banner guardado.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar banner.", "error");
    }
  }

  async function addUser(e) {
    if (e && e.preventDefault) e.preventDefault();
    var form = document.getElementById("userForm");
    if (!form) return;
    var submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn && submitBtn.disabled) return;
    if (submitBtn) submitBtn.disabled = true;
    var fd = new FormData(form);
    try {
      var res = await fetch(apiBase() + "/api/admin/users", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({
          username: fd.get("username"),
          password: fd.get("password"),
          role: fd.get("role"),
        }),
      });
      if (!res.ok) {
        var errBody = await res.json().catch(function () { return {}; });
        throw new Error(apiErrorMessage(errBody, "Erro ao criar utilizador."));
      }
      form.reset();
      global.OuviescreviUI.toast("Utilizador criado.", "success");
      loadSystem();
    } catch (err) {
      global.OuviescreviUI.toast(err.message || "Erro ao criar utilizador.", "error");
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  }

  function onTab(tab) {
    if (tab === "dashboard" && global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
      global.OuviescreviAdmin.carregarDashboard();
    }
    if (tab === "sugestoes") loadSugestoes();
    if (tab === "ai-insights") loadAiInsights();
    if (tab === "emails") loadEmails();
    if (tab === "planos" && global.OuviescreviBillingAdmin) global.OuviescreviBillingAdmin.loadBilling();
    if (tab === "sistema") {
      loadSystem();
      if (serverLogTimer) {
        clearInterval(serverLogTimer);
        serverLogTimer = null;
      }
    } else if (tab === "logs") {
      loadLogs();
    } else if (serverLogTimer) {
      clearInterval(serverLogTimer);
      serverLogTimer = null;
    }
  }

  function init() {
    initExports();
    initSugestaoModal();
    var seoForm = document.getElementById("seoForm");
    if (seoForm) seoForm.addEventListener("submit", saveSeo);
    var btnMaint = document.getElementById("btnSaveMaintenance");
    if (btnMaint) btnMaint.addEventListener("click", saveMaintenanceMessage);
    var cfgForm = document.getElementById("configForm");
    if (cfgForm) cfgForm.addEventListener("submit", saveConfig);
    var bannerForm = document.getElementById("bannerForm");
    if (bannerForm) bannerForm.addEventListener("submit", saveBanner);
    var userForm = document.getElementById("userForm");
    var btnAddUser = document.getElementById("btnAddUser");
    if (btnAddUser) btnAddUser.addEventListener("click", addUser);
    if (userForm) userForm.addEventListener("submit", addUser);
    var sugUnread = document.getElementById("sugUnreadOnly");
    if (sugUnread) sugUnread.addEventListener("change", loadSugestoes);
    var sugLang = document.getElementById("sugLangFilter");
    if (sugLang) sugLang.addEventListener("change", loadSugestoes);
    var btnGenAi = document.getElementById("btnGenerateAiInsights");
    if (btnGenAi) btnGenAi.addEventListener("click", generateAiInsights);
    var btnRefreshAi = document.getElementById("btnRefreshAiInsights");
    if (btnRefreshAi) btnRefreshAi.addEventListener("click", loadAiInsights);
    var btnCopyAllAi = document.getElementById("btnCopyAllAiInsights");
    if (btnCopyAllAi) btnCopyAllAi.addEventListener("click", copyAllAiInsightsToCursor);
    var aiStatusFilter = document.getElementById("aiInsightStatusFilter");
    if (aiStatusFilter) aiStatusFilter.addEventListener("change", loadAiInsights);
    var emailForm = document.getElementById("emailConfigForm");
    if (emailForm) emailForm.addEventListener("submit", saveEmailConfig);
    var btnRefreshEmails = document.getElementById("btnRefreshEmails");
    if (btnRefreshEmails) btnRefreshEmails.addEventListener("click", loadEmails);
    var btnTestActivity = document.getElementById("btnTestActivityEmail");
    if (btnTestActivity) btnTestActivity.addEventListener("click", testActivityEmail);
    var btnTestAlertEmails = document.getElementById("btnTestAlertEmailEmails");
    if (btnTestAlertEmails) btnTestAlertEmails.addEventListener("click", testAlertEmail);
    var btnWeekly = document.getElementById("btnSendWeeklyTip");
    if (btnWeekly) btnWeekly.addEventListener("click", function () { sendLifecycleEmail("weekly_tip"); });
    var btnNudge = document.getElementById("btnSendQuotaNudge");
    if (btnNudge) btnNudge.addEventListener("click", function () { sendLifecycleEmail("quota_nudge"); });
    var btnSys = document.getElementById("btnRefreshSystem");
    if (btnSys) btnSys.addEventListener("click", loadSystem);
    var btnLogRefresh = document.getElementById("btnRefreshServerLogs");
    if (btnLogRefresh) btnLogRefresh.addEventListener("click", loadServerLogs);
    var btnLogCopy = document.getElementById("btnCopyServerLogs");
    if (btnLogCopy) btnLogCopy.addEventListener("click", copyServerLogs);
    var btnLogDl = document.getElementById("btnDownloadServerLogs");
    if (btnLogDl) btnLogDl.addEventListener("click", downloadServerLogs);
    var logFilter = document.getElementById("serverLogFilter");
    if (logFilter) logFilter.addEventListener("change", loadServerLogs);
    var logAuto = document.getElementById("serverLogAutoRefresh");
    if (logAuto) {
      logAuto.addEventListener("change", scheduleServerLogRefresh);
    }
    var btnJobs = document.getElementById("btnRefreshLogsJobs");
    if (btnJobs) btnJobs.addEventListener("click", loadServerLogs);
    var btnErr = document.getElementById("btnRefreshErrorLog");
    if (btnErr) btnErr.addEventListener("click", loadAuditAndErrors);
    var btnAud = document.getElementById("btnRefreshAuditLog");
    if (btnAud) btnAud.addEventListener("click", loadAuditAndErrors);
    var btnOpenLogs = document.getElementById("btnOpenLogsFromSystem");
    if (btnOpenLogs) {
      btnOpenLogs.addEventListener("click", function () {
        var nav = document.querySelector('.oe-admin-nav button[data-tab="logs"]');
        if (nav) nav.click();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  function resizeCloudflareChart() {
    if (chartCloudflare && typeof chartCloudflare.resize === "function") {
      chartCloudflare.resize();
    }
  }

  global.OuviescreviAdminExt = {
    setupSeo: setupSeo,
    renderReferrersAndDevices: renderReferrersAndDevices,
    destroyCharts: destroyCharts,
    resizeCloudflareChart: resizeCloudflareChart,
    onTab: onTab,
    loadSugestoes: loadSugestoes,
    loadAiInsights: loadAiInsights,
    loadSystem: loadSystem,
    loadLogs: loadLogs,
    loadEmails: loadEmails,
    initVisitReportExport: initVisitReportExport,
    exportVisitReport: runVisitReportExport,
  };
})(window);
