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

  function initExports() {
    var base = apiBase();
    var token = global.OuviescreviAPI.getAdminToken();
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
            var a = document.createElement("a");
            a.href = URL.createObjectURL(blob);
            a.download = name;
            a.click();
          })
          .catch(function () {
            global.OuviescreviUI.toast("Erro ao exportar.", "error");
          });
      };
    });
  }

  function renderReferrersAndDevices(data) {
    var refDiv = document.getElementById("topReferrers");
    var devDiv = document.getElementById("deviceStats");
    if (refDiv) {
      refDiv.innerHTML = "";
      var refs = data.top_referrers || [];
      if (!refs.length) refDiv.innerHTML = '<p class="oe-admin-empty">Sem referrers.</p>';
      else refDiv.appendChild(buildTable(["Origem", "Visitas"], refs.map(function (r) { return [r.referrer, String(r.total)]; })));
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
    var sugEl = document.getElementById("statSugestoesNovas");
    if (sugEl) sugEl.textContent = String(unread);
    var card = document.getElementById("cardSugestoes");
    if (card) card.classList.toggle("oe-admin-card--pulse", unread > 0);
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
    } catch (e) {
      var grid = document.getElementById("emailStatusCards");
      if (grid) grid.innerHTML = "<p class='oe-admin-empty'>Erro ao carregar estado de email.</p>";
    }
    await loadEmailLogs();
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
      if (!items.length) {
        div.innerHTML = '<p class="oe-admin-empty">Sem sugestões.</p>';
        return;
      }
      div.innerHTML = "";
      var table = buildTable(
        ["Nome", "Mensagem", "Idioma", "Data", "Estado", "Ações"],
        items.map(function (s) {
          return [
            s.nome || "—",
            (s.mensagem || "").slice(0, 100),
            s.lang || "pt",
            (s.created_at || "").replace("T", " ").replace("Z", "").slice(0, 19),
            s.lida ? "Lida" : "Nova",
            "",
          ];
        })
      );
      table.querySelectorAll("tbody tr").forEach(function (tr, i) {
        var s = items[i];
        if (!s) return;
        if (!s.lida) {
          tr.cells[4].innerHTML = '<span class="oe-admin-badge oe-admin-badge--warn">Nova</span>';
        }
        if (s.mensagem && s.mensagem.length > 100) tr.cells[1].title = s.mensagem;
        var td = tr.cells[5];
        if (!td) return;
        if (!s.lida) {
          var readBtn = document.createElement("button");
          readBtn.type = "button";
          readBtn.className = "oe-admin-btn oe-admin-btn--secondary oe-admin-btn--sm";
          readBtn.textContent = "Marcar lida";
          readBtn.addEventListener("click", function () { markSugestaoRead(s.id); });
          td.appendChild(readBtn);
        }
        var delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "oe-admin-btn oe-admin-btn--danger oe-admin-btn--sm";
        delBtn.textContent = "Apagar";
        delBtn.style.marginLeft = "6px";
        delBtn.addEventListener("click", function () { deleteSugestao(s.id); });
        td.appendChild(delBtn);
      });
      div.appendChild(table);
    } catch (e) {
      div.innerHTML = '<p class="oe-admin-empty">Erro ao carregar.</p>';
    }
  }

  async function deleteSugestao(id) {
    if (!confirm("Apagar esta sugestão?")) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/sugestoes/" + id, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Sugestão apagada.", "success");
      loadSugestoes();
      if (global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
        global.OuviescreviAdmin.carregarDashboard();
      }
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao apagar.", "error");
    }
  }

  async function markSugestaoRead(id) {
    try {
      var res = await fetch(apiBase() + "/api/admin/sugestoes/read", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ id: id }),
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Sugestão marcada como lida.", "success");
      loadSugestoes();
      if (global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
        global.OuviescreviAdmin.carregarDashboard();
      }
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao atualizar.", "error");
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
    if (!box) return;
    var active = (items || []).filter(function (j) {
      return j.status === "processing";
    });
    if (!active.length) {
      box.classList.add("hidden");
      box.innerHTML = "";
      return;
    }
    box.classList.remove("hidden");
    box.innerHTML =
      "<p class='oe-admin-jobs-box__title'>Tarefas em curso</p>" +
      active
        .map(function (j) {
          var pct = j.progress != null ? j.progress + "%" : "—";
          var msg = j.message || j.status || "";
          var file = j.filename ? " · " + j.filename : "";
          return (
            "<div class='oe-admin-jobs-box__item'><strong>" +
            pct +
            "</strong> " +
            msg +
            file +
            "</div>"
          );
        })
        .join("");
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
    navigator.clipboard.writeText(lastServerLogText).then(
      function () {
        global.OuviescreviUI.toast("Logs copiados.", "success");
      },
      function () {
        global.OuviescreviUI.toast("Não foi possível copiar.", "error");
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
              ["Utilizador", "Papel", "Criado", ""],
              users.map(function (u) {
                return [
                  u.username,
                  u.role,
                  (u.created_at || "").replace("T", " ").slice(0, 10),
                  "del",
                ];
              })
            );
            utable.querySelectorAll("tbody tr").forEach(function (tr, i) {
              var u = users[i];
              var td = tr.cells[3];
              if (!td || !u) return;
              var btn = document.createElement("button");
              btn.type = "button";
              btn.className = "oe-admin-btn oe-admin-btn--danger oe-admin-btn--sm";
              btn.textContent = "Remover";
              btn.addEventListener("click", function () {
                deleteUser(u.id, u.username);
              });
              td.appendChild(btn);
            });
            ul.appendChild(utable);
          }
        } catch (e) {
          ul.innerHTML = "<p class='oe-admin-empty'>Sem permissão.</p>";
        }
      }
      if ((sessionStorage.getItem("ouviescrevi_admin_role") || "admin") === "admin") {
        var ares = await fetch(apiBase() + "/api/admin/audit?limit=15", { headers: authHeaders() });
        var adata = await ares.json();
        var auditDiv = document.getElementById("auditLog");
        if (auditDiv) {
          auditDiv.innerHTML = "";
          var logs = adata.items || [];
          auditDiv.appendChild(
            buildTable(
              ["Quem", "Ação", "Quando"],
              logs.map(function (l) {
                return [l.actor, l.action, (l.created_at || "").replace("T", " ")];
              })
            )
          );
        }
      } else {
        var auditDivOnly = document.getElementById("auditLog");
        if (auditDivOnly) auditDivOnly.innerHTML = '<p class="oe-admin-empty">Apenas administradores.</p>';
      }
      var eres = await fetch(apiBase() + "/api/admin/errors?limit=15", { headers: authHeaders() });
      var edata = await eres.json();
      var errDiv = document.getElementById("errorLog");
      if (errDiv) {
        errDiv.innerHTML = "";
        errDiv.appendChild(
          buildTable(
            ["Path", "Status", "Quando"],
            (edata.items || []).map(function (e) {
              return [e.path, String(e.status_code), (e.created_at || "").replace("T", " ")];
            })
          )
        );
      }
    } catch (e) {
      /* Config/utilizadores/auditoria são opcionais — o painel de saúde já foi renderizado. */
    }
    loadServerLogs();
    scheduleServerLogRefresh();
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
    e.preventDefault();
    var fd = new FormData(e.target);
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
      if (!res.ok) throw new Error();
      e.target.reset();
      global.OuviescreviUI.toast("Utilizador criado.", "success");
      loadSystem();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao criar utilizador.", "error");
    }
  }

  function onTab(tab) {
    if (tab === "dashboard" && global.OuviescreviAdmin && global.OuviescreviAdmin.carregarDashboard) {
      global.OuviescreviAdmin.carregarDashboard();
    }
    if (tab === "sugestoes") loadSugestoes();
    if (tab === "emails") loadEmails();
    if (tab === "sistema") {
      loadSystem();
      scheduleServerLogRefresh();
    } else if (serverLogTimer) {
      clearInterval(serverLogTimer);
      serverLogTimer = null;
    }
  }

  function init() {
    initExports();
    var seoForm = document.getElementById("seoForm");
    if (seoForm) seoForm.addEventListener("submit", saveSeo);
    var btnMaint = document.getElementById("btnSaveMaintenance");
    if (btnMaint) btnMaint.addEventListener("click", saveMaintenanceMessage);
    var cfgForm = document.getElementById("configForm");
    if (cfgForm) cfgForm.addEventListener("submit", saveConfig);
    var bannerForm = document.getElementById("bannerForm");
    if (bannerForm) bannerForm.addEventListener("submit", saveBanner);
    var userForm = document.getElementById("userForm");
    if (userForm) userForm.addEventListener("submit", addUser);
    var sugUnread = document.getElementById("sugUnreadOnly");
    if (sugUnread) sugUnread.addEventListener("change", loadSugestoes);
    var sugLang = document.getElementById("sugLangFilter");
    if (sugLang) sugLang.addEventListener("change", loadSugestoes);
    var emailForm = document.getElementById("emailConfigForm");
    if (emailForm) emailForm.addEventListener("submit", saveEmailConfig);
    var btnRefreshEmails = document.getElementById("btnRefreshEmails");
    if (btnRefreshEmails) btnRefreshEmails.addEventListener("click", loadEmails);
    var btnTestActivity = document.getElementById("btnTestActivityEmail");
    if (btnTestActivity) btnTestActivity.addEventListener("click", testActivityEmail);
    var btnTestAlertEmails = document.getElementById("btnTestAlertEmailEmails");
    if (btnTestAlertEmails) btnTestAlertEmails.addEventListener("click", testAlertEmail);
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  global.OuviescreviAdminExt = {
    setupSeo: setupSeo,
    renderReferrersAndDevices: renderReferrersAndDevices,
    destroyCharts: destroyCharts,
    onTab: onTab,
    loadSugestoes: loadSugestoes,
    loadSystem: loadSystem,
    loadEmails: loadEmails,
  };
})(window);
