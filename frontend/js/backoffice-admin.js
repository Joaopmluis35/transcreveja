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
  var seoContent = {};

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
    if (cEl) cEl.textContent = "$" + (cost.custo_estimado_usd || 0);
    var msg = document.getElementById("maintenanceMessage");
    if (msg && data.maintenance_message) msg.value = data.maintenance_message.replace(/<[^>]+>/g, function (t) {
      return t === "<br>" || t === "<br/>" ? "\n" : "";
    }).replace(/<p>/gi, "").replace(/<\/p>/gi, "\n").replace(/<[^>]+>/g, "").trim();
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

  function setupSeo(pages, content) {
    seoPages = (pages || []).filter(function (p) { return p.category === "seo"; });
    seoContent = content || {};
    var sel = document.getElementById("seoPageSelect");
    if (!sel) return;
    sel.innerHTML = "";
    seoPages.forEach(function (p, i) {
      var opt = document.createElement("option");
      opt.value = p.id;
      opt.textContent = p.label;
      sel.appendChild(opt);
      if (i === 0) renderSeoFields(p);
    });
    sel.onchange = function () {
      var page = seoPages.find(function (p) { return p.id === sel.value; });
      if (page) renderSeoFields(page);
    };
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
      box.appendChild(wrap);
    });
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
    try {
      var res = await fetch(apiBase() + "/api/admin/sugestoes", { headers: authHeaders() });
      var data = await res.json();
      var items = data.items || [];
      if (!items.length) {
        div.innerHTML = '<p class="oe-admin-empty">Sem sugestões.</p>';
        return;
      }
      div.innerHTML = "";
      div.appendChild(
        buildTable(
          ["Nome", "Mensagem", "Data", "Lida"],
          items.map(function (s) {
            return [
              s.nome || "—",
              (s.mensagem || "").slice(0, 80),
              (s.created_at || "").replace("T", " ").replace("Z", ""),
              s.lida ? "Sim" : "Não",
            ];
          })
        )
      );
    } catch (e) {
      div.innerHTML = '<p class="oe-admin-empty">Erro ao carregar.</p>';
    }
  }

  async function loadSystem() {
    var health = document.getElementById("systemHealth");
    try {
      var hres = await fetch(apiBase() + "/api/admin/health", { headers: authHeaders() });
      var h = await hres.json();
      if (health) {
        health.innerHTML =
          "<ul style='margin:0;padding-left:20px;line-height:1.8'>" +
          "<li><strong>API:</strong> " + h.api + "</li>" +
          "<li><strong>Base de dados:</strong> " + h.database + " (" + Math.round((h.database_bytes || 0) / 1024) + " KB)</li>" +
          "<li><strong>OpenAI:</strong> " + h.openai + "</li>" +
          "<li><strong>Disco livre:</strong> " + (h.disk_free_mb != null ? h.disk_free_mb + " MB" : "—") + "</li>" +
          "</ul>";
      }
      var cres = await fetch(apiBase() + "/api/admin/config", { headers: authHeaders() });
      var cdata = await cres.json();
      var cfg = cdata.config || {};
      document.getElementById("cfgMaxMb").value = cfg.max_file_size_mb || "";
      document.getElementById("cfgAlertEmail").checked = cfg.alert_email_enabled === "1";
      document.getElementById("cfgAlertTo").value = cfg.alert_email_to || "";
      document.getElementById("cfgAlertTrans").value = cfg.alert_transcriptions_daily || "";
      document.getElementById("cfgAlertVisits").value = cfg.alert_visits_daily || "";
      document.getElementById("cfgWhisperCost").value = cfg.whisper_cost_per_minute_usd || "0.006";
      document.getElementById("cfgCfZone").value = cfg.cloudflare_zone_id || "";
      document.getElementById("cfgCfToken").value = "";
      var bres = await fetch(apiBase() + "/api/admin/banners", { headers: authHeaders() });
      var bdata = await bres.json();
      var banner = (bdata.items || [])[0];
      if (banner) {
        document.getElementById("bannerTexto").value = banner.texto || "";
        document.getElementById("bannerLink").value = banner.link || "";
        document.getElementById("bannerAtivo").checked = !!banner.ativo;
      }
      var ures = await fetch(apiBase() + "/api/admin/users", { headers: authHeaders() });
      var udata = await ures.json();
      var ul = document.getElementById("usersList");
      if (ul) {
        var users = udata.items || [];
        ul.innerHTML = users.length
          ? "<ul>" + users.map(function (u) {
            return "<li><strong>" + u.username + "</strong> — " + u.role + "</li>";
          }).join("") + "</ul>"
          : "<p class='oe-admin-empty'>Só o admin por defeito.</p>";
      }
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
      if (health) health.innerHTML = '<p class="oe-admin-empty">Erro ao carregar sistema.</p>';
    }
  }

  async function saveConfig(e) {
    e.preventDefault();
    var updates = {
      max_file_size_mb: document.getElementById("cfgMaxMb").value,
      alert_email_enabled: document.getElementById("cfgAlertEmail").checked ? "1" : "0",
      alert_email_to: document.getElementById("cfgAlertTo").value,
      alert_transcriptions_daily: document.getElementById("cfgAlertTrans").value,
      alert_visits_daily: document.getElementById("cfgAlertVisits").value,
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
    if (tab === "sugestoes") loadSugestoes();
    if (tab === "sistema") loadSystem();
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  global.OuviescreviAdminExt = {
    setupSeo: setupSeo,
    renderReferrersAndDevices: renderReferrersAndDevices,
    onTab: onTab,
    loadSugestoes: loadSugestoes,
    loadSystem: loadSystem,
  };
})(window);
