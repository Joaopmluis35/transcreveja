/**
 * Backoffice — editor de menu e rodapé (nav_config_*).
 */
(function (global) {
  "use strict";

  var NAV_LANGS = [
    { id: "pt", label: "Português" },
    { id: "en", label: "English" },
    { id: "es", label: "Español" },
    { id: "fr", label: "Français" },
    { id: "de", label: "Deutsch" },
  ];

  var navLang = "pt";
  var navDraft = null;
  var navLoaded = false;

  function apiBase() {
    return global.OuviescreviAPI.getBase();
  }

  function navKey(lang) {
    return lang === "pt" ? "nav_config_pt" : "nav_config_" + lang;
  }

  function defaultNav(lang) {
    var isEn = lang !== "pt";
    return {
      menuToolsLabel: isEn ? "Tools" : "Ferramentas",
      menuAudienceLabel: isEn ? "For" : "Para quem",
      tools: [],
      audience: [],
      topLinks: [],
      ctaLabel: isEn ? "Transcribe free" : "Transcrever grátis",
      ctaHref: "index.html",
      footerTagline: isEn
        ? "Transcribe, summarize and translate with AI — free and made in Portugal."
        : "Transcreve, resume e traduz com IA — grátis e feito em Portugal.",
      footerEmail: "ouviescrevi@gmail.com",
      footerCopyright: isEn ? "© 2026 Ouviescrevi · Made in Portugal" : "© 2026 Ouviescrevi · Feito em Portugal",
      footerColumns: [],
    };
  }

  function parseNav(raw, lang) {
    if (!raw) return defaultNav(lang);
    try {
      var data = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (data && typeof data === "object") return data;
    } catch (e) {}
    return defaultNav(lang);
  }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/"/g, "&quot;")
      .replace(/</g, "&lt;");
  }

  function linkRowHtml(link, prefix) {
    link = link || {};
    return (
      '<div class="oe-admin-nav-row" data-nav-prefix="' +
      prefix +
      '">' +
      '<input type="text" class="oe-admin-nav-label" placeholder="Texto" value="' +
      esc(link.label) +
      '">' +
      '<input type="text" class="oe-admin-nav-href" placeholder="href (ex. resumo.html)" value="' +
      esc(link.href) +
      '">' +
      '<input type="text" class="oe-admin-nav-page" placeholder="slug (opcional)" value="' +
      esc(link.page || "") +
      '">' +
      '<label class="oe-admin-nav-check"><input type="checkbox" class="oe-admin-nav-pricing"' +
      (link.pricingOnly ? " checked" : "") +
      "> Pro</label>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary oe-admin-nav-rm" title="Remover">✕</button>' +
      "</div>"
    );
  }

  function columnHtml(col, colIdx) {
    col = col || { title: "", links: [] };
    var links = (col.links || []).map(function (l) {
      return linkRowHtml(l, "col-" + colIdx);
    }).join("");
    return (
      '<div class="oe-admin-nav-col" data-col-idx="' +
      colIdx +
      '">' +
      '<div class="oe-admin-field"><label>Título da coluna</label>' +
      '<input type="text" class="oe-admin-nav-col-title" value="' +
      esc(col.title) +
      '"></div>' +
      '<div class="oe-admin-nav-links">' +
      links +
      "</div>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary oe-admin-nav-add-link" data-col="' +
      colIdx +
      '">+ Link</button>' +
      "</div>"
    );
  }

  function renderEditor() {
    var root = document.getElementById("navEditorRoot");
    if (!root || !navDraft) return;
    var d = navDraft;
    root.innerHTML =
      '<div class="oe-admin-form oe-admin-nav-form">' +
      '<div class="oe-admin-field-row">' +
      '<div class="oe-admin-field"><label>Menu — Ferramentas (rótulo)</label>' +
      '<input type="text" id="navMenuToolsLabel" value="' +
      esc(d.menuToolsLabel) +
      '"></div>' +
      '<div class="oe-admin-field"><label>Menu — Para quem (rótulo)</label>' +
      '<input type="text" id="navMenuAudienceLabel" value="' +
      esc(d.menuAudienceLabel) +
      '"></div></div>' +
      "<h3 class=\"oe-admin-nav-section\">Links — Ferramentas</h3>" +
      '<div id="navToolsLinks" class="oe-admin-nav-links">' +
      (d.tools || []).map(function (l) { return linkRowHtml(l, "tools"); }).join("") +
      "</div>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary" id="navAddTool">+ Ferramenta</button>' +
      "<h3 class=\"oe-admin-nav-section\">Links — Para quem</h3>" +
      '<div id="navAudienceLinks" class="oe-admin-nav-links">' +
      (d.audience || []).map(function (l) { return linkRowHtml(l, "audience"); }).join("") +
      "</div>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary" id="navAddAudience">+ Link</button>' +
      "<h3 class=\"oe-admin-nav-section\">Links no topo (Ajuda, Preços…)</h3>" +
      '<div id="navTopLinks" class="oe-admin-nav-links">' +
      (d.topLinks || []).map(function (l) { return linkRowHtml(l, "top"); }).join("") +
      "</div>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary" id="navAddTop">+ Link topo</button>' +
      '<div class="oe-admin-field-row" style="margin-top:16px">' +
      '<div class="oe-admin-field"><label>Botão CTA</label><input type="text" id="navCtaLabel" value="' +
      esc(d.ctaLabel) +
      '"></div>' +
      '<div class="oe-admin-field"><label>CTA href</label><input type="text" id="navCtaHref" value="' +
      esc(d.ctaHref) +
      '"></div></div>' +
      "<h3 class=\"oe-admin-nav-section\">Rodapé</h3>" +
      '<div class="oe-admin-field"><label>Tagline</label><input type="text" id="navFooterTagline" value="' +
      esc(d.footerTagline) +
      '"></div>' +
      '<div class="oe-admin-field-row">' +
      '<div class="oe-admin-field"><label>Email</label><input type="text" id="navFooterEmail" value="' +
      esc(d.footerEmail) +
      '"></div>' +
      '<div class="oe-admin-field"><label>Copyright</label><input type="text" id="navFooterCopyright" value="' +
      esc(d.footerCopyright) +
      '"></div></div>' +
      '<div id="navFooterCols">' +
      (d.footerColumns || []).map(function (c, i) { return columnHtml(c, i); }).join("") +
      "</div>" +
      '<button type="button" class="oe-admin-btn oe-admin-btn--secondary" id="navAddColumn">+ Coluna rodapé</button>' +
      "</div>";

    bindEditorEvents();
  }

  function bindEditorEvents() {
    var root = document.getElementById("navEditorRoot");
    if (!root) return;
    root.querySelectorAll(".oe-admin-nav-rm").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var row = btn.closest(".oe-admin-nav-row");
        if (row) row.remove();
      });
    });
    var addTool = document.getElementById("navAddTool");
    if (addTool) {
      addTool.addEventListener("click", function () {
        document.getElementById("navToolsLinks").insertAdjacentHTML("beforeend", linkRowHtml({}, "tools"));
        bindEditorEvents();
      });
    }
    var addAud = document.getElementById("navAddAudience");
    if (addAud) {
      addAud.addEventListener("click", function () {
        document.getElementById("navAudienceLinks").insertAdjacentHTML("beforeend", linkRowHtml({}, "audience"));
        bindEditorEvents();
      });
    }
    var addTop = document.getElementById("navAddTop");
    if (addTop) {
      addTop.addEventListener("click", function () {
        document.getElementById("navTopLinks").insertAdjacentHTML("beforeend", linkRowHtml({}, "top"));
        bindEditorEvents();
      });
    }
    root.querySelectorAll(".oe-admin-nav-add-link").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var col = btn.closest(".oe-admin-nav-col");
        var links = col.querySelector(".oe-admin-nav-links");
        var idx = col.getAttribute("data-col-idx");
        links.insertAdjacentHTML("beforeend", linkRowHtml({}, "col-" + idx));
        bindEditorEvents();
      });
    });
    var addCol = document.getElementById("navAddColumn");
    if (addCol) {
      addCol.addEventListener("click", function () {
        var cols = document.getElementById("navFooterCols");
        var n = cols.querySelectorAll(".oe-admin-nav-col").length;
        cols.insertAdjacentHTML("beforeend", columnHtml({ title: "Nova coluna", links: [] }, n));
        bindEditorEvents();
      });
    }
  }

  function collectLinks(containerId) {
    var box = document.getElementById(containerId);
    if (!box) return [];
    return Array.prototype.map.call(box.querySelectorAll(".oe-admin-nav-row"), function (row) {
      var item = {
        label: (row.querySelector(".oe-admin-nav-label") || {}).value || "",
        href: (row.querySelector(".oe-admin-nav-href") || {}).value || "",
      };
      var page = (row.querySelector(".oe-admin-nav-page") || {}).value || "";
      if (page) item.page = page;
      if ((row.querySelector(".oe-admin-nav-pricing") || {}).checked) item.pricingOnly = true;
      return item;
    }).filter(function (l) { return l.label && l.href; });
  }

  function collectDraft() {
    var cols = [];
    document.querySelectorAll("#navFooterCols .oe-admin-nav-col").forEach(function (col) {
      var title = (col.querySelector(".oe-admin-nav-col-title") || {}).value || "";
      var links = Array.prototype.map.call(col.querySelectorAll(".oe-admin-nav-row"), function (row) {
        return {
          label: (row.querySelector(".oe-admin-nav-label") || {}).value || "",
          href: (row.querySelector(".oe-admin-nav-href") || {}).value || "",
        };
      }).filter(function (l) { return l.label && l.href; });
      if (title || links.length) cols.push({ title: title, links: links });
    });
    return {
      menuToolsLabel: (document.getElementById("navMenuToolsLabel") || {}).value || "",
      menuAudienceLabel: (document.getElementById("navMenuAudienceLabel") || {}).value || "",
      tools: collectLinks("navToolsLinks"),
      audience: collectLinks("navAudienceLinks"),
      topLinks: collectLinks("navTopLinks"),
      ctaLabel: (document.getElementById("navCtaLabel") || {}).value || "",
      ctaHref: (document.getElementById("navCtaHref") || {}).value || "index.html",
      footerTagline: (document.getElementById("navFooterTagline") || {}).value || "",
      footerEmail: (document.getElementById("navFooterEmail") || {}).value || "",
      footerCopyright: (document.getElementById("navFooterCopyright") || {}).value || "",
      footerColumns: cols,
    };
  }

  async function loadNavEditor() {
    var root = document.getElementById("navEditorRoot");
    if (!root) return;
    root.innerHTML = "<p class=\"oe-admin-empty\">A carregar…</p>";
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        headers: global.OuviescreviAPI.adminAuthHeaders(),
      });
      var data = await res.json();
      var content = data.content || {};
      navDraft = parseNav(content[navKey(navLang)], navLang);
      renderEditor();
      navLoaded = true;
    } catch (e) {
      root.innerHTML = "<p class=\"oe-admin-empty\">Erro ao carregar menu.</p>";
    }
  }

  async function saveNav() {
    var draft = collectDraft();
    var updates = {};
    updates[navKey(navLang)] = JSON.stringify(draft);
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content", {
        method: "PUT",
        headers: global.OuviescreviAPI.adminAuthHeaders(),
        body: JSON.stringify({ updates: updates }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      navDraft = draft;
      global.OuviescreviUI.toast("Menu e rodapé guardados (" + navLang.toUpperCase() + ").", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar menu.", "error");
    }
  }

  async function resetNav() {
    if (!confirm("Repor menu e rodapé de " + navLang.toUpperCase() + " aos valores originais?")) return;
    try {
      var res = await fetch(apiBase() + "/api/admin/site-content/reset", {
        method: "POST",
        headers: global.OuviescreviAPI.adminAuthHeaders(),
        body: JSON.stringify({ keys: [navKey(navLang)] }),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      await loadNavEditor();
      global.OuviescreviUI.toast("Menu reposto.", "success");
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao repor.", "error");
    }
  }

  function setup() {
    var langSel = document.getElementById("navLangSelect");
    if (langSel) {
      NAV_LANGS.forEach(function (l) {
        var opt = document.createElement("option");
        opt.value = l.id;
        opt.textContent = l.label;
        langSel.appendChild(opt);
      });
      langSel.addEventListener("change", function () {
        navLang = langSel.value || "pt";
        loadNavEditor();
      });
    }
    var saveBtn = document.getElementById("navSaveBtn");
    if (saveBtn) saveBtn.addEventListener("click", saveNav);
    var resetBtn = document.getElementById("navResetBtn");
    if (resetBtn) resetBtn.addEventListener("click", resetNav);
  }

  function onTab(tab) {
    if (tab === "nav") {
      if (!navLoaded) loadNavEditor();
      else renderEditor();
    }
  }

  global.OuviescreviAdminNav = {
    setup: setup,
    onTab: onTab,
    load: loadNavEditor,
  };
})(window);
