/**
 * Pequenos utilitários de UX — toasts, estado de loading em botões.
 */
(function (global) {
  let toastHost = null;

  function ensureToastHost() {
    if (toastHost && document.body.contains(toastHost)) return toastHost;
    toastHost = document.createElement("div");
    toastHost.className = "oe-toast-host";
    toastHost.setAttribute("aria-live", "polite");
    toastHost.setAttribute("aria-atomic", "true");
    document.body.appendChild(toastHost);
    return toastHost;
  }

  function toast(message, type) {
    const host = ensureToastHost();
    const el = document.createElement("div");
    el.className = "oe-toast" + (type ? " oe-toast--" + type : "");
    el.textContent = message;
    host.appendChild(el);
    setTimeout(function () {
      el.style.opacity = "0";
      el.style.transition = "opacity 0.25s";
      setTimeout(function () { el.remove(); }, 280);
    }, 3800);
  }

  function setButtonLoading(btn, loading, loadingLabel) {
    if (!btn) return;
    if (loading) {
      if (!btn.dataset.oeOriginalHtml) {
        btn.dataset.oeOriginalHtml = btn.innerHTML;
      }
      btn.disabled = true;
      btn.innerHTML =
        '<span class="oe-spinner" aria-hidden="true"></span> ' +
        (loadingLabel || "A processar...");
      btn.setAttribute("aria-busy", "true");
    } else {
      btn.disabled = false;
      if (btn.dataset.oeOriginalHtml) {
        btn.innerHTML = btn.dataset.oeOriginalHtml;
        delete btn.dataset.oeOriginalHtml;
      }
      btn.removeAttribute("aria-busy");
    }
  }

  function markCurrentNav() {
    if (global.OuviescreviNav && global.OuviescreviNav.markCurrentPage) {
      global.OuviescreviNav.markCurrentPage();
      return;
    }
    const path = global.location.pathname.replace(/\/$/, "");
    const page = path.split("/").pop().replace(".html", "") || "index";
    document.querySelectorAll("nav#nichoMenu button").forEach(function (btn) {
      const onclick = btn.getAttribute("onclick") || "";
      if (onclick.indexOf("'" + page + "'") !== -1 || onclick.indexOf('"' + page + '"') !== -1) {
        btn.style.fontWeight = "800";
        btn.style.boxShadow = "0 0 0 2px rgba(255,255,255,0.9)";
      }
    });
  }

  function injectScriptsFromHtml(html, root) {
    (root || document).querySelectorAll("script").forEach(function (oldScript) {
      if (oldScript.src && oldScript.src.indexOf("ouviescrevi-nav.js") !== -1) return;
      const newScript = document.createElement("script");
      if (oldScript.src) newScript.src = oldScript.src;
      else newScript.textContent = oldScript.textContent;
      document.body.appendChild(newScript);
    });
  }

  function ensureI18nScript() {
    if (global.OuviescreviI18n) return Promise.resolve();
    var existing = document.querySelector('script[data-oe-i18n="1"]');
    if (existing) {
      if (existing.getAttribute("data-ready") === "1") return Promise.resolve();
      return new Promise(function (resolve) {
        existing.addEventListener("load", resolve, { once: true });
      });
    }
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = "/js/ouviescrevi-i18n.js";
      s.dataset.oeI18n = "1";
      s.onload = function () {
        s.setAttribute("data-ready", "1");
        resolve();
      };
      s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  function ensureNavScript() {
    return ensureI18nScript().then(function () {
      if (global.OuviescreviNav) return Promise.resolve();
      var existing = document.querySelector('script[data-oe-nav="1"]');
      if (existing) {
        if (existing.getAttribute("data-ready") === "1") return Promise.resolve();
        return new Promise(function (resolve) {
          existing.addEventListener("load", resolve, { once: true });
        });
      }
      return new Promise(function (resolve) {
        var s = document.createElement("script");
        s.src = "/js/ouviescrevi-nav.js";
        s.dataset.oeNav = "1";
        s.onload = function () {
          s.setAttribute("data-ready", "1");
          resolve();
        };
        s.onerror = resolve;
        document.head.appendChild(s);
      });
    });
  }

  function ensureAuthScript() {
    if (global.OuviescreviAuth) return Promise.resolve();
    var existing = document.querySelector('script[data-oe-auth="1"]');
    if (existing) {
      if (existing.getAttribute("data-ready") === "1") return Promise.resolve();
      return new Promise(function (resolve) {
        existing.addEventListener("load", resolve, { once: true });
      });
    }
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = "/js/auth-ui.js";
      s.dataset.oeAuth = "1";
      s.onload = function () {
        s.setAttribute("data-ready", "1");
        resolve();
      };
      s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  function ensurePricingVisibilityScript() {
    if (global.OuviescreviPricingVisibility) {
      return global.OuviescreviPricingVisibility.init();
    }
    var existing = document.querySelector('script[data-oe-pricing-vis="1"]');
    if (existing) {
      if (existing.getAttribute("data-ready") === "1" && global.OuviescreviPricingVisibility) {
        return global.OuviescreviPricingVisibility.init();
      }
      return new Promise(function (resolve) {
        existing.addEventListener("load", function () {
          if (global.OuviescreviPricingVisibility) {
            global.OuviescreviPricingVisibility.init().then(resolve);
          } else {
            resolve();
          }
        }, { once: true });
      });
    }
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = "/js/pricing-visibility.js";
      s.dataset.oePricingVis = "1";
      s.onload = function () {
        s.setAttribute("data-ready", "1");
        if (global.OuviescreviPricingVisibility) {
          global.OuviescreviPricingVisibility.init().then(resolve);
        } else {
          resolve();
        }
      };
      s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  function ensureThemeScript() {
    if (global.OuviescreviTheme) {
      global.OuviescreviTheme.init();
      return Promise.resolve();
    }
    var existing = document.querySelector('script[data-oe-theme="1"]');
    if (existing) {
      if (existing.getAttribute("data-ready") === "1" && global.OuviescreviTheme) {
        global.OuviescreviTheme.init();
        return Promise.resolve();
      }
      return new Promise(function (resolve) {
        existing.addEventListener("load", function () {
          if (global.OuviescreviTheme) global.OuviescreviTheme.init();
          resolve();
        }, { once: true });
      });
    }
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = "/js/theme-ui.js";
      s.dataset.oeTheme = "1";
      s.onload = function () {
        s.setAttribute("data-ready", "1");
        if (global.OuviescreviTheme) global.OuviescreviTheme.init();
        resolve();
      };
      s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  function ensureNewsTickerScript() {
    if (global.OuviescreviNewsTicker) return Promise.resolve();
    var existing = document.querySelector('script[data-oe-news-ticker="1"]');
    if (existing) {
      if (existing.getAttribute("data-ready") === "1") return Promise.resolve();
      return new Promise(function (resolve) {
        existing.addEventListener("load", resolve, { once: true });
      });
    }
    return new Promise(function (resolve) {
      var s = document.createElement("script");
      s.src = "/js/news-ticker.js?v=1";
      s.dataset.oeNewsTicker = "1";
      s.onload = function () {
        s.setAttribute("data-ready", "1");
        resolve();
      };
      s.onerror = resolve;
      document.head.appendChild(s);
    });
  }

  function mountNewsTicker() {
    return ensureNewsTickerScript().then(function () {
      if (global.OuviescreviNewsTicker && global.OuviescreviNewsTicker.mount) {
        global.OuviescreviNewsTicker.mount();
      }
    });
  }

  var LAYOUT_V = "10";
  var siteContentCache = null;

  function withLayoutVersion(url) {
    if (!url) return url;
    return url + (url.indexOf("?") === -1 ? "?" : "&") + "v=" + LAYOUT_V;
  }

  function resolveHeaderUrl(explicit) {
    if (explicit) return withLayoutVersion(explicit);
    var el = document.getElementById("header");
    if (el && el.getAttribute("data-header-url")) {
      return withLayoutVersion(el.getAttribute("data-header-url"));
    }
    return withLayoutVersion("header.html");
  }

  function resolveFooterUrl(explicit) {
    if (explicit) return withLayoutVersion(explicit);
    var el = document.getElementById("footer");
    if (el && el.getAttribute("data-footer-url")) {
      return withLayoutVersion(el.getAttribute("data-footer-url"));
    }
    return withLayoutVersion("footer.html");
  }

  function loadHeader(url) {
    url = resolveHeaderUrl(url);
    return ensureNavScript()
      .then(function () {
        return ensureThemeScript();
      })
      .then(function () {
        return fetch(url);
      })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (html) {
        const el = document.getElementById("header");
        if (!el) return;
        el.innerHTML = html;
        el.dataset.oeHeaderLoaded = "true";
        const temp = document.createElement("div");
        temp.innerHTML = html;
        injectScriptsFromHtml(html, temp);
        if (global.OuviescreviNav && global.OuviescreviNav.init) {
          global.OuviescreviNav.init();
        }
        if (global.OuviescreviTheme) global.OuviescreviTheme.init();
        markCurrentNav();
        maybeApplyNavFromCache();
        return ensureAuthScript();
      })
      .then(function () {
        if (global.OuviescreviAuth && global.OuviescreviAuth.init) {
          global.OuviescreviAuth.init();
        }
        return ensurePricingVisibilityScript();
      })
      .then(function () {
        return mountNewsTicker();
      })
      .catch(function (err) {
        console.error("OuviescreviUI: falha ao carregar", url, err);
      });
  }

  function loadFooter(url) {
    url = resolveFooterUrl(url);
    return fetch(url)
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function (html) {
        let el = document.getElementById("footer");
        if (!el) {
          el = document.createElement("div");
          el.id = "footer";
          document.body.appendChild(el);
        }
        el.innerHTML = html;
        el.dataset.oeFooterLoaded = "true";
        const temp = document.createElement("div");
        temp.innerHTML = html;
        injectScriptsFromHtml(html, temp);
        maybeApplyNavFromCache();
      })
      .catch(function (err) {
        console.error("OuviescreviUI: falha ao carregar", url, err);
      });
  }

  function autoLoadLayout() {
    ensureThemeScript();
    var headerEl = document.getElementById("header");
    if (
      headerEl &&
      headerEl.dataset.oeSkipHeader !== "true" &&
      !headerEl.querySelector("#oeProHeader") &&
      !headerEl.querySelector("#topoOuviescrevi")
    ) {
      loadHeader();
    }
    var footerEl = document.getElementById("footer");
    if (
      footerEl &&
      footerEl.dataset.oeSkipFooter !== "true" &&
      !footerEl.querySelector("#rodapeOuviescrevi")
    ) {
      loadFooter();
    } else if (!footerEl && document.body.dataset.oeAutoFooter !== "false") {
      loadFooter();
    }
  }

  function trackPageView() {
    var path;
    var key;
    try {
      if ((global.location.pathname || "").indexOf("backoffice") !== -1) return;
      path = global.location.pathname || "/";
      key = "oe_track_" + path;
      if (sessionStorage.getItem(key)) return;
    } catch (e) {
      return;
    }

    function sendTrack(base) {
      if (!base) return;
      fetch(base + "/api/track-visit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "omit",
        keepalive: true,
        body: JSON.stringify({
          path: path,
          referrer: document.referrer || "",
        }),
      })
        .then(function (res) {
          if (res.ok) {
            try {
              sessionStorage.setItem(key, "1");
            } catch (e) {}
          }
        })
        .catch(function () {});
    }

    if (global.OuviescreviAPI && global.OuviescreviAPI.init) {
      global.OuviescreviAPI.init()
        .then(function () {
          sendTrack(global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase());
        })
        .catch(function () {
          sendTrack(global.OuviescreviAPI.detectApiBase());
        });
    } else if (global.OuviescreviAPI && global.OuviescreviAPI.detectApiBase) {
      sendTrack(global.OuviescreviAPI.detectApiBase());
    }
  }

  function cookieBannerPath() {
    var I = global.OuviescreviI18n;
    if (I) return I.uiStrings(I.localeFromPath()).cookiesPath;
    var path = global.location.pathname || "";
    return path.indexOf("/en/") !== -1 ? "/en/cookies.html" : "/cookies.html";
  }

  function maybeShowCookieBanner() {
    try {
      if ((global.location.pathname || "").indexOf("backoffice") !== -1) return;
      if (localStorage.getItem("oe_cookies_ack")) return;
      if (document.getElementById("oe-cookie-banner")) return;
    } catch (e) {
      return;
    }
    var loc = global.OuviescreviI18n
      ? global.OuviescreviI18n.localeFromPath()
      : (global.location.pathname || "").indexOf("/en/") !== -1
        ? "en"
        : "pt";
    var strings = global.OuviescreviI18n
      ? global.OuviescreviI18n.uiStrings(loc)
      : loc === "en"
        ? {
            cookieAria: "Cookie notice",
            cookieText: "We use essential browser storage for the service to work. ",
            cookieLink: "Cookie Policy",
            cookieBtn: "OK",
          }
        : {
            cookieAria: "Aviso de cookies",
            cookieText: "Utilizamos armazenamento essencial no browser para o serviço funcionar. ",
            cookieLink: "Política de Cookies",
            cookieBtn: "Compreendi",
          };
    var banner = document.createElement("div");
    banner.id = "oe-cookie-banner";
    banner.className = "oe-cookie-banner";
    banner.setAttribute("role", "dialog");
    banner.setAttribute("aria-label", strings.cookieAria);
    banner.innerHTML =
      strings.cookieText +
      '<a href="' + cookieBannerPath() + '">' +
      strings.cookieLink +
      "</a>";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "oe-cookie-banner__btn";
    btn.textContent = strings.cookieBtn;
    btn.addEventListener("click", function () {
      try {
        localStorage.setItem("oe_cookies_ack", "1");
      } catch (e) {}
      banner.remove();
    });
    banner.appendChild(btn);
    document.body.appendChild(banner);
  }

  function cmsApiBase() {
    if (global.OuviescreviAPI && global.OuviescreviAPI.detectApiBase) {
      return global.OuviescreviAPI.detectApiBase();
    }
    if (global.OUVIESCREVI_API_BASE) {
      return global.OUVIESCREVI_API_BASE.replace(/\/$/, "");
    }
    var meta = document.querySelector('meta[name="ouviescrevi-api-base"]');
    if (meta && meta.content) return meta.content.replace(/\/$/, "");
    var host = global.location && global.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") return "http://127.0.0.1:8000";
    return "https://api.ouviescrevi.pt";
  }

  function detectSiteLocale() {
    if (global.OuviescreviI18n && global.OuviescreviI18n.getLocale) {
      return global.OuviescreviI18n.getLocale();
    }
    var p = (global.location.pathname || "");
    var m = p.match(/^\/(en|es|fr|de)(?:\/|$)/);
    return m ? m[1] : "pt";
  }

  function navConfigKeyForLocale(locale) {
    return locale === "pt" ? "nav_config_pt" : "nav_config_" + locale;
  }

  function createNavLink(item, role) {
    var a = document.createElement("a");
    a.href = item.href || "#";
    a.textContent = item.label;
    if (item.page) a.setAttribute("data-nav-page", item.page);
    if (item.pricingOnly) a.setAttribute("data-pricing-only", "");
    if (role) a.setAttribute("role", role);
    return a;
  }

  function navToolsLocale(locale) {
    if (locale === "pt" || locale === "en" || locale === "es" || locale === "fr" || locale === "de") {
      return locale;
    }
    return "en";
  }

  function builtinToolsForLocale(locale) {
    if (locale === "en") {
      return [
        { label: "Summarize PDF / Word", href: "resumo.html", page: "resumo", category: "Summarize & analyze" },
        { label: "URL Summary", href: "url-resumo.html", page: "url-resumo", category: "Summarize & analyze" },
        { label: "Chapters & timestamps", href: "capitulos.html", page: "capitulos", category: "Summarize & analyze" },
        { label: "Podcast & YouTube", href: "podcast-youtube.html", page: "podcast-youtube", category: "Summarize & analyze" },
        { label: "YouTube Description", href: "descricao-youtube.html", page: "descricao-youtube", category: "Summarize & analyze" },
        { label: "AI Questions", href: "perguntas.html", page: "perguntas", category: "Study & teaching" },
        { label: "Lesson Ready", href: "aula-pronta.html", page: "aula-pronta", category: "Study & teaching" },
        { label: "Flashcards", href: "flashcards.html", page: "flashcards", category: "Study & teaching" },
        { label: "File Converter", href: "conversor.html", page: "conversor", category: "Convert & edit" },
        { label: "Image Converter", href: "conversor-imagens.html", page: "conversor-imagens", category: "Convert & edit" },
        { label: "Text proofreader", href: "corretor.html", page: "corretor", category: "Convert & edit" },
      ];
    }
    if (locale === "es") {
      return [
        { label: "Resumen PDF / Word", href: "resumo.html", page: "resumo", category: "Resumir y analizar" },
        { label: "Resumen por URL", href: "url-resumo.html", page: "url-resumo", category: "Resumir y analizar" },
        { label: "Podcast y YouTube", href: "podcast-youtube.html", page: "podcast-youtube", category: "Resumir y analizar" },
        { label: "Descripción YouTube", href: "descricao-youtube.html", page: "descricao-youtube", category: "Resumir y analizar" },
        { label: "Preguntas con IA", href: "perguntas.html", page: "perguntas", category: "Estudio y enseñanza" },
        { label: "Flashcards", href: "flashcards.html", page: "flashcards", category: "Estudio y enseñanza" },
        { label: "Conversor de archivos", href: "conversor.html", page: "conversor", category: "Convertir y corregir" },
        { label: "Corrector de texto", href: "corretor.html", page: "corretor", category: "Convertir y corregir" },
      ];
    }
    if (locale === "fr") {
      return [
        { label: "Résumé PDF / Word", href: "resumo.html", page: "resumo", category: "Résumer et analyser" },
        { label: "Résumé par URL", href: "url-resumo.html", page: "url-resumo", category: "Résumer et analyser" },
        { label: "Podcast & YouTube", href: "podcast-youtube.html", page: "podcast-youtube", category: "Résumer et analyser" },
        { label: "Description YouTube", href: "descricao-youtube.html", page: "descricao-youtube", category: "Résumer et analyser" },
        { label: "Questions IA", href: "perguntas.html", page: "perguntas", category: "Étude et enseignement" },
        { label: "Flashcards", href: "flashcards.html", page: "flashcards", category: "Étude et enseignement" },
        { label: "Convertisseur de fichiers", href: "conversor.html", page: "conversor", category: "Convertir et corriger" },
        { label: "Correcteur de texte", href: "corretor.html", page: "corretor", category: "Convertir et corriger" },
      ];
    }
    if (locale === "de") {
      return [
        { label: "PDF-/Word-Zusammenfassung", href: "resumo.html", page: "resumo", category: "Zusammenfassen & analysieren" },
        { label: "URL-Zusammenfassung", href: "url-resumo.html", page: "url-resumo", category: "Zusammenfassen & analysieren" },
        { label: "Podcast & YouTube", href: "podcast-youtube.html", page: "podcast-youtube", category: "Zusammenfassen & analysieren" },
        { label: "YouTube-Beschreibung", href: "descricao-youtube.html", page: "descricao-youtube", category: "Zusammenfassen & analysieren" },
        { label: "KI-Fragen", href: "perguntas.html", page: "perguntas", category: "Lernen & Unterricht" },
        { label: "Karteikarten", href: "flashcards.html", page: "flashcards", category: "Lernen & Unterricht" },
        { label: "Dateikonverter", href: "conversor.html", page: "conversor", category: "Konvertieren & korrigieren" },
        { label: "Textkorrektur", href: "corretor.html", page: "corretor", category: "Konvertieren & korrigieren" },
      ];
    }
    return [
      { label: "Resumo PDF / Word", href: "resumo.html", page: "resumo", category: "Resumir e analisar" },
      { label: "Resumo por URL", href: "url-resumo.html", page: "url-resumo", category: "Resumir e analisar" },
      { label: "Capítulos & timestamps", href: "capitulos.html", page: "capitulos", category: "Resumir e analisar" },
      { label: "Podcast & YouTube", href: "podcast-youtube.html", page: "podcast-youtube", category: "Resumir e analisar" },
      { label: "Descrição YouTube", href: "descricao-youtube.html", page: "descricao-youtube", category: "Resumir e analisar" },
      { label: "Perguntas com IA", href: "perguntas.html", page: "perguntas", category: "Estudo e ensino" },
      { label: "Aula Pronta", href: "aula-pronta.html", page: "aula-pronta", category: "Estudo e ensino" },
      { label: "Flashcards", href: "flashcards.html", page: "flashcards", category: "Estudo e ensino" },
      { label: "Conversor de ficheiros", href: "conversor.html", page: "conversor", category: "Converter e corrigir" },
      { label: "Conversor de imagens", href: "conversor-imagens.html", page: "conversor-imagens", category: "Converter e corrigir" },
      { label: "Corretor de texto", href: "corretor.html", page: "corretor", category: "Converter e corrigir" },
    ];
  }

  function mergeToolsConfig(stored, locale) {
    var builtin = builtinToolsForLocale(navToolsLocale(locale));
    var storedMap = {};
    (stored || []).forEach(function (item) {
      if (item && item.page) storedMap[item.page] = item;
    });
    var result = [];
    var seen = {};
    builtin.forEach(function (base) {
      var custom = storedMap[base.page];
      if (custom && custom.hidden) return;
      if (custom) {
        result.push(
          Object.assign({}, base, custom, {
            category: custom.category || base.category,
          })
        );
      } else {
        result.push(Object.assign({}, base));
      }
      if (base.page) seen[base.page] = true;
    });
    (stored || []).forEach(function (item) {
      if (!item || !item.label || item.hidden) return;
      if (item.page && seen[item.page]) return;
      result.push(
        Object.assign({}, item, {
          category: item.category || "",
        })
      );
    });
    return result;
  }

  function enrichToolsWithCategories(tools, defaults) {
    var defaultMap = {};
    (defaults || []).forEach(function (d) {
      if (d && d.category) {
        if (d.page) defaultMap[d.page] = d.category;
        if (d.href) defaultMap[d.href] = d.category;
      }
    });
    return (tools || []).map(function (item) {
      if (!item || item.category) return item;
      var cat = defaultMap[item.page] || defaultMap[item.href];
      return cat ? Object.assign({}, item, { category: cat }) : item;
    });
  }

  function renderNavLinkList(container, links, role, linkClass) {
    if (!container) return;
    var visible = (links || []).filter(function (item) {
      return item && item.label && !item.hidden;
    });
    if (!visible.length) return;
    container.innerHTML = "";
    container.classList.remove("oe-pro-nav__menu--grouped");
    visible.forEach(function (item) {
      var a = createNavLink(item, role);
      if (linkClass) a.className = linkClass;
      container.appendChild(a);
    });
  }

  function renderNavToolMenu(container, links, role) {
    if (!container) return;
    var visible = (links || []).filter(function (item) {
      return item && item.label && !item.hidden;
    });
    if (!visible.length) return;

    var hasCategories = visible.some(function (item) { return item.category; });
    if (!hasCategories) {
      renderNavLinkList(container, visible, role);
      return;
    }

    container.innerHTML = "";
    container.classList.add("oe-pro-nav__menu--grouped");

    var groups = [];
    var groupMap = {};
    visible.forEach(function (item) {
      var cat = item.category || "";
      if (!groupMap[cat]) {
        groupMap[cat] = { label: cat, items: [] };
        groups.push(groupMap[cat]);
      }
      groupMap[cat].items.push(item);
    });

    groups.forEach(function (group, idx) {
      var grid = container.querySelector(".oe-pro-nav__mega-grid");
      if (!grid) {
        grid = document.createElement("div");
        grid.className = "oe-pro-nav__mega-grid";
        container.appendChild(grid);
      }
      var col = document.createElement("div");
      col.className = "oe-pro-nav__mega-col";
      if (group.label) {
        var label = document.createElement("div");
        label.className = "oe-pro-nav__group-label";
        label.setAttribute("role", "presentation");
        label.textContent = group.label;
        col.appendChild(label);
      } else if (idx > 0) {
        var sep = document.createElement("div");
        sep.className = "oe-pro-nav__group-sep";
        sep.setAttribute("role", "separator");
        col.appendChild(sep);
      }
      var wrap = document.createElement("div");
      wrap.className = "oe-pro-nav__group";
      group.items.forEach(function (item) {
        wrap.appendChild(createNavLink(item, role));
      });
      col.appendChild(wrap);
      grid.appendChild(col);
    });
  }

  function parseNavConfigClient(raw, defaults) {
    var cfg = defaults ? Object.assign({}, defaults) : {};
    if (!raw) return cfg;
    try {
      var data = typeof raw === "string" ? JSON.parse(raw) : raw;
      if (data && typeof data === "object") {
        Object.keys(data).forEach(function (k) {
          if (data[k] != null && data[k] !== "") cfg[k] = data[k];
        });
      }
    } catch (e) {}
    ["tools", "audience", "topLinks", "footerColumns"].forEach(function (key) {
      if (!cfg[key] || !cfg[key].length) {
        cfg[key] = (defaults && defaults[key]) ? defaults[key].slice() : [];
      }
    });
    return cfg;
  }

  function applyNavConfig(content) {
    if (!content) return;
    var locale = detectSiteLocale();
    var raw = content[navConfigKeyForLocale(locale)] || content.nav_config_pt;
    if (!raw) return;
    var cfg;
    try {
      cfg = typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch (e) {
      return;
    }
    if (!cfg || typeof cfg !== "object") return;
    var fallbackLang = locale === "pt" ? "pt" : "en";
    var fallbackRaw = content[navConfigKeyForLocale(fallbackLang)] || content.nav_config_pt;
    var fallbackDefaults;
    try {
      fallbackDefaults = fallbackRaw ? (typeof fallbackRaw === "string" ? JSON.parse(fallbackRaw) : fallbackRaw) : null;
    } catch (e2) {
      fallbackDefaults = null;
    }
    cfg = parseNavConfigClient(cfg, fallbackDefaults);

    var toolsMenu = document.querySelector('[data-nav-slot="tools"]');
    var audienceMenu = document.querySelector('[data-nav-slot="audience"]');
    var topLinks = document.querySelector('[data-nav-slot="top-links"]');
    var toolsLocale = navToolsLocale(locale);
    var mergedTools = mergeToolsConfig(
      enrichToolsWithCategories(cfg.tools, fallbackDefaults && fallbackDefaults.tools),
      toolsLocale
    );
    renderNavToolMenu(toolsMenu, mergedTools, "menuitem");
    renderNavLinkList(audienceMenu, cfg.audience, "menuitem");
    renderNavLinkList(topLinks, cfg.topLinks, null, "oe-pro-nav__link");

    var dropdowns = document.querySelectorAll(".oe-pro-nav__dropdown");
    if (dropdowns[0] && cfg.menuToolsLabel) {
      var toolsBtn = dropdowns[0].querySelector(".oe-pro-nav__trigger");
      if (toolsBtn) {
        toolsBtn.childNodes[0].textContent = cfg.menuToolsLabel + " ";
      }
    }
    if (dropdowns[1] && cfg.menuAudienceLabel) {
      var audBtn = dropdowns[1].querySelector(".oe-pro-nav__trigger");
      if (audBtn) {
        audBtn.childNodes[0].textContent = cfg.menuAudienceLabel + " ";
      }
    }

    var cta = document.querySelector(".oe-pro-nav__cta");
    if (cta) {
      if (cfg.ctaHref) cta.href = cfg.ctaHref;
      if (cfg.ctaLabel) cta.textContent = cfg.ctaLabel;
    }

    var tagline = document.querySelector('[data-nav-slot="footer-tagline"]');
    if (tagline && cfg.footerTagline) tagline.textContent = cfg.footerTagline;
    var email = document.querySelector('[data-nav-slot="footer-email"]');
    if (email && cfg.footerEmail) {
      email.href = "mailto:" + cfg.footerEmail;
      email.textContent = cfg.footerEmail;
    }
    var copyright = document.querySelector('[data-nav-slot="footer-copyright"]');
    if (copyright && cfg.footerCopyright) copyright.textContent = cfg.footerCopyright;

    var cols = document.querySelector('[data-nav-slot="footer-cols"]');
    if (cols && cfg.footerColumns && cfg.footerColumns.length) {
      cols.innerHTML = "";
      cfg.footerColumns.forEach(function (col) {
        var nav = document.createElement("nav");
        nav.className = "oe-pro-footer__col";
        nav.setAttribute("aria-label", col.title || "");
        var h3 = document.createElement("h3");
        h3.textContent = col.title || "";
        nav.appendChild(h3);
        (col.links || []).forEach(function (link) {
          if (!link || !link.label || link.hidden) return;
          var a = document.createElement("a");
          a.href = link.href || "#";
          a.textContent = link.label;
          nav.appendChild(a);
        });
        cols.appendChild(nav);
      });
    }

    markCurrentNav();
    if (global.OuviescreviNav && global.OuviescreviNav.init) {
      global.OuviescreviNav.init();
    }
    if (global.OuviescreviPricing && global.OuviescreviPricing.apply) {
      global.OuviescreviPricing.apply();
    }
  }

  function maybeApplyNavFromCache() {
    if (siteContentCache && siteContentCache.content) {
      applyNavConfig(siteContentCache.content);
    }
  }

  function applyCmsContent(content) {
    if (!content) return;
    document.querySelectorAll("[data-cms-key]").forEach(function (el) {
      var key = el.getAttribute("data-cms-key");
      var mode = el.getAttribute("data-cms-mode") || "html";
      var val = content[key];
      if (val == null || val === "") return;
      if (mode === "text") {
        el.textContent = val;
      } else if (mode === "lines") {
        var lines = val.split("\n").filter(Boolean);
        el.innerHTML = "<p><strong>Funcionalidades principais:</strong></p>" +
          lines.map(function () { return "<p></p>"; }).join("");
        var ps = el.querySelectorAll("p");
        lines.forEach(function (line, i) {
          if (ps[i + 1]) ps[i + 1].textContent = line;
        });
      } else {
        el.innerHTML = val;
      }
    });
  }

  function loadSiteConfig(onLoaded) {
    if ((global.location.pathname || "").indexOf("backoffice") !== -1) {
      return Promise.resolve(null);
    }
    var base = cmsApiBase();
    if (!base) return Promise.resolve(null);
    return fetch(base + "/api/site-content", { credentials: "omit" })
      .then(function (res) {
        if (!res.ok) return null;
        return res.json();
      })
      .then(function (data) {
        if (!data) return null;
        siteContentCache = data;
        applyNavConfig(data.content || {});
        if (document.querySelector("[data-cms-key]")) {
          applyCmsContent(data.content || {});
        }
        if (data.banner && data.banner.texto) showSiteBanner(data.banner);
        if (global.OuviescreviSEO && data.seo) global.OuviescreviSEO.applyOverrides(data.seo);
        if (typeof onLoaded === "function") onLoaded(data);
        return data;
      })
      .catch(function (err) {
        console.warn("OuviescreviUI: site config indisponível", err);
        return null;
      });
  }

  function loadCms(onLoaded) {
    return loadSiteConfig(onLoaded);
  }

  function showSiteBanner(banner) {
    if (!banner || !banner.texto) return;
    if (document.getElementById("oe-site-banner")) return;
    var el = document.createElement("div");
    el.id = "oe-site-banner";
    el.className = "oe-site-banner";
    if (banner.link) {
      el.innerHTML = '<a href="' + banner.link + '">' + banner.texto + "</a>";
    } else {
      el.textContent = banner.texto;
    }
    document.body.prepend(el);
  }

  function bootLayout() {
    if (global.OuviescreviSEO) global.OuviescreviSEO.apply();
    ensureI18nScript().then(function () {
      maybeShowCookieBanner();
    });
    ensureNavScript().then(function () {
      autoLoadLayout();
      setTimeout(markCurrentNav, 400);
      mountNewsTicker();
    });
    trackPageView();
    if (document.body && document.body.dataset.cmsAuto !== "false") {
      loadSiteConfig();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootLayout);
  } else {
    bootLayout();
  }

  global.OuviescreviUI = {
    toast: toast,
    setButtonLoading: setButtonLoading,
    markCurrentNav: markCurrentNav,
    loadHeader: loadHeader,
    loadFooter: loadFooter,
    applyCmsContent: applyCmsContent,
    applyNavConfig: applyNavConfig,
    loadCms: loadCms,
    loadSiteConfig: loadSiteConfig,
  };
})(window);
