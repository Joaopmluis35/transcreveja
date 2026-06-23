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

  function ensureNavScript() {
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
  }

  function resolveHeaderUrl(explicit) {
    if (explicit) return explicit;
    var el = document.getElementById("header");
    if (el && el.getAttribute("data-header-url")) {
      return el.getAttribute("data-header-url");
    }
    return "header.html";
  }

  function resolveFooterUrl(explicit) {
    if (explicit) return explicit;
    var el = document.getElementById("footer");
    if (el && el.getAttribute("data-footer-url")) {
      return el.getAttribute("data-footer-url");
    }
    return "footer.html";
  }

  function loadHeader(url) {
    url = resolveHeaderUrl(url);
    return ensureNavScript()
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
        markCurrentNav();
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
      })
      .catch(function (err) {
        console.error("OuviescreviUI: falha ao carregar", url, err);
      });
  }

  function autoLoadLayout() {
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
    try {
      if ((global.location.pathname || "").indexOf("backoffice") !== -1) return;
      var path = global.location.pathname || "/";
      var key = "oe_track_" + path;
      if (sessionStorage.getItem(key)) return;
      sessionStorage.setItem(key, "1");
    } catch (e) {
      return;
    }
    var base =
      global.OuviescreviAPI && global.OuviescreviAPI.detectApiBase
        ? global.OuviescreviAPI.detectApiBase()
        : null;
    if (!base) return;
    fetch(base + "/api/track-visit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      body: JSON.stringify({
        path: global.location.pathname || "/",
        referrer: document.referrer || "",
      }),
    }).catch(function () {});
  }

  function cookieBannerPath() {
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
    var en = (global.location.pathname || "").indexOf("/en/") !== -1;
    var banner = document.createElement("div");
    banner.id = "oe-cookie-banner";
    banner.className = "oe-cookie-banner";
    banner.setAttribute("role", "dialog");
    banner.setAttribute("aria-label", en ? "Cookie notice" : "Aviso de cookies");
    banner.innerHTML =
      (en
        ? 'We use essential browser storage for the service to work. '
        : "Utilizamos armazenamento essencial no browser para o serviço funcionar. ") +
      '<a href="' + cookieBannerPath() + '">' +
      (en ? "Cookie Policy" : "Política de Cookies") +
      "</a>";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "oe-cookie-banner__btn";
    btn.textContent = en ? "OK" : "Compreendi";
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

  function loadCms(onLoaded) {
    if ((global.location.pathname || "").indexOf("backoffice") !== -1) {
      return Promise.resolve(null);
    }
    if (!document.querySelector("[data-cms-key]")) {
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
        applyCmsContent(data.content || {});
        if (data.banner && data.banner.texto) showSiteBanner(data.banner);
        if (global.OuviescreviSEO && data.seo) global.OuviescreviSEO.applyOverrides(data.seo);
        if (typeof onLoaded === "function") onLoaded(data);
        return data;
      })
      .catch(function (err) {
        console.warn("OuviescreviUI: CMS indisponível", err);
        return null;
      });
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
    ensureNavScript().then(function () {
      autoLoadLayout();
      setTimeout(markCurrentNav, 400);
    });
    trackPageView();
    maybeShowCookieBanner();
    if (document.body && document.body.dataset.cmsAuto !== "false") {
      loadCms();
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
    loadCms: loadCms,
  };
})(window);
