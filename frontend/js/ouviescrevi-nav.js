/**
 * Menu interativo Ouviescrevi — event delegation (funciona após fetch do header).
 */
(function (global) {
  "use strict";

  var BOUND = false;

  function isEn() {
    return (global.location.pathname || "").indexOf("/en/") !== -1;
  }

  function pageId() {
    var path = global.location.pathname.replace(/\/$/, "");
    var file = path.split("/").pop() || "index.html";
    return file.replace(".html", "") || "index";
  }

  function headerEl() {
    return document.getElementById("oeProHeader");
  }

  function closeAllDropdowns(except) {
    document.querySelectorAll(".oe-pro-nav__dropdown.is-open").forEach(function (el) {
      if (except && el === except) return;
      el.classList.remove("is-open");
      var btn = el.querySelector(".oe-pro-nav__trigger");
      if (btn) btn.setAttribute("aria-expanded", "false");
    });
  }

  function closeMobileNav() {
    var header = headerEl();
    if (!header) return;
    header.classList.remove("is-mobile-open");
    var toggle = header.querySelector(".oe-pro-nav__mobile-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", isEn() ? "Open menu" : "Abrir menu");
    }
    document.body.classList.remove("oe-nav-open");
  }

  function openMobileNav() {
    var header = headerEl();
    if (!header) return;
    header.classList.add("is-mobile-open");
    var toggle = header.querySelector(".oe-pro-nav__mobile-toggle");
    if (toggle) {
      toggle.setAttribute("aria-expanded", "true");
      toggle.setAttribute("aria-label", isEn() ? "Close menu" : "Fechar menu");
    }
    document.body.classList.add("oe-nav-open");
    closeAllDropdowns();
  }

  function toggleMobileNav() {
    var header = headerEl();
    if (!header) return;
    if (header.classList.contains("is-mobile-open")) closeMobileNav();
    else openMobileNav();
  }

  function toggleDropdown(dropdown) {
    if (!dropdown) return;
    var open = dropdown.classList.contains("is-open");
    closeAllDropdowns();
    if (!open) {
      dropdown.classList.add("is-open");
      var btn = dropdown.querySelector(".oe-pro-nav__trigger");
      if (btn) btn.setAttribute("aria-expanded", "true");
    }
  }

  function markCurrentPage() {
    var current = pageId();
    document.querySelectorAll(".oe-pro-nav__link.is-active, .oe-pro-nav__trigger.is-active").forEach(function (el) {
      el.classList.remove("is-active");
    });
    document.querySelectorAll("[data-nav-page]").forEach(function (el) {
      var p = el.getAttribute("data-nav-page");
      if (p === current) {
        el.classList.add("is-active");
        var dropdown = el.closest(".oe-pro-nav__dropdown");
        if (dropdown) {
          var trigger = dropdown.querySelector(".oe-pro-nav__trigger");
          if (trigger) trigger.classList.add("is-active");
        }
      } else {
        el.classList.remove("is-active");
      }
    });
  }

  function syncLangFlag() {
    var btn = document.getElementById("oeLangBtn");
    if (!btn) return;
    try {
      var lang = localStorage.getItem("lang") || (isEn() ? "en" : "pt");
      var img = btn.querySelector("img");
      if (img) {
        img.src = "/icons/" + lang + ".png";
        img.alt = lang.toUpperCase();
      }
    } catch (e) {}
  }

  function setLanguage(lang) {
    try { localStorage.setItem("lang", lang); } catch (e) {}
    var path = global.location.pathname;
    var inEn = path.indexOf("/en/") !== -1;
    if (lang === "en" && !inEn) {
      global.location.href = "/en/index.html";
    } else if (lang === "pt" && inEn) {
      global.location.href = "/index.html";
    } else {
      global.location.reload();
    }
  }

  function onDocumentClick(e) {
    var header = headerEl();
    if (!header) return;

    if (e.target.closest(".oe-pro-nav__backdrop")) {
      closeMobileNav();
      return;
    }

    if (e.target.closest(".oe-pro-nav__mobile-toggle")) {
      e.preventDefault();
      toggleMobileNav();
      return;
    }

    if (e.target.closest(".oe-pro-nav__trigger")) {
      e.preventDefault();
      e.stopPropagation();
      var dropdown = e.target.closest(".oe-pro-nav__dropdown");
      if (global.innerWidth < 900) {
        if (dropdown) {
          dropdown.classList.toggle("is-open");
          var trigger = dropdown.querySelector(".oe-pro-nav__trigger");
          if (trigger) {
            trigger.setAttribute(
              "aria-expanded",
              dropdown.classList.contains("is-open") ? "true" : "false"
            );
          }
        }
      } else {
        toggleDropdown(dropdown);
      }
      return;
    }

    if (e.target.closest("#oeLangBtn")) {
      e.preventDefault();
      e.stopPropagation();
      var menu = document.getElementById("oeLangMenu");
      var langBtn = document.getElementById("oeLangBtn");
      if (menu && langBtn) {
        var open = menu.classList.toggle("is-open");
        langBtn.setAttribute("aria-expanded", open ? "true" : "false");
      }
      return;
    }

    var langItem = e.target.closest("[data-lang]");
    if (langItem && langItem.closest("#oeLangMenu")) {
      e.preventDefault();
      setLanguage(langItem.getAttribute("data-lang"));
      return;
    }

    if (e.target.closest(".oe-pro-nav__menu a, .oe-pro-nav__link, .oe-pro-nav__cta")) {
      closeMobileNav();
      closeAllDropdowns();
      return;
    }

    if (!e.target.closest(".oe-pro-nav__dropdown") && !e.target.closest(".oe-pro-lang")) {
      closeAllDropdowns();
      var langMenu = document.getElementById("oeLangMenu");
      var langBtn = document.getElementById("oeLangBtn");
      if (langMenu) langMenu.classList.remove("is-open");
      if (langBtn) langBtn.setAttribute("aria-expanded", "false");
    }
    if (!header.contains(e.target)) {
      closeMobileNav();
    }
  }

  function onKeydown(e) {
    if (e.key === "Escape") {
      closeAllDropdowns();
      closeMobileNav();
    }
  }

  function onResize() {
    if (global.innerWidth >= 900) {
      closeMobileNav();
      bindDesktopHover();
    }
  }

  function bindGlobal() {
    if (BOUND) return;
    BOUND = true;
    document.addEventListener("click", onDocumentClick);
    document.addEventListener("keydown", onKeydown);
    global.addEventListener("resize", onResize);
  }

  function bindDesktopHover() {
    if (global.innerWidth < 900) return;
    document.querySelectorAll(".oe-pro-nav__dropdown").forEach(function (dropdown) {
      if (dropdown.dataset.hoverBound) return;
      dropdown.dataset.hoverBound = "1";
      var closeTimer;
      dropdown.addEventListener("mouseenter", function () {
        clearTimeout(closeTimer);
        closeAllDropdowns(dropdown);
        dropdown.classList.add("is-open");
        var btn = dropdown.querySelector(".oe-pro-nav__trigger");
        if (btn) btn.setAttribute("aria-expanded", "true");
      });
      dropdown.addEventListener("mouseleave", function () {
        closeTimer = setTimeout(function () {
          dropdown.classList.remove("is-open");
          var btn = dropdown.querySelector(".oe-pro-nav__trigger");
          if (btn) btn.setAttribute("aria-expanded", "false");
        }, 120);
      });
    });
  }

  function init() {
    bindGlobal();
    if (!headerEl()) return;
    bindDesktopHover();
    syncLangFlag();
    markCurrentPage();
  }

  global.OuviescreviNav = {
    init: init,
    markCurrentPage: markCurrentPage,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  global.addEventListener("load", init);
})(window);
