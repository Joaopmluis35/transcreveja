/**
 * Tema claro/escuro do site público — persiste em localStorage (oe_site_theme).
 * Sem preferência guardada, segue o sistema (prefers-color-scheme).
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "oe_site_theme";

  function prefersDark() {
    try {
      return global.matchMedia("(prefers-color-scheme: dark)").matches;
    } catch (e) {
      return false;
    }
  }

  function savedTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function isDarkTheme(theme) {
    if (theme === "dark") return true;
    if (theme === "light") return false;
    return prefersDark();
  }

  function isBackofficePage() {
    return (global.location.pathname || "").indexOf("backoffice") !== -1;
  }

  function applyTheme(theme) {
    if (isBackofficePage()) return false;
    var dark = isDarkTheme(theme);
    document.documentElement.classList.toggle("oe-theme-dark", dark);
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
    var btn = document.getElementById("oeThemeToggle");
    if (btn) {
      btn.textContent = dark ? "☀️" : "🌙";
      btn.title = dark ? "Modo claro" : "Modo escuro";
      btn.setAttribute("aria-label", btn.title);
    }
    try {
      if (theme === "light" || theme === "dark") {
        localStorage.setItem(STORAGE_KEY, theme);
      }
    } catch (e) {}
    return dark;
  }

  function resolveTheme() {
    var saved = savedTheme();
    if (saved === "light" || saved === "dark") return saved;
    return prefersDark() ? "dark" : "light";
  }

  function initTheme() {
    if (isBackofficePage()) return;
    applyTheme(savedTheme() || (prefersDark() ? "dark" : "light"));
    var btn = document.getElementById("oeThemeToggle");
    if (btn && !btn.dataset.oeThemeBound) {
      btn.dataset.oeThemeBound = "1";
      btn.addEventListener("click", function () {
        var next = document.documentElement.classList.contains("oe-theme-dark") ? "light" : "dark";
        applyTheme(next);
      });
    }
    try {
      var mq = global.matchMedia("(prefers-color-scheme: dark)");
      if (mq && mq.addEventListener && !global.__oeThemeMqBound) {
        global.__oeThemeMqBound = true;
        mq.addEventListener("change", function () {
          if (!savedTheme()) applyTheme(prefersDark() ? "dark" : "light");
        });
      }
    } catch (e) {}
  }

  // Aplicar o mais cedo possível (antes do resto da UI).
  if (!isBackofficePage()) {
    applyTheme(savedTheme() || (prefersDark() ? "dark" : "light"));
  }

  global.OuviescreviTheme = {
    init: initTheme,
    apply: applyTheme,
    resolve: resolveTheme,
  };
})(window);
