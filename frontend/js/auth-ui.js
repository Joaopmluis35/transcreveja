/**
 * Login / registo no site público — envia X-Site-Session nas chamadas à API.
 */
(function (global) {
  const SITE_SESSION_KEY = "ouviescrevi_site_session";
  const SITE_ROLE_KEY = "ouviescrevi_site_role";
  const SITE_EMAIL_KEY = "ouviescrevi_site_email";
  const SITE_NAME_KEY = "ouviescrevi_site_name";

  var AUTH_FALLBACK = {
    pt: {
      close: "Fechar",
      titleLogin: "Entrar na conta",
      titleRegister: "Criar conta",
      titleAdmin: "Entrar como administrador",
      tabLogin: "Entrar",
      tabRegister: "Registar",
      tabAdmin: "Admin",
      email: "Email",
      password: "Palavra-passe",
      passwordMin: "Palavra-passe (mín. 8)",
      nameOptional: "Nome (opcional)",
      username: "Utilizador",
      loginBtn: "Entrar",
      registerBtn: "Criar conta",
      adminBtn: "Entrar como admin",
      registerHint: "Ao registares-te podes usar o site com a tua conta. Atividade normal envia notificação ao administrador.",
      adminHint: "Conta de equipa — atividade no site não envia emails de notificação.",
      accountLabel: "Conta",
      staffSuffix: " (equipa)",
      logoutToast: "Sessão terminada.",
      welcomeBack: "Bem-vindo de volta!",
      accountCreated: "Conta criada — 20 transcrições por dia!",
      adminSession: "Sessão de administrador ativa.",
      loginFail: "Não foi possível entrar.",
      registerFail: "Não foi possível registar.",
      invalidCreds: "Credenciais inválidas.",
      loginError: "Erro ao entrar.",
      registerError: "Erro ao registar.",
    },
  };

  function locale() {
    if (global.OuviescreviI18n) return global.OuviescreviI18n.localeFromPath();
    var m = (global.location && global.location.pathname || "").match(/^\/(en|es|fr|de)(\/|$)/);
    return m ? m[1] : "pt";
  }

  function t() {
    if (global.OuviescreviI18n && global.OuviescreviI18n.authStrings) {
      return global.OuviescreviI18n.authStrings(locale());
    }
    return AUTH_FALLBACK.pt;
  }

  function staffRoles() {
    return ["admin", "editor", "viewer"];
  }

  function persistSession(data) {
    if (!data || !data.sessionToken) return;
    sessionStorage.setItem(SITE_SESSION_KEY, data.sessionToken);
    sessionStorage.setItem(SITE_ROLE_KEY, data.role || "user");
    if (data.email) sessionStorage.setItem(SITE_EMAIL_KEY, data.email);
    else sessionStorage.removeItem(SITE_EMAIL_KEY);
    if (data.username) sessionStorage.setItem(SITE_EMAIL_KEY, data.username);
    if (data.name) sessionStorage.setItem(SITE_NAME_KEY, data.name);
    else sessionStorage.removeItem(SITE_NAME_KEY);
    if (data.isStaff && global.OuviescreviAPI && global.OuviescreviAPI.syncAdminFromSiteSession) {
      global.OuviescreviAPI.syncAdminFromSiteSession(data);
    }
  }

  function clearSession() {
    sessionStorage.removeItem(SITE_SESSION_KEY);
    sessionStorage.removeItem(SITE_ROLE_KEY);
    sessionStorage.removeItem(SITE_EMAIL_KEY);
    sessionStorage.removeItem(SITE_NAME_KEY);
    if (global.OuviescreviAPI && global.OuviescreviAPI.adminLogout) {
      global.OuviescreviAPI.adminLogout();
    }
  }

  function getDisplayLabel() {
    var strings = t();
    var name = sessionStorage.getItem(SITE_NAME_KEY);
    var email = sessionStorage.getItem(SITE_EMAIL_KEY);
    var role = sessionStorage.getItem(SITE_ROLE_KEY) || "";
    if (staffRoles().indexOf(role) !== -1) {
      return (email || "Admin") + strings.staffSuffix;
    }
    return name || email || strings.accountLabel;
  }

  function isLoggedIn() {
    return !!(sessionStorage.getItem(SITE_SESSION_KEY) || (global.OuviescreviAPI && global.OuviescreviAPI.getAdminToken && global.OuviescreviAPI.getAdminToken()));
  }

  function isStaff() {
    var role = sessionStorage.getItem(SITE_ROLE_KEY);
    if (staffRoles().indexOf(role) !== -1) return true;
    if (global.OuviescreviAPI && global.OuviescreviAPI.isAdminSession && global.OuviescreviAPI.isAdminSession()) return true;
    return false;
  }

  function applyModalStrings(modal) {
    if (!modal) return;
    var s = t();
    var closeBtn = modal.querySelector(".oe-auth-modal__close");
    if (closeBtn) closeBtn.setAttribute("aria-label", s.close);
    modal.querySelector('[data-oe-auth-tab="login"]').textContent = s.tabLogin;
    modal.querySelector('[data-oe-auth-tab="register"]').textContent = s.tabRegister;
    modal.querySelector('[data-oe-auth-tab="admin"]').textContent = s.tabAdmin;

    var loginForm = modal.querySelector('[data-oe-auth-panel="login"]');
    if (loginForm) {
      loginForm.querySelector("label:nth-of-type(1)").childNodes[0].textContent = s.email;
      loginForm.querySelector("label:nth-of-type(2)").childNodes[0].textContent = s.password;
      loginForm.querySelector('button[type="submit"]').textContent = s.loginBtn;
    }
    var regForm = modal.querySelector('[data-oe-auth-panel="register"]');
    if (regForm) {
      regForm.querySelector("label:nth-of-type(1)").childNodes[0].textContent = s.nameOptional;
      regForm.querySelector("label:nth-of-type(2)").childNodes[0].textContent = s.email;
      regForm.querySelector("label:nth-of-type(3)").childNodes[0].textContent = s.passwordMin;
      regForm.querySelector(".oe-auth-form__hint").textContent = s.registerHint;
      regForm.querySelector('button[type="submit"]').textContent = s.registerBtn;
    }
    var adminForm = modal.querySelector('[data-oe-auth-panel="admin"]');
    if (adminForm) {
      adminForm.querySelector("label:nth-of-type(1)").childNodes[0].textContent = s.username;
      adminForm.querySelector("label:nth-of-type(2)").childNodes[0].textContent = s.password;
      adminForm.querySelector(".oe-auth-form__hint").textContent = s.adminHint;
      adminForm.querySelector('button[type="submit"]').textContent = s.adminBtn;
    }
  }

  function ensureModal() {
    if (document.getElementById("oeAuthModal")) {
      applyModalStrings(document.getElementById("oeAuthModal"));
      return document.getElementById("oeAuthModal");
    }
    var s = t();
    var wrap = document.createElement("div");
    wrap.id = "oeAuthModal";
    wrap.className = "oe-auth-modal hidden";
    wrap.setAttribute("role", "dialog");
    wrap.setAttribute("aria-modal", "true");
    wrap.setAttribute("aria-labelledby", "oeAuthModalTitle");
    wrap.innerHTML =
      '<div class="oe-auth-modal__backdrop" data-oe-auth-close="1"></div>' +
      '<div class="oe-auth-modal__card">' +
      '  <button type="button" class="oe-auth-modal__close" data-oe-auth-close="1" aria-label="' + s.close + '">✕</button>' +
      '  <h2 id="oeAuthModalTitle" class="oe-auth-modal__title">' + s.titleLogin + '</h2>' +
      '  <div class="oe-auth-tabs" role="tablist">' +
      '    <button type="button" class="oe-auth-tabs__btn oe-auth-tabs__btn--active" data-oe-auth-tab="login" role="tab">' + s.tabLogin + '</button>' +
      '    <button type="button" class="oe-auth-tabs__btn" data-oe-auth-tab="register" role="tab">' + s.tabRegister + '</button>' +
      '    <button type="button" class="oe-auth-tabs__btn" data-oe-auth-tab="admin" role="tab">' + s.tabAdmin + '</button>' +
      "  </div>" +
      '  <form id="oeAuthLoginForm" class="oe-auth-form" data-oe-auth-panel="login">' +
      '    <label>' + s.email + '<input type="email" name="email" required autocomplete="email" /></label>' +
      '    <label>' + s.password + '<input type="password" name="password" required minlength="8" autocomplete="current-password" /></label>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthLoginError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">' + s.loginBtn + '</button>' +
      "  </form>" +
      '  <form id="oeAuthRegisterForm" class="oe-auth-form hidden" data-oe-auth-panel="register">' +
      '    <label>' + s.nameOptional + '<input type="text" name="name" autocomplete="name" /></label>' +
      '    <label>' + s.email + '<input type="email" name="email" required autocomplete="email" /></label>' +
      '    <label>' + s.passwordMin + '<input type="password" name="password" required minlength="8" autocomplete="new-password" /></label>' +
      '    <p class="oe-auth-form__hint">' + s.registerHint + '</p>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthRegisterError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">' + s.registerBtn + '</button>' +
      "  </form>" +
      '  <form id="oeAuthAdminForm" class="oe-auth-form hidden" data-oe-auth-panel="admin">' +
      '    <label>' + s.username + '<input type="text" name="email" value="admin" autocomplete="username" /></label>' +
      '    <label>' + s.password + '<input type="password" name="password" required autocomplete="current-password" /></label>' +
      '    <p class="oe-auth-form__hint">' + s.adminHint + '</p>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthAdminError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">' + s.adminBtn + '</button>' +
      "  </form>" +
      "</div>";
    document.body.appendChild(wrap);
    return wrap;
  }

  function openModal(tab) {
    var modal = ensureModal();
    modal.classList.remove("hidden");
    setTab(tab || "login");
    document.body.classList.add("oe-auth-modal-open");
  }

  function closeModal() {
    var modal = document.getElementById("oeAuthModal");
    if (modal) modal.classList.add("hidden");
    document.body.classList.remove("oe-auth-modal-open");
  }

  function setTab(tab) {
    var modal = document.getElementById("oeAuthModal");
    if (!modal) return;
    modal.querySelectorAll("[data-oe-auth-tab]").forEach(function (btn) {
      btn.classList.toggle("oe-auth-tabs__btn--active", btn.getAttribute("data-oe-auth-tab") === tab);
    });
    modal.querySelectorAll("[data-oe-auth-panel]").forEach(function (panel) {
      panel.classList.toggle("hidden", panel.getAttribute("data-oe-auth-panel") !== tab);
    });
    var s = t();
    var titles = { login: s.titleLogin, register: s.titleRegister, admin: s.titleAdmin };
    var title = document.getElementById("oeAuthModalTitle");
    if (title) title.textContent = titles[tab] || titles.login;
  }

  function showError(id, msg) {
    var el = document.getElementById(id);
    if (!el) return;
    if (msg) {
      el.textContent = msg;
      el.classList.remove("hidden");
    } else {
      el.textContent = "";
      el.classList.add("hidden");
    }
  }

  async function apiBase() {
    if (global.OuviescreviAPI) {
      await global.OuviescreviAPI.init();
      return global.OuviescreviAPI.getBase();
    }
    return global.OUVIESCREVI_API_BASE || "https://api.ouviescrevi.pt";
  }

  function refreshChrome() {
    document.dispatchEvent(new CustomEvent("oe-auth-change"));
    var userEl = document.getElementById("oeAuthUser");
    var loginBtn = document.getElementById("oeAuthLogin");
    var regBtn = document.getElementById("oeAuthRegister");
    var logoutBtn = document.getElementById("oeAuthLogout");
    var logged = isLoggedIn();
    if (userEl) {
      userEl.textContent = logged ? getDisplayLabel() : "";
      userEl.classList.toggle("hidden", !logged);
    }
    if (loginBtn) loginBtn.classList.toggle("hidden", logged);
    if (regBtn) regBtn.classList.toggle("hidden", logged);
    if (logoutBtn) logoutBtn.classList.toggle("hidden", !logged);
  }

  function bindChrome() {
    document.getElementById("oeAuthLogin")?.addEventListener("click", function () {
      openModal("login");
    });
    document.getElementById("oeAuthRegister")?.addEventListener("click", function () {
      openModal("register");
    });
    document.getElementById("oeAuthLogout")?.addEventListener("click", function () {
      var s = t();
      clearSession();
      refreshChrome();
      if (global.OuviescreviUI && global.OuviescreviUI.toast) {
        global.OuviescreviUI.toast(s.logoutToast, "info");
      }
    });

    var modal = ensureModal();
    modal.addEventListener("click", function (e) {
      if (e.target.closest("[data-oe-auth-close]")) closeModal();
    });
    modal.querySelectorAll("[data-oe-auth-tab]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        setTab(btn.getAttribute("data-oe-auth-tab"));
      });
    });

    document.getElementById("oeAuthLoginForm")?.addEventListener("submit", async function (e) {
      e.preventDefault();
      var s = t();
      showError("oeAuthLoginError", "");
      var fd = new FormData(e.target);
      try {
        var base = await apiBase();
        var res = await fetch(base + "/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: fd.get("email"), password: fd.get("password"), admin: false }),
        });
        var data = await res.json().catch(function () { return {}; });
        if (!res.ok) throw new Error(data.detail || s.loginFail);
        persistSession(data);
        closeModal();
        refreshChrome();
        if (global.OuviescreviUI && global.OuviescreviUI.toast) {
          global.OuviescreviUI.toast(s.welcomeBack, "success");
        }
      } catch (err) {
        showError("oeAuthLoginError", err.message || s.loginError);
      }
    });

    document.getElementById("oeAuthRegisterForm")?.addEventListener("submit", async function (e) {
      e.preventDefault();
      var s = t();
      showError("oeAuthRegisterError", "");
      var fd = new FormData(e.target);
      try {
        var base = await apiBase();
        var res = await fetch(base + "/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: fd.get("email"),
            password: fd.get("password"),
            name: fd.get("name") || null,
          }),
        });
        var data = await res.json().catch(function () { return {}; });
        if (!res.ok) throw new Error(data.detail || s.registerFail);
        persistSession(data);
        closeModal();
        refreshChrome();
        if (global.OuviescreviUI && global.OuviescreviUI.toast) {
          global.OuviescreviUI.toast(s.accountCreated, "success");
        }
      } catch (err) {
        showError("oeAuthRegisterError", err.message || s.registerError);
      }
    });

    document.getElementById("oeAuthAdminForm")?.addEventListener("submit", async function (e) {
      e.preventDefault();
      var s = t();
      showError("oeAuthAdminError", "");
      var fd = new FormData(e.target);
      try {
        var base = await apiBase();
        var res = await fetch(base + "/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: fd.get("email") || "admin",
            password: fd.get("password"),
            admin: true,
          }),
        });
        var data = await res.json().catch(function () { return {}; });
        if (!res.ok) throw new Error(data.detail || s.invalidCreds);
        persistSession(data);
        if (global.OuviescreviAPI) {
          sessionStorage.setItem("ouviescrevi_admin_token", data.sessionToken);
          sessionStorage.setItem("ouviescrevi_admin_ok", "true");
          sessionStorage.setItem("ouviescrevi_admin_role", data.role || "admin");
          sessionStorage.setItem("ouviescrevi_admin_username", data.username || "admin");
        }
        closeModal();
        refreshChrome();
        if (global.OuviescreviUI && global.OuviescreviUI.toast) {
          global.OuviescreviUI.toast(s.adminSession, "success");
        }
      } catch (err) {
        showError("oeAuthAdminError", err.message || s.loginError);
      }
    });
  }

  function init() {
    ensureModal();
    bindChrome();
    refreshChrome();
  }

  global.OuviescreviAuth = {
    init,
    openModal,
    closeModal,
    refreshChrome,
    isLoggedIn,
    isStaff,
    clearSession,
    SITE_SESSION_KEY,
  };
})(window);
