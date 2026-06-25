/**
 * Login / registo no site público — envia X-Site-Session nas chamadas à API.
 */
(function (global) {
  const SITE_SESSION_KEY = "ouviescrevi_site_session";
  const SITE_ROLE_KEY = "ouviescrevi_site_role";
  const SITE_EMAIL_KEY = "ouviescrevi_site_email";
  const SITE_NAME_KEY = "ouviescrevi_site_name";

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
    var name = sessionStorage.getItem(SITE_NAME_KEY);
    var email = sessionStorage.getItem(SITE_EMAIL_KEY);
    var role = sessionStorage.getItem(SITE_ROLE_KEY) || "";
    if (staffRoles().indexOf(role) !== -1) {
      return (email || "Admin") + " (equipa)";
    }
    return name || email || "Conta";
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

  function ensureModal() {
    if (document.getElementById("oeAuthModal")) return document.getElementById("oeAuthModal");
    var wrap = document.createElement("div");
    wrap.id = "oeAuthModal";
    wrap.className = "oe-auth-modal hidden";
    wrap.setAttribute("role", "dialog");
    wrap.setAttribute("aria-modal", "true");
    wrap.setAttribute("aria-labelledby", "oeAuthModalTitle");
    wrap.innerHTML =
      '<div class="oe-auth-modal__backdrop" data-oe-auth-close="1"></div>' +
      '<div class="oe-auth-modal__card">' +
      '  <button type="button" class="oe-auth-modal__close" data-oe-auth-close="1" aria-label="Fechar">✕</button>' +
      '  <h2 id="oeAuthModalTitle" class="oe-auth-modal__title">Entrar na conta</h2>' +
      '  <div class="oe-auth-tabs" role="tablist">' +
      '    <button type="button" class="oe-auth-tabs__btn oe-auth-tabs__btn--active" data-oe-auth-tab="login" role="tab">Entrar</button>' +
      '    <button type="button" class="oe-auth-tabs__btn" data-oe-auth-tab="register" role="tab">Registar</button>' +
      '    <button type="button" class="oe-auth-tabs__btn" data-oe-auth-tab="admin" role="tab">Admin</button>' +
      "  </div>" +
      '  <form id="oeAuthLoginForm" class="oe-auth-form" data-oe-auth-panel="login">' +
      '    <label>Email<input type="email" name="email" required autocomplete="email" /></label>' +
      '    <label>Palavra-passe<input type="password" name="password" required minlength="8" autocomplete="current-password" /></label>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthLoginError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">Entrar</button>' +
      "  </form>" +
      '  <form id="oeAuthRegisterForm" class="oe-auth-form hidden" data-oe-auth-panel="register">' +
      '    <label>Nome (opcional)<input type="text" name="name" autocomplete="name" /></label>' +
      '    <label>Email<input type="email" name="email" required autocomplete="email" /></label>' +
      '    <label>Palavra-passe (mín. 8)<input type="password" name="password" required minlength="8" autocomplete="new-password" /></label>' +
      '    <p class="oe-auth-form__hint">Ao registares-te podes usar o site com a tua conta. Atividade normal envia notificação ao administrador.</p>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthRegisterError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">Criar conta</button>' +
      "  </form>" +
      '  <form id="oeAuthAdminForm" class="oe-auth-form hidden" data-oe-auth-panel="admin">' +
      '    <label>Utilizador<input type="text" name="email" value="admin" autocomplete="username" /></label>' +
      '    <label>Palavra-passe<input type="password" name="password" required autocomplete="current-password" /></label>' +
      '    <p class="oe-auth-form__hint">Conta de equipa — atividade no site não envia emails de notificação.</p>' +
      '    <p class="oe-auth-form__error hidden" id="oeAuthAdminError"></p>' +
      '    <button type="submit" class="oe-pro-btn oe-pro-btn--primary">Entrar como admin</button>' +
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
    var titles = { login: "Entrar na conta", register: "Criar conta", admin: "Entrar como administrador" };
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
      clearSession();
      refreshChrome();
      if (global.OuviescreviUI && global.OuviescreviUI.toast) {
        global.OuviescreviUI.toast("Sessão terminada.", "info");
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
        if (!res.ok) throw new Error(data.detail || "Não foi possível entrar.");
        persistSession(data);
        closeModal();
        refreshChrome();
        if (global.OuviescreviUI && global.OuviescreviUI.toast) {
          global.OuviescreviUI.toast("Bem-vindo de volta!", "success");
        }
      } catch (err) {
        showError("oeAuthLoginError", err.message || "Erro ao entrar.");
      }
    });

    document.getElementById("oeAuthRegisterForm")?.addEventListener("submit", async function (e) {
      e.preventDefault();
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
        if (!res.ok) throw new Error(data.detail || "Não foi possível registar.");
        persistSession(data);
        closeModal();
        refreshChrome();
        if (global.OuviescreviUI && global.OuviescreviUI.toast) {
          global.OuviescreviUI.toast("Conta criada — 20 transcrições por dia!", "success");
        }
      } catch (err) {
        showError("oeAuthRegisterError", err.message || "Erro ao registar.");
      }
    });

    document.getElementById("oeAuthAdminForm")?.addEventListener("submit", async function (e) {
      e.preventDefault();
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
        if (!res.ok) throw new Error(data.detail || "Credenciais inválidas.");
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
          global.OuviescreviUI.toast("Sessão de administrador ativa.", "success");
        }
      } catch (err) {
        showError("oeAuthAdminError", err.message || "Erro ao entrar.");
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
