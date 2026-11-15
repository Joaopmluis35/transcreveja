/**
 * Cliente API Ouviescrevi — token e base URL vêm do servidor (.env), não do HTML.
 */
(function (global) {
  const DEFAULT_API = "https://api.ouviescrevi.pt";
  const TOKEN_KEY = "ouviescrevi_api_token";
  const ADMIN_KEY = "ouviescrevi_admin_token";
  const SITE_SESSION_KEY = "ouviescrevi_site_session";

  let apiBase = "";
  let apiToken = null;
  let maxFileSizeMb = 500;
  let initPromise = null;

  function detectApiBase() {
    if (global.OUVIESCREVI_API_BASE) {
      return global.OUVIESCREVI_API_BASE.replace(/\/$/, "");
    }
    const meta = document.querySelector('meta[name="ouviescrevi-api-base"]');
    if (meta && meta.content) {
      return meta.content.replace(/\/$/, "");
    }
    const host = global.location && global.location.hostname;
    if (host === "localhost" || host === "127.0.0.1") {
      return "http://127.0.0.1:8000";
    }
    return DEFAULT_API;
  }

  async function init(forceRefresh) {
    if (initPromise && !forceRefresh) {
      return initPromise;
    }
    const attempt = (async () => {
      apiBase = detectApiBase();
      if (!forceRefresh) {
        const cached = sessionStorage.getItem(TOKEN_KEY);
        if (cached) {
          apiToken = cached;
          // O servidor adormece ao fim de uns minutos sem visitas: acorda-o já, sem esperar.
          fetch(`${apiBase}/api/frontend-config`, { method: "GET", credentials: "omit" }).catch(function () {});
          return { apiBase, token: apiToken };
        }
      }
      // Se a resposta demorar (servidor a acordar, até ~1 min), avisa em vez de parecer parado.
      const wakeTimer = setTimeout(showWakeNotice, 4000);
      let res;
      try {
        res = await fetch(`${apiBase}/api/frontend-config`, {
          method: "GET",
          credentials: "omit",
        });
      } finally {
        clearTimeout(wakeTimer);
        hideWakeNotice();
      }
      if (!res.ok) {
        throw new Error(`Configuração da API indisponível (${res.status})`);
      }
      const data = await res.json();
      apiBase = (data.apiBase || apiBase).replace(/\/$/, "");
      apiToken = data.token;
      if (data.maxFileSizeMb) {
        maxFileSizeMb = Number(data.maxFileSizeMb) || maxFileSizeMb;
      }
      if (apiToken) {
        sessionStorage.setItem(TOKEN_KEY, apiToken);
      }
      return data;
    })();
    initPromise = attempt;
    // Uma falha (rede, servidor a acordar) não pode ficar guardada: a tentativa seguinte volta a ligar.
    attempt.catch(function () {
      if (initPromise === attempt) initPromise = null;
    });
    return attempt;
  }

  const WAKE_TEXT = {
    pt: "A ligar ao servidor… na primeira utilização pode demorar até 1 minuto.",
    en: "Connecting to the server… the first use can take up to 1 minute.",
    es: "Conectando con el servidor… la primera vez puede tardar hasta 1 minuto.",
    fr: "Connexion au serveur… la première utilisation peut prendre jusqu’à 1 minute.",
    de: "Verbindung zum Server… beim ersten Mal kann es bis zu 1 Minute dauern.",
  };

  function showWakeNotice() {
    if (!document.body || document.getElementById("oe-wake-notice")) return;
    const lang = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    const box = document.createElement("div");
    box.id = "oe-wake-notice";
    box.setAttribute("role", "status");
    box.textContent = WAKE_TEXT[lang] || WAKE_TEXT.pt;
    box.style.cssText =
      "position:fixed;left:50%;top:14px;transform:translateX(-50%);z-index:9999;max-width:calc(100vw - 24px);" +
      "padding:10px 16px;border-radius:999px;background:#1f1b3a;color:#fff;border:1px solid #7c5cff;" +
      "font:600 14px/1.35 system-ui,sans-serif;box-shadow:0 8px 28px rgba(0,0,0,.45);text-align:center";
    document.body.appendChild(box);
  }

  function hideWakeNotice() {
    const box = document.getElementById("oe-wake-notice");
    if (box) box.remove();
  }

  function getBase() {
    return apiBase || detectApiBase();
  }

  function getToken() {
    return apiToken || sessionStorage.getItem(TOKEN_KEY);
  }

  function getAdminToken() {
    return sessionStorage.getItem(ADMIN_KEY);
  }

  function getSiteSessionToken() {
    function read(key) {
      try {
        return localStorage.getItem(key) || sessionStorage.getItem(key);
      } catch (e) {
        try {
          return sessionStorage.getItem(key);
        } catch (e2) {
          return null;
        }
      }
    }
    return read(SITE_SESSION_KEY) || read(ADMIN_KEY) || null;
  }

  function syncAdminFromSiteSession(data) {
    if (!data || !data.isStaff || !data.sessionToken) return;
    sessionStorage.setItem(ADMIN_KEY, data.sessionToken);
    sessionStorage.setItem("ouviescrevi_admin_ok", "true");
    if (data.role) sessionStorage.setItem("ouviescrevi_admin_role", data.role);
    if (data.username) sessionStorage.setItem("ouviescrevi_admin_username", data.username);
  }

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    const token = getToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    const siteSession = getSiteSessionToken();
    if (siteSession) {
      headers["X-Site-Session"] = siteSession;
    }
    return headers;
  }

  function adminAuthHeaders(extra) {
    const headers = Object.assign({ "Content-Type": "application/json" }, extra || {});
    const token = getAdminToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }
    return headers;
  }

  function authJson(body) {
    const payload = Object.assign({}, body || {});
    const token = getToken();
    if (token && !payload.token) {
      payload.token = token;
    }
    return payload;
  }

  function toAbsUrl(path) {
    if (!path) return "";
    if (/^https?:\/\//i.test(path)) return path;
    const base = getBase();
    return `${base}${path.startsWith("/") ? "" : "/"}${path}`;
  }

  async function fetchApi(path, options) {
    await init();
    const url = toAbsUrl(path);
    const opts = Object.assign({}, options || {});
    opts.headers = authHeaders(opts.headers || {});
    return fetch(url, opts);
  }

  async function adminLogin(password, username) {
    const base = detectApiBase();
    const body = { password };
    if (username && String(username).trim()) body.username = String(username).trim();
    const res = await fetch(`${base}/api/admin/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Credenciais inválidas.");
    }
    sessionStorage.setItem(ADMIN_KEY, data.adminToken);
    sessionStorage.setItem("ouviescrevi_admin_ok", "true");
    if (data.role) sessionStorage.setItem("ouviescrevi_admin_role", data.role);
    if (data.username) sessionStorage.setItem("ouviescrevi_admin_username", data.username);
    apiBase = base;
    return data;
  }

  function adminLogout() {
    sessionStorage.removeItem(ADMIN_KEY);
    sessionStorage.removeItem("ouviescrevi_admin_ok");
    sessionStorage.removeItem("ouviescrevi_admin_role");
    sessionStorage.removeItem("ouviescrevi_admin_username");
  }

  function isAdminSession() {
    return sessionStorage.getItem("ouviescrevi_admin_ok") === "true" && !!getAdminToken();
  }

  function getMaxFileSizeMb() {
    return maxFileSizeMb;
  }

  global.OuviescreviAPI = {
    init,
    getBase,
    getToken,
    getMaxFileSizeMb,
    getAdminToken,
    getSiteSessionToken,
    syncAdminFromSiteSession,
    authHeaders,
    adminAuthHeaders,
    authJson,
    toAbsUrl,
    fetchApi,
    adminLogin,
    adminLogout,
    isAdminSession,
    detectApiBase,
  };
})(window);
