/**
 * Cliente API Ouviescrevi — token e base URL vêm do servidor (.env), não do HTML.
 */
(function (global) {
  const DEFAULT_API = "https://api.ouviescrevi.pt";
  const TOKEN_KEY = "ouviescrevi_api_token";
  const ADMIN_KEY = "ouviescrevi_admin_token";

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
    initPromise = (async () => {
      apiBase = detectApiBase();
      if (!forceRefresh) {
        const cached = sessionStorage.getItem(TOKEN_KEY);
        if (cached) {
          apiToken = cached;
          return { apiBase, token: apiToken };
        }
      }
      const res = await fetch(`${apiBase}/api/frontend-config`, {
        method: "GET",
        credentials: "omit",
      });
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
    return initPromise;
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

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    const token = getToken();
    if (token) {
      headers.Authorization = `Bearer ${token}`;
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
    apiBase = base;
    return data;
  }

  function adminLogout() {
    sessionStorage.removeItem(ADMIN_KEY);
    sessionStorage.removeItem("ouviescrevi_admin_ok");
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
