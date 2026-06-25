/**
 * Backoffice — gestão de planos Pro e Stripe.
 */
(function (global) {
  var lastBillingData = null;

  function apiBase() {
    return global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase();
  }

  function authHeaders() {
    return global.OuviescreviAPI.adminAuthHeaders({ "Content-Type": "application/json" });
  }

  function billingConfigSlice(cfg) {
    cfg = cfg || {};
    return {
      billing_enabled: cfg.billing_enabled || "0",
      pricing_hidden: cfg.pricing_hidden !== "0" ? "1" : "0",
      stripe_public_key: cfg.stripe_public_key || "",
      stripe_price_id_pro: cfg.stripe_price_id_pro || "",
      pro_quota_daily: cfg.pro_quota_daily || "200",
      pro_price_label: cfg.pro_price_label || "9,99 €/mês",
      quota_anonymous_daily: cfg.quota_anonymous_daily || "3",
      quota_registered_daily: cfg.quota_registered_daily || "20",
      stripe_secret_set: Boolean(cfg.stripe_secret_set || cfg.stripe_secret_key),
      stripe_webhook_set: Boolean(cfg.stripe_webhook_set || cfg.stripe_webhook_secret),
    };
  }

  function applyQuotaFields(cfg) {
    var quotaAnon = document.getElementById("billingCfgQuotaAnon");
    var quotaReg = document.getElementById("billingCfgQuotaReg");
    if (quotaAnon) quotaAnon.value = cfg.quota_anonymous_daily || "3";
    if (quotaReg) quotaReg.value = cfg.quota_registered_daily || "20";
  }

  function syncSistemaQuotaFields(cfg) {
    var elAnon = document.getElementById("cfgQuotaAnon");
    var elReg = document.getElementById("cfgQuotaReg");
    if (elAnon) elAnon.value = cfg.quota_anonymous_daily || "3";
    if (elReg) elReg.value = cfg.quota_registered_daily || "20";
  }

  function mergeBillingConfig(cfg) {
    var slice = billingConfigSlice(cfg);
    if (lastBillingData) {
      lastBillingData.config = Object.assign({}, lastBillingData.config || {}, slice);
      renderBillingCards(lastBillingData);
      return lastBillingData;
    }
    var data = { config: slice, status: {} };
    renderBillingCards(data);
    return data;
  }

  function setQuotasSaveProgress(visible, percent, message) {
    var wrap = document.getElementById("quotasSaveProgress");
    var bar = document.getElementById("quotasSaveProgressBar");
    var status = document.getElementById("quotasSaveStatus");
    if (wrap) {
      wrap.classList.toggle("hidden", !visible);
      wrap.setAttribute("aria-hidden", visible ? "false" : "true");
      wrap.setAttribute("aria-valuenow", String(percent || 0));
    }
    if (bar) bar.style.width = Math.max(0, Math.min(100, percent || 0)) + "%";
    if (status) {
      status.classList.toggle("hidden", !visible || !message);
      status.textContent = message || "";
    }
  }

  function renderBillingCards(data) {
    var grid = document.getElementById("billingStatusCards");
    if (!grid) return;
    var st = data || {};
    var cfg = st.status || {};
    var siteCfg = st.config || {};
    var cards = [
      {
        cls: "oe-admin-card--purple",
        label: "Quota anónimos",
        value: (siteCfg.quota_anonymous_daily || "3") + "/dia",
        sub: "Por IP — transcrições + legendagens",
      },
      {
        cls: "oe-admin-card--blue",
        label: "Quota registados",
        value: (siteCfg.quota_registered_daily || "20") + "/dia",
        sub: "Contas gratuitas",
      },
      {
        cls: cfg.pricing_hidden ? "oe-admin-card--green" : "oe-admin-card--blue",
        label: "Preços no site",
        value: cfg.pricing_hidden ? "Escondidos" : "Visíveis",
        sub: cfg.pricing_hidden ? "Modo gratuito público" : "Planos e valores visíveis",
      },
      {
        cls: cfg.enabled ? "oe-admin-card--green" : "oe-admin-card--amber",
        label: "Pagamentos",
        value: cfg.enabled ? "Ativos" : "Desativados",
        sub: cfg.enabled ? "Checkout e DOCX Pro" : "Modo preparação",
      },
      {
        cls: cfg.checkout_ready ? "oe-admin-card--green" : "oe-admin-card--purple",
        label: "Stripe checkout",
        value: cfg.checkout_ready ? "Pronto" : "Incompleto",
        sub: cfg.stripe_configured ? "Chave secreta OK" : "Falta secret/price",
      },
      {
        cls: "oe-admin-card--blue",
        label: "Preço Pro",
        value: cfg.pricing_hidden ? "—" : (cfg.price_label || "—"),
        sub: cfg.pricing_hidden ? "Oculto no site" : "Quota " + (cfg.pro_quota_daily || "—") + "/dia",
      },
      {
        cls: "oe-admin-card--blue",
        label: "Subscrições",
        value: String((st.subscriptions || []).length),
        sub: "Registos na base",
      },
    ];
    grid.innerHTML = "";
    cards.forEach(function (c) {
      var card = document.createElement("div");
      card.className = "oe-admin-card " + c.cls;
      card.innerHTML =
        '<div class="oe-admin-card__label">' + c.label + "</div>" +
        '<div class="oe-admin-card__value" style="font-size:1rem">' + c.value + "</div>" +
        '<div class="oe-admin-card__sub">' + c.sub + "</div>";
      grid.appendChild(card);
    });
    var hint = document.getElementById("billingEnvHint");
    if (hint) {
      hint.innerHTML = cfg.enabled
        ? "Pagamentos <strong>ativos</strong>. Webhook: <code>" + apiBase() + "/api/billing/webhook</code>"
        : "Pagamentos <strong>desativados</strong>. Ativa quando o Stripe estiver configurado.";
    }
    var wh = document.getElementById("billingWebhookUrl");
    if (wh) wh.textContent = apiBase() + "/api/billing/webhook";
  }

  function renderSubs(items) {
    var box = document.getElementById("billingSubsTable");
    if (!box) return;
    if (!items || !items.length) {
      box.innerHTML = "<p class='oe-admin-empty'>Ainda sem subscrições Pro.</p>";
      return;
    }
    if (!global.OuviescreviAdmin || !global.OuviescreviAdmin.buildTable) {
      box.textContent = JSON.stringify(items);
      return;
    }
    box.innerHTML = "";
    box.appendChild(
      global.OuviescreviAdmin.buildTable(
        ["Email", "Plano", "Estado", "Até", "Atualizado"],
        items.map(function (r) {
          return [
            r.user_email,
            r.plan,
            r.status || "—",
            (r.current_period_end || "—").toString().slice(0, 10),
            (r.updated_at || "—").replace("T", " ").slice(0, 16),
          ];
        })
      )
    );
  }

  async function loadBilling() {
    try {
      var res = await fetch(apiBase() + "/api/admin/billing?_=" + Date.now(), {
        headers: authHeaders(),
        cache: "no-store",
      });
      if (!res.ok) throw new Error();
      var data = await res.json();
      lastBillingData = data;
      var cfg = data.config || {};
      applyQuotaFields(cfg);
      document.getElementById("billingCfgEnabled").checked = cfg.billing_enabled === "1";
      document.getElementById("billingCfgPricingHidden").checked = cfg.pricing_hidden !== "0";
      document.getElementById("billingCfgPriceLabel").value = cfg.pro_price_label || "";
      document.getElementById("billingCfgProQuota").value = cfg.pro_quota_daily || "200";
      document.getElementById("billingCfgStripePk").value = cfg.stripe_public_key || "";
      document.getElementById("billingCfgStripePrice").value = cfg.stripe_price_id_pro || "";
      document.getElementById("billingCfgStripeSecret").value = "";
      document.getElementById("billingCfgStripeSecret").placeholder = cfg.stripe_secret_set
        ? "•••••••• (definida — deixa vazio para manter)"
        : "sk_live_...";
      document.getElementById("billingCfgStripeWebhook").value = "";
      document.getElementById("billingCfgStripeWebhook").placeholder = cfg.stripe_webhook_set
        ? "•••••••• (definida — deixa vazio para manter)"
        : "whsec_...";
      renderBillingCards(data);
      renderSubs(data.subscriptions || []);
    } catch (e) {
      var grid = document.getElementById("billingStatusCards");
      if (grid) grid.innerHTML = "<p class='oe-admin-empty'>Erro ao carregar planos.</p>";
    }
  }

  async function saveQuotas(e) {
    e.preventDefault();
    var btn = document.getElementById("btnSaveQuotas");
    var anonWanted = String(document.getElementById("billingCfgQuotaAnon").value || "").trim();
    var regWanted = String(document.getElementById("billingCfgQuotaReg").value || "").trim();
    var progressTimer = null;

    global.OuviescreviUI.setButtonLoading(btn, true, "A guardar…");
    setQuotasSaveProgress(true, 12, "A enviar alterações…");
    progressTimer = setInterval(function () {
      var bar = document.getElementById("quotasSaveProgressBar");
      var current = bar ? parseInt(bar.style.width, 10) || 12 : 12;
      if (current < 88) {
        setQuotasSaveProgress(true, Math.min(88, current + 7), "A gravar na base de dados…");
      }
    }, 350);

    try {
      var updates = {
        quota_anonymous_daily: anonWanted,
        quota_registered_daily: regWanted,
      };
      var res = await fetch(apiBase() + "/api/admin/config", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ updates: updates }),
        cache: "no-store",
      });
      setQuotasSaveProgress(true, 92, "A confirmar…");
      var data = await res.json().catch(function () {
        return {};
      });
      if (!res.ok) {
        throw new Error(data.detail || "Erro ao guardar quotas.");
      }
      var cfg = data.config || {};
      if (
        String(cfg.quota_anonymous_daily) !== anonWanted ||
        String(cfg.quota_registered_daily) !== regWanted
      ) {
        throw new Error(
          "O servidor não confirmou os valores (" +
            (cfg.quota_anonymous_daily || "?") +
            "/" +
            (cfg.quota_registered_daily || "?") +
            ")."
        );
      }
      setQuotasSaveProgress(true, 100, "Concluído");
      lastBillingData = mergeBillingConfig(cfg);
      applyQuotaFields(cfg);
      syncSistemaQuotaFields(cfg);
      global.OuviescreviUI.toast(
        "Quotas guardadas: " + cfg.quota_anonymous_daily + " anónimos, " + cfg.quota_registered_daily + " registados / dia.",
        "success"
      );
      setTimeout(function () {
        setQuotasSaveProgress(false, 0, "");
      }, 800);
    } catch (err) {
      setQuotasSaveProgress(false, 0, "");
      global.OuviescreviUI.toast(err.message || "Erro ao guardar quotas.", "error");
    } finally {
      if (progressTimer) clearInterval(progressTimer);
      global.OuviescreviUI.setButtonLoading(btn, false);
    }
  }

  async function saveBilling(e) {
    e.preventDefault();
    var updates = {
      billing_enabled: document.getElementById("billingCfgEnabled").checked ? "1" : "0",
      pricing_hidden: document.getElementById("billingCfgPricingHidden").checked ? "1" : "0",
      pro_price_label: document.getElementById("billingCfgPriceLabel").value.trim(),
      pro_quota_daily: document.getElementById("billingCfgProQuota").value,
      stripe_public_key: document.getElementById("billingCfgStripePk").value.trim(),
      stripe_price_id_pro: document.getElementById("billingCfgStripePrice").value.trim(),
    };
    var sec = document.getElementById("billingCfgStripeSecret").value.trim();
    var wh = document.getElementById("billingCfgStripeWebhook").value.trim();
    if (sec) updates.stripe_secret_key = sec;
    if (wh) updates.stripe_webhook_secret = wh;
    try {
      var res = await fetch(apiBase() + "/api/admin/billing", {
        method: "PUT",
        headers: authHeaders(),
        body: JSON.stringify({ updates: updates }),
        cache: "no-store",
      });
      if (!res.ok) throw new Error();
      var data = await res.json().catch(function () {
        return {};
      });
      if (data.config) {
        lastBillingData = mergeBillingConfig(data.config);
      }
      global.OuviescreviUI.toast("Configuração de planos guardada.", "success");
      loadBilling();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  function init() {
    var form = document.getElementById("billingConfigForm");
    if (form) form.addEventListener("submit", saveBilling);
    var quotasForm = document.getElementById("quotasConfigForm");
    if (quotasForm) quotasForm.addEventListener("submit", saveQuotas);
    var btn = document.getElementById("btnRefreshBilling");
    if (btn) btn.addEventListener("click", loadBilling);
  }

  global.OuviescreviBillingAdmin = { init, loadBilling };
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
