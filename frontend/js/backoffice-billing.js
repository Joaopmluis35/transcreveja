/**
 * Backoffice — gestão de planos Pro e Stripe.
 */
(function (global) {
  function apiBase() {
    return global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase();
  }

  function authHeaders() {
    return global.OuviescreviAPI.adminAuthHeaders({ "Content-Type": "application/json" });
  }

  function renderBillingCards(status) {
    var grid = document.getElementById("billingStatusCards");
    if (!grid) return;
    var st = status || {};
    var cfg = st.status || {};
    var cards = [
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
        value: cfg.price_label || "—",
        sub: "Quota " + (cfg.pro_quota_daily || "—") + "/dia",
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
      var res = await fetch(apiBase() + "/api/admin/billing", { headers: authHeaders() });
      if (!res.ok) throw new Error();
      var data = await res.json();
      var cfg = data.config || {};
      document.getElementById("billingCfgEnabled").checked = cfg.billing_enabled === "1";
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

  async function saveBilling(e) {
    e.preventDefault();
    var updates = {
      billing_enabled: document.getElementById("billingCfgEnabled").checked ? "1" : "0",
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
      });
      if (!res.ok) throw new Error();
      global.OuviescreviUI.toast("Configuração de planos guardada.", "success");
      loadBilling();
    } catch (e) {
      global.OuviescreviUI.toast("Erro ao guardar.", "error");
    }
  }

  function init() {
    var form = document.getElementById("billingConfigForm");
    if (form) form.addEventListener("submit", saveBilling);
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
