/**
 * Upsell — limite diário e pós-transcrição.
 */
(function (global) {
  var billingCache = null;

  function ensureModal() {
    var el = document.getElementById("oeUpsellModal");
    if (el) return el;
    el = document.createElement("div");
    el.id = "oeUpsellModal";
    el.className = "oe-upsell-modal hidden";
    el.setAttribute("role", "dialog");
    el.setAttribute("aria-modal", "true");
    el.innerHTML =
      '<div class="oe-upsell-modal__backdrop" data-oe-upsell-close="1"></div>' +
      '<div class="oe-upsell-modal__card">' +
      '  <button type="button" class="oe-upsell-modal__close" data-oe-upsell-close="1" aria-label="Fechar">✕</button>' +
      '  <h2 id="oeUpsellTitle" class="oe-upsell-modal__title">Queres mais?</h2>' +
      '  <p id="oeUpsellText" class="oe-upsell-modal__text"></p>' +
      '  <div class="oe-upsell-modal__actions">' +
      '    <a href="precos.html" class="oe-upsell-btn oe-upsell-btn--primary" id="oeUpsellCtaPro" data-pricing-only>Ver plano Pro</a>' +
      '    <button type="button" class="oe-upsell-btn oe-upsell-btn--ghost" id="oeUpsellCtaRegister">Criar conta grátis</button>' +
      '    <button type="button" class="oe-upsell-btn oe-upsell-btn--ghost" data-oe-upsell-close="1">Agora não</button>' +
      '  </div>' +
      "</div>";
    document.body.appendChild(el);
    el.addEventListener("click", function (e) {
      if (e.target.closest("[data-oe-upsell-close]")) hide();
    });
    var reg = document.getElementById("oeUpsellCtaRegister");
    if (reg) {
      reg.addEventListener("click", function () {
        hide();
        if (global.OuviescreviAuth && global.OuviescreviAuth.openModal) {
          global.OuviescreviAuth.openModal("register");
        }
      });
    }
    return el;
  }

  function hide() {
    var el = document.getElementById("oeUpsellModal");
    if (el) el.classList.add("hidden");
  }

  function show(title, text, opts) {
    opts = opts || {};
    var el = ensureModal();
    document.getElementById("oeUpsellTitle").textContent = title;
    document.getElementById("oeUpsellText").textContent = text;
    var regBtn = document.getElementById("oeUpsellCtaRegister");
    var proBtn = document.getElementById("oeUpsellCtaPro");
    if (regBtn) regBtn.classList.toggle("hidden", !!opts.hideRegister);
    if (proBtn) {
      proBtn.textContent = opts.proLabel || "Ver plano Pro";
      proBtn.classList.toggle("hidden", !!opts.hidePro);
    }
    el.classList.remove("hidden");
  }

  async function pricingHidden() {
    if (global.OuviescreviPricing && typeof global.OuviescreviPricing.hidden === "boolean") {
      return global.OuviescreviPricing.hidden;
    }
    if (global.OuviescreviPricingVisibility && global.OuviescreviPricingVisibility.whenReady) {
      await global.OuviescreviPricingVisibility.whenReady();
      if (global.OuviescreviPricing) return !!global.OuviescreviPricing.hidden;
    }
    var billing = await fetchBilling();
    return !!billing.pricing_hidden;
  }

  async function fetchBilling() {
    if (billingCache) return billingCache;
    try {
      await global.OuviescreviAPI.init();
      var res = await fetch(global.OuviescreviAPI.getBase() + "/api/billing/status");
      if (res.ok) billingCache = await res.json();
    } catch (e) {}
    return billingCache || { enabled: false, pricing_hidden: true, price_label: "" };
  }

  async function showLimit(quota) {
    quota = quota || {};
    var tier = quota.tier || "anonymous";
    var hidden = await pricingHidden();
    if (tier === "anonymous") {
      show(
        "Limite diário atingido",
        quota.message ||
          (hidden
            ? "Criaste as transcrições grátis de hoje. Regista-te para mais utilizações ou tenta novamente amanhã."
            : "Criaste as transcrições grátis de hoje. Regista-te para mais utilizações ou passa ao Pro para exportação DOCX e limites maiores."),
        { hideRegister: false, hidePro: hidden }
      );
    } else if (quota.plan !== "pro") {
      show(
        hidden ? "Limite diário atingido" : "Limite da conta grátis",
        quota.message ||
          (hidden
            ? "Atingiste o limite diário da tua conta. Tenta novamente amanhã."
            : "Atingiste o limite diário da conta grátis. O plano Pro inclui mais transcrições, exportação DOCX e histórico alargado."),
        { hideRegister: true, hidePro: hidden }
      );
    } else {
      show("Limite Pro de hoje", quota.message || "Tenta novamente amanhã.", {
        hideRegister: true,
        hidePro: true,
      });
    }
  }

  async function afterTranscriptionSuccess(quota) {
    quota = quota || {};
    if (quota.plan === "pro") return;
    var hidden = await pricingHidden();
    if (hidden) {
      if (quota.tier !== "registered") return;
      var bannerOnly = document.getElementById("oeSuccessUpsell");
      if (!bannerOnly) {
        bannerOnly = document.createElement("div");
        bannerOnly.id = "oeSuccessUpsell";
        bannerOnly.className = "oe-success-upsell hidden";
        bannerOnly.setAttribute("role", "status");
        var output = document.getElementById("output");
        if (output && output.parentNode) {
          output.parentNode.insertBefore(bannerOnly, output);
        }
      }
      bannerOnly.innerHTML = "<strong>Transcrição guardada no histórico.</strong>";
      bannerOnly.classList.remove("hidden");
      return;
    }
    var billing = await fetchBilling();
    var price = billing.price_label || "9,99 €/mês";
    var banner = document.getElementById("oeSuccessUpsell");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "oeSuccessUpsell";
      banner.className = "oe-success-upsell hidden";
      banner.setAttribute("role", "status");
      var output = document.getElementById("output");
      if (output && output.parentNode) {
        output.parentNode.insertBefore(banner, output);
      } else {
        document.getElementById("adminContent")?.appendChild(banner);
      }
    }
    var isUser = quota.tier === "registered";
    banner.innerHTML =
      "<strong>Transcrição guardada" +
      (isUser ? " no histórico" : "") +
      ".</strong> " +
      (billing.enabled
        ? "Pro (" + price + "): exportação DOCX, mais transcrições/dia. "
        : "Em breve: plano Pro com DOCX e mais transcrições. ") +
      '<a href="precos.html">Ver planos</a>' +
      (isUser ? "" : ' · <a href="#" id="oeSuccessUpsellReg">Criar conta</a>');
    banner.classList.remove("hidden");
    var reg = document.getElementById("oeSuccessUpsellReg");
    if (reg) {
      reg.addEventListener("click", function (e) {
        e.preventDefault();
        if (global.OuviescreviAuth) global.OuviescreviAuth.openModal("register");
      });
    }
  }

  global.OuviescreviUpsell = {
    show,
    showLimit,
    afterTranscriptionSuccess,
    hide,
  };
})(window);
