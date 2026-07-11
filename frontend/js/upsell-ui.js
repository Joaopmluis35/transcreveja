/**
 * Upsell — limite diário e pós-transcrição.
 */
(function (global) {
  var billingCache = null;

  function locale() {
    if (global.OuviescreviI18n) return global.OuviescreviI18n.localeFromPath();
    var m = (global.location && global.location.pathname || "").match(/^\/(en|es|fr|de)(\/|$)/);
    return m ? m[1] : "pt";
  }

  function t() {
    if (global.OuviescreviI18n && global.OuviescreviI18n.upsellStrings) {
      return global.OuviescreviI18n.upsellStrings(locale());
    }
    return {
      close: "Fechar",
      titleDefault: "Queres mais?",
      proCta: "Ver plano Pro",
      registerCta: "Criar conta grátis",
      dismiss: "Agora não",
      limitTitle: "Limite diário atingido",
      freeLimitTitle: "Limite da conta grátis",
      proLimitTitle: "Limite Pro de hoje",
      anonLimitHidden: "Criaste as transcrições grátis de hoje. Regista-te para mais utilizações ou tenta novamente amanhã.",
      anonLimit: "Criaste as transcrições grátis de hoje. Regista-te para mais utilizações ou passa ao Pro para exportação DOCX e limites maiores.",
      regLimitHidden: "Atingiste o limite diário da tua conta. Tenta novamente amanhã.",
      regLimit: "Atingiste o limite diário da conta grátis. O plano Pro inclui mais transcrições, exportação DOCX e histórico alargado.",
      proLimit: "Tenta novamente amanhã.",
      savedHistory: "Transcrição guardada no histórico.",
      saved: "Transcrição guardada",
      savedInHistory: " no histórico",
      proPitch: "Pro ({price}): exportação DOCX, mais transcrições/dia. ",
      proSoon: "Em breve: plano Pro com DOCX e mais transcrições. ",
      viewPlans: "Ver planos",
      createAccount: "Criar conta",
    };
  }

  function pricingPath() {
    if (global.OuviescreviI18n && global.OuviescreviI18n.pathFor) {
      return global.OuviescreviI18n.pathFor(locale(), "precos");
    }
    return locale() === "pt" ? "precos.html" : "en/precos.html";
  }

  function ensureModal() {
    var el = document.getElementById("oeUpsellModal");
    var strings = t();
    var href = pricingPath();
    if (el) {
      var proLink = el.querySelector("#oeUpsellCtaPro");
      if (proLink) proLink.setAttribute("href", href);
      return el;
    }
    el = document.createElement("div");
    el.id = "oeUpsellModal";
    el.className = "oe-upsell-modal hidden";
    el.setAttribute("role", "dialog");
    el.setAttribute("aria-modal", "true");
    el.innerHTML =
      '<div class="oe-upsell-modal__backdrop" data-oe-upsell-close="1"></div>' +
      '<div class="oe-upsell-modal__card">' +
      '  <button type="button" class="oe-upsell-modal__close" data-oe-upsell-close="1" aria-label="' + strings.close + '">✕</button>' +
      '  <h2 id="oeUpsellTitle" class="oe-upsell-modal__title">' + strings.titleDefault + '</h2>' +
      '  <p id="oeUpsellText" class="oe-upsell-modal__text"></p>' +
      '  <div class="oe-upsell-modal__actions">' +
      '    <a href="' + href + '" class="oe-upsell-btn oe-upsell-btn--primary" id="oeUpsellCtaPro" data-pricing-only>' + strings.proCta + '</a>' +
      '    <button type="button" class="oe-upsell-btn oe-upsell-btn--ghost" id="oeUpsellCtaRegister">' + strings.registerCta + '</button>' +
      '    <button type="button" class="oe-upsell-btn oe-upsell-btn--ghost" data-oe-upsell-close="1">' + strings.dismiss + '</button>' +
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
    var strings = t();
    var el = ensureModal();
    document.getElementById("oeUpsellTitle").textContent = title;
    document.getElementById("oeUpsellText").textContent = text;
    var regBtn = document.getElementById("oeUpsellCtaRegister");
    var proBtn = document.getElementById("oeUpsellCtaPro");
    if (regBtn) {
      regBtn.textContent = strings.registerCta;
      regBtn.classList.toggle("hidden", !!opts.hideRegister);
    }
    if (proBtn) {
      proBtn.textContent = opts.proLabel || strings.proCta;
      proBtn.setAttribute("href", pricingPath());
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
    var strings = t();
    if (tier === "anonymous") {
      show(
        strings.limitTitle,
        quota.message || (hidden ? strings.anonLimitHidden : strings.anonLimit),
        { hideRegister: false, hidePro: hidden }
      );
    } else if (quota.plan !== "pro") {
      show(
        hidden ? strings.limitTitle : strings.freeLimitTitle,
        quota.message || (hidden ? strings.regLimitHidden : strings.regLimit),
        { hideRegister: true, hidePro: hidden }
      );
    } else {
      show(strings.proLimitTitle, quota.message || strings.proLimit, {
        hideRegister: true,
        hidePro: true,
      });
    }
  }

  async function afterTranscriptionSuccess(quota) {
    quota = quota || {};
    if (quota.plan === "pro") return;
    var hidden = await pricingHidden();
    var strings = t();
    var plansHref = pricingPath();
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
      bannerOnly.innerHTML = "<strong>" + strings.savedHistory + "</strong>";
      bannerOnly.classList.remove("hidden");
      return;
    }
    var billing = await fetchBilling();
    var price = billing.price_label || (locale() === "en" ? "€9.99/month" : "9,99 €/mês");
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
    var pitch = billing.enabled
      ? strings.proPitch.replace("{price}", price)
      : strings.proSoon;
    banner.innerHTML =
      "<strong>" + strings.saved + (isUser ? strings.savedInHistory : "") + ".</strong> " +
      pitch +
      '<a href="' + plansHref + '">' + strings.viewPlans + "</a>" +
      (isUser ? "" : ' · <a href="#" id="oeSuccessUpsellReg">' + strings.createAccount + "</a>");
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
