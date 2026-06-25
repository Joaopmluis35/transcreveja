/**
 * Esconde preços e CTAs Pro no site público (modo gratuito).
 */
(function (global) {
  var hidden = true;
  var ready = false;
  var initPromise = null;

  function apply(isHidden) {
    hidden = !!isHidden;
    ready = true;
    var root = document.documentElement;
    var body = document.body;
    root.classList.toggle("oe-pricing-hidden", hidden);
    root.classList.add("oe-pricing-ready");
    if (body) body.classList.toggle("oe-pricing-hidden", hidden);
    global.OuviescreviPricing = { hidden: hidden, ready: true };
    document.querySelectorAll("[data-pricing-only]").forEach(function (el) {
      el.hidden = hidden;
      el.setAttribute("aria-hidden", hidden ? "true" : "false");
    });
    document.querySelectorAll("[data-pricing-free-only]").forEach(function (el) {
      el.hidden = !hidden;
    });
  }

  async function fetchHidden() {
    var base = "";
    try {
      if (global.OuviescreviAPI && global.OuviescreviAPI.init) {
        await global.OuviescreviAPI.init();
        base = global.OuviescreviAPI.getBase() || "";
      }
    } catch (e) {}
    try {
      var res = await fetch(base + "/api/billing/status");
      if (res.ok) {
        var data = await res.json();
        if (typeof data.pricing_hidden === "boolean") return data.pricing_hidden;
      }
    } catch (e) {}
    try {
      var res2 = await fetch(base + "/api/frontend-config");
      if (res2.ok) {
        var cfg = await res2.json();
        if (typeof cfg.pricingHidden === "boolean") return cfg.pricingHidden;
      }
    } catch (e) {}
    return true;
  }

  function init() {
    if (initPromise) return initPromise;
    document.documentElement.classList.add("oe-pricing-hidden");
    initPromise = fetchHidden()
      .then(apply)
      .catch(function () {
        apply(true);
      });
    return initPromise;
  }

  global.OuviescreviPricingVisibility = {
    init: init,
    isHidden: function () {
      return hidden;
    },
    whenReady: function () {
      return initPromise || init();
    },
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
