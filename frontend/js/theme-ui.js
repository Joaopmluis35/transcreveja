/**
 * Design do site público, escolhido no botão «Design» do cabeçalho, sem mudar de página.
 * Guardado no browser (oe_design). Sem escolha = «Mensagens de voz» (principal).
 * «Quadro de ardósia» reaproveita as regras escuras (html.oe-theme-dark) recoloridas em theme-dark.css.
 */
(function (global) {
  "use strict";

  var CHAVE = "oe_design";
  var DESIGNS = ["mensagens", "caderno", "ardosia", "cortica", "folha"];

  function isBackofficePage() {
    return (global.location.pathname || "").indexOf("backoffice") !== -1;
  }
  function lerDesign() {
    try {
      var d = localStorage.getItem(CHAVE);
      return DESIGNS.indexOf(d) !== -1 ? d : "mensagens";
    } catch (e) {
      return "mensagens";
    }
  }

  function aplicar(design) {
    if (isBackofficePage()) return;
    if (DESIGNS.indexOf(design) === -1) design = "mensagens";
    var html = document.documentElement;
    if (design === "mensagens") html.removeAttribute("data-design");
    else html.setAttribute("data-design", design);
    var escuro = design === "ardosia";
    html.classList.toggle("oe-theme-dark", escuro);
    html.setAttribute("data-theme", escuro ? "dark" : "light");
    try { localStorage.setItem(CHAVE, design); } catch (e) {}
    marcarEscolha(design);
  }

  function marcarEscolha(design) {
    var ops = document.querySelectorAll(".oe-design-op");
    for (var i = 0; i < ops.length; i++) {
      ops[i].setAttribute("aria-checked", ops[i].getAttribute("data-design") === design ? "true" : "false");
    }
  }

  function init() {
    if (isBackofficePage()) return;
    marcarEscolha(lerDesign());
    var btn = document.getElementById("oeThemeToggle");
    if (btn) btn.hidden = true;
  }

  // O cabeçalho chega depois (fetch): por isso os cliques são tratados aqui, por delegação.
  document.addEventListener("click", function (e) {
    var op = e.target.closest && e.target.closest(".oe-design-op");
    if (op) {
      aplicar(op.getAttribute("data-design"));
      var d = op.closest("details");
      if (d) d.open = false;
      return;
    }
    var aberta = document.querySelector(".oe-design-troca[open]");
    // Fecha a lista «Design» ao clicar fora dela (inclui abrir o menu de três riscas).
    if (aberta && !aberta.contains(e.target)) aberta.open = false;
  });
  // Ao abrir a lista «Design», fecha o menu de três riscas e marca a escolha atual.
  document.addEventListener("toggle", function (e) {
    var d = e.target;
    if (!d.classList || !d.classList.contains("oe-design-troca") || !d.open) return;
    marcarEscolha(lerDesign());
    var header = document.getElementById("oeProHeader");
    if (header && header.classList.contains("is-mobile-open")) {
      header.classList.remove("is-mobile-open");
      var t = header.querySelector(".oe-pro-nav__mobile-toggle");
      if (t) t.setAttribute("aria-expanded", "false");
      document.body.classList.remove("oe-nav-open");
    }
  }, true);

  // Aplicar o mais cedo possível (antes de desenhar a página), para não piscar.
  aplicar(lerDesign());

  global.OuviescreviTheme = { init: init, apply: aplicar, resolve: lerDesign };
})(window);
