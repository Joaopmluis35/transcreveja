/**
 * Menu «Design» nas páginas de design em teste: junta ao menu principal uma lista
 * para saltar entre o site clássico e as versões novas. Funciona em qualquer das páginas
 * porque usa a sua própria aparência (painel branco) e fica visível também no telemóvel.
 */
(function () {
  "use strict";
  var DESIGNS = [
    ["/index.html", "Clássico (site atual)"],
    ["/design/inicio-mensagens.html", "Mensagens de voz"],
    ["/design/inicio-caderno.html", "Caderno quadriculado"],
    ["/design/inicio-ardosia.html", "Quadro de ardósia"],
    ["/design/inicio-cortica.html", "Quadro de cortiça"],
    ["/design/inicio-folha.html", "Folha de 25 linhas"]
  ];
  var nav = document.querySelector("nav[aria-label='Principal'] ul");
  if (!nav) return;

  var css = document.createElement("style");
  css.textContent =
    ".oe-design{position:relative}" +
    ".oe-design summary{list-style:none;cursor:pointer;white-space:nowrap}" +
    ".oe-design summary::-webkit-details-marker{display:none}" +
    ".oe-design summary::after{content:'';display:inline-block;width:.42em;height:.42em;margin-left:.4em;border:solid currentColor;border-width:0 2px 2px 0;transform:translateY(-.15em) rotate(45deg)}" +
    ".oe-design ul{position:absolute;right:0;top:calc(100% + .5rem);z-index:20;min-width:14.5rem;margin:0;padding:.4rem;list-style:none;" +
    "background:#fff;color:#1d2420;border-radius:6px;box-shadow:0 2px 4px rgba(0,0,0,.15),0 18px 36px -12px rgba(0,0,0,.45);" +
    "font:500 .98rem/1.3 'Schibsted Grotesk',system-ui,sans-serif}" +
    // As páginas escondem itens do menu no telemóvel (.nav li); os desta lista ficam sempre visíveis.
    ".oe-design ul li{display:block!important}" +
    ".oe-design li a{display:block;padding:.65rem .8rem;border-radius:4px;text-decoration:none;color:inherit}" +
    ".oe-design li a:hover,.oe-design li a:focus-visible{background:#eef1f6}" +
    ".oe-design li a[aria-current='page']{font-weight:800}" +
    ".oe-design li a[aria-current='page']::after{content:' ✓'}" +
    ".oe-design .oe-design__nota{padding:.5rem .8rem .3rem;font-size:.8rem;color:#5b625e;border-top:1px solid #e3e6ea;margin-top:.3rem}";
  document.head.appendChild(css);

  var aqui = location.pathname.replace(/\/$/, "/index.html");
  var li = document.createElement("li");
  li.className = "manter oe-design";
  var itens = DESIGNS.map(function (d) {
    var atual = aqui === d[0] ? ' aria-current="page"' : "";
    return '<li><a href="' + d[0] + '"' + atual + ">" + d[1] + "</a></li>";
  }).join("");
  li.innerHTML = "<details><summary>Design</summary><ul>" + itens +
    '<li class="oe-design__nota">Versões novas em teste. Para transcrever, os botões levam à ferramenta atual.</li></ul></details>';
  nav.insertBefore(li, nav.lastElementChild);

  // Fecha a lista ao clicar fora ou com a tecla Esc.
  var det = li.querySelector("details");
  document.addEventListener("click", function (e) { if (!li.contains(e.target)) det.open = false; });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape" && det.open) { det.open = false; det.querySelector("summary").focus(); } });
})();
