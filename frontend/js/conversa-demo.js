/**
 * Página inicial: conversa de exemplo em que cada mensagem de voz «toca» e passa a texto.
 * Exemplo ilustrativo (não é uma gravação real). Só usa voz do aparelho se for português de Portugal.
 */
(function () {
  "use strict";
  var lista = document.getElementById("oeConversaMsgs");
  var ouvir = document.getElementById("oeConversaOuvir");
  if (!lista || !ouvir) return;
  var aviso = document.getElementById("oeConversaAviso");
  var FALAS = [
    ["00:00", "Jornalista", "Quando é que decidiu abrir a loja aqui na vila?"],
    ["00:04", "Ana", "Foi em 2019. Já tinha clientes de fora e faltava um sítio para os receber."],
    ["00:11", "Jornalista", "E o que mudou desde então?"],
    ["00:14", "Ana", "Começámos só com pão e bolos. Agora fazemos encomendas para casamentos e batizados."],
    ["00:26", "Ana", "Ao sábado vêm os vizinhos de sempre. No verão, os emigrantes que voltam."],
    ["00:37", "Ana", "Prefiro fazer bem aqui do que fazer mais em todo o lado."]
  ];
  var calmo = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var temFala = "speechSynthesis" in window;
  var voz = null, corrida = 0;
  var vazioHTML = lista.innerHTML;

  function escolherVoz() {
    voz = temFala ? (speechSynthesis.getVoices().filter(function (v) { return /^pt[-_]PT/i.test(v.lang); })[0] || null) : null;
  }
  if (temFala) { escolherVoz(); if (speechSynthesis.addEventListener) speechSynthesis.addEventListener("voiceschanged", escolherVoz); }
  function pausa(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }
  function falar(texto) {
    return new Promise(function (r) {
      if (!voz) return r();
      var u = new SpeechSynthesisUtterance(texto);
      u.voice = voz; u.lang = voz.lang; u.rate = 1.02;
      var fim = setTimeout(r, 15000);
      u.onend = u.onerror = function () { clearTimeout(fim); r(); };
      speechSynthesis.speak(u);
    });
  }
  // No computador a conversa tem scroll próprio; no telemóvel cresce com a página e não mexemos no scroll.
  function descer() { if (lista.scrollHeight > lista.clientHeight + 4) lista.scrollTop = lista.scrollHeight; }
  function duracao(s) { var n = Math.max(2, Math.round(s.length / 14)); return "0:" + (n < 10 ? "0" : "") + n; }

  function bolhaVoz(t, quem, texto) {
    var li = document.createElement("li");
    li.className = "oe-voz oe-entra";
    var semente = 7;
    for (var c = 0; c < texto.length; c++) semente = (semente * 31 + texto.charCodeAt(c)) % 9973;
    var barras = "";
    for (var b = 0; b < 26; b++) { semente = (semente * 73 + 41) % 997; barras += '<i style="height:' + (20 + semente % 80) + '%"></i>'; }
    li.innerHTML = '<span class="oe-voz__play" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M7 4.5v15l12-7.5z" fill="currentColor"/></svg></span>' +
      '<span class="oe-voz__onda" aria-hidden="true">' + barras + '</span><span class="oe-voz__meta"></span>';
    var meta = li.querySelector(".oe-voz__meta");
    var nome = document.createElement("b"); nome.textContent = quem;
    meta.appendChild(nome); meta.appendChild(document.createTextNode(duracao(texto) + " · " + t));
    li.setAttribute("aria-label", "Mensagem de voz de " + quem + ", " + t);
    lista.appendChild(li);
    return li;
  }
  function bolhaTexto(t, quem) {
    var li = document.createElement("li");
    li.className = "oe-texto oe-entra";
    li.innerHTML = '<span class="oe-texto__autor"></span><span class="oe-texto__frase"></span>' +
      '<span class="oe-texto__rodape"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l4-1 10-10-3-3L5 16l-1 4zM14 7l3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>transcrito · ' + t + "</span>";
    li.querySelector(".oe-texto__autor").textContent = quem;
    lista.appendChild(li);
    return li.querySelector(".oe-texto__frase");
  }
  function resumoFinal() {
    var li = document.createElement("li");
    li.className = "oe-resumo-bolha oe-entra";
    li.innerHTML = "<h3>Resumo em 3 pontos</h3><ol><li>A Ana abriu a loja na vila em 2019.</li><li>Hoje também faz encomendas para festas.</li><li>Prefere fazer bem num só sítio.</li></ol>";
    lista.appendChild(li); descer();
    ouvir.querySelector("span").textContent = "Ouvir outra vez";
  }
  function limpar() { corrida++; if (temFala) speechSynthesis.cancel(); lista.innerHTML = ""; }

  function mostrarTudo() {
    limpar();
    FALAS.forEach(function (f) {
      var b = bolhaVoz(f[0], f[1], f[2]);
      [].forEach.call(b.querySelectorAll(".oe-voz__onda i"), function (i) { i.classList.add("tocada"); });
      bolhaTexto(f[0], f[1]).textContent = f[2];
    });
    [].forEach.call(lista.querySelectorAll(".oe-entra"), function (e) { e.classList.remove("oe-entra"); });
    resumoFinal();
  }

  async function tocar() {
    limpar();
    var esta = corrida;
    if (aviso) aviso.textContent = voz ? "" : "Este aparelho não tem voz em português de Portugal: o exemplo corre sem som.";
    for (var k = 0; k < FALAS.length; k++) {
      var t = FALAS[k][0], quem = FALAS[k][1], texto = FALAS[k][2];
      if (esta !== corrida) return;
      // 1. Chega a mensagem de voz e «toca»: a onda vai ficando laranja.
      var b = bolhaVoz(t, quem, texto); descer();
      await pausa(calmo ? 0 : 350);
      b.classList.add("a-tocar");
      var fala = falar(texto);
      var barras = b.querySelectorAll(".oe-voz__onda i");
      var passo = calmo ? 0 : Math.max(30, Math.min(70, texto.length * 2.4));
      for (var i = 0; i < barras.length; i++) { if (esta !== corrida) return; barras[i].classList.add("tocada"); await pausa(passo); }
      await fala;
      b.classList.remove("a-tocar");
      if (esta !== corrida) return;
      // 2. «A escrever…» e depois o texto.
      if (!calmo) {
        var pontos = document.createElement("li");
        pontos.className = "oe-digitar oe-entra"; pontos.innerHTML = "<i></i><i></i><i></i>";
        lista.appendChild(pontos); descer(); await pausa(650); pontos.remove();
      }
      if (esta !== corrida) return;
      var frase = bolhaTexto(t, quem); descer();
      if (calmo) frase.textContent = texto;
      else for (var j = 1; j <= texto.length; j += 2) { if (esta !== corrida) return; frase.textContent = texto.slice(0, j); await pausa(14); }
      frase.textContent = texto; descer();
      await pausa(calmo ? 0 : 250);
    }
    if (esta === corrida) resumoFinal();
  }

  ouvir.addEventListener("click", tocar);
  lista.addEventListener("click", function (e) { if (e.target && e.target.id === "oeConversaTudo") mostrarTudo(); });
  void vazioHTML;

  // Ao fazer scroll, os cartões de exemplos entram como mensagens.
  if (!calmo && "IntersectionObserver" in window) {
    var alvos = document.querySelectorAll(".oe-home-usecases__card, .oe-home-usecases__title");
    if (alvos.length) {
      document.documentElement.classList.add("oe-anima");
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("visto"); io.unobserve(e.target); } });
      }, { rootMargin: "0px 0px -8% 0px" });
      [].forEach.call(alvos, function (el) {
        el.classList.add("oe-aparece");
        el.style.setProperty("--i", [].indexOf.call(el.parentElement.children, el));
        io.observe(el);
      });
    }
  }
})();
