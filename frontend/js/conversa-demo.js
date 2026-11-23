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
  // Textos por língua da página (<html lang>); exemplo ilustrativo, não é uma gravação real.
  var TEXTOS = {"pt": {"falas": [["00:00", "Jornalista", "Quando é que decidiu abrir a loja aqui na vila?"], ["00:04", "Ana", "Foi em 2019. Já tinha clientes de fora e faltava um sítio para os receber."], ["00:11", "Jornalista", "E o que mudou desde então?"], ["00:14", "Ana", "Começámos só com pão e bolos. Agora fazemos encomendas para casamentos e batizados."], ["00:26", "Ana", "Ao sábado vêm os vizinhos de sempre. No verão, os emigrantes que voltam."], ["00:37", "Ana", "Prefiro fazer bem aqui do que fazer mais em todo o lado."]], "voz": "^pt[-_]PT", "ouvir": "Ouvir um exemplo", "outra": "Ouvir outra vez", "parar": "Parar", "semVoz": "Este aparelho não tem voz em português de Portugal: o exemplo corre sem som.", "transcrito": "transcrito", "msgVoz": "Mensagem de voz de", "resumoTitulo": "Resumo em 3 pontos", "resumo": ["A Ana abriu a loja na vila em 2019.", "Hoje também faz encomendas para festas.", "Prefere fazer bem num só sítio."], "visto": "Transcrito!"}, "en": {"falas": [["00:00", "Journalist", "When did you decide to open the shop here in the village?"], ["00:04", "Ana", "In 2019. I already had customers from outside and there was nowhere to welcome them."], ["00:11", "Journalist", "And what has changed since then?"], ["00:14", "Ana", "We started with just bread and cakes. Now we take orders for weddings and christenings."], ["00:26", "Ana", "On Saturdays the usual neighbours come by. In summer, the emigrants who come back."], ["00:37", "Ana", "I'd rather do things well here than do more everywhere."]], "voz": "^en", "ouvir": "Play an example", "outra": "Play again", "parar": "Stop", "semVoz": "This device has no English voice: the example runs without sound.", "transcrito": "transcribed", "msgVoz": "Voice message from", "resumoTitulo": "Summary in 3 points", "resumo": ["Ana opened the village shop in 2019.", "She now also takes orders for parties.", "She prefers to do things well in one place."], "visto": "Transcribed!"}, "es": {"falas": [["00:00", "Periodista", "¿Cuándo decidió abrir la tienda aquí en el pueblo?"], ["00:04", "Ana", "En 2019. Ya tenía clientes de fuera y faltaba un sitio para recibirlos."], ["00:11", "Periodista", "¿Y qué ha cambiado desde entonces?"], ["00:14", "Ana", "Empezamos solo con pan y pasteles. Ahora hacemos encargos para bodas y bautizos."], ["00:26", "Ana", "Los sábados vienen los vecinos de siempre. En verano, los emigrantes que vuelven."], ["00:37", "Ana", "Prefiero hacer bien las cosas aquí que hacer más en todas partes."]], "voz": "^es", "ouvir": "Escuchar un ejemplo", "outra": "Escuchar otra vez", "parar": "Detener", "semVoz": "Este dispositivo no tiene voz en español: el ejemplo funciona sin sonido.", "transcrito": "transcrito", "msgVoz": "Mensaje de voz de", "resumoTitulo": "Resumen en 3 puntos", "resumo": ["Ana abrió la tienda del pueblo en 2019.", "Ahora también hace encargos para fiestas.", "Prefiere hacer bien las cosas en un solo sitio."], "visto": "¡Transcrito!"}, "fr": {"falas": [["00:00", "Journaliste", "Quand avez-vous décidé d'ouvrir la boutique ici, au village ?"], ["00:04", "Ana", "En 2019. J'avais déjà des clients de l'extérieur et il manquait un endroit pour les accueillir."], ["00:11", "Journaliste", "Et qu'est-ce qui a changé depuis ?"], ["00:14", "Ana", "On a commencé avec du pain et des gâteaux. Maintenant on prend des commandes pour les mariages et les baptêmes."], ["00:26", "Ana", "Le samedi, les voisins de toujours passent. En été, les émigrés qui reviennent."], ["00:37", "Ana", "Je préfère bien faire ici que faire plus partout."]], "voz": "^fr", "ouvir": "Écouter un exemple", "outra": "Réécouter", "parar": "Arrêter", "semVoz": "Cet appareil n'a pas de voix française : l'exemple fonctionne sans son.", "transcrito": "transcrit", "msgVoz": "Message vocal de", "resumoTitulo": "Résumé en 3 points", "resumo": ["Ana a ouvert la boutique du village en 2019.", "Elle prend aussi des commandes pour les fêtes.", "Elle préfère bien faire en un seul endroit."], "visto": "Transcrit !"}, "de": {"falas": [["00:00", "Journalistin", "Wann haben Sie beschlossen, den Laden hier im Dorf zu eröffnen?"], ["00:04", "Ana", "2019. Ich hatte schon Kunden von außerhalb, und es fehlte ein Ort, um sie zu empfangen."], ["00:11", "Journalistin", "Und was hat sich seitdem geändert?"], ["00:14", "Ana", "Wir haben nur mit Brot und Kuchen angefangen. Jetzt nehmen wir Bestellungen für Hochzeiten und Taufen an."], ["00:26", "Ana", "Samstags kommen die Nachbarn von immer. Im Sommer die Ausgewanderten, die zurückkehren."], ["00:37", "Ana", "Ich mache lieber hier alles gut, als überall mehr zu machen."]], "voz": "^de", "ouvir": "Beispiel anhören", "outra": "Nochmal anhören", "parar": "Stopp", "semVoz": "Dieses Gerät hat keine deutsche Stimme: Das Beispiel läuft ohne Ton.", "transcrito": "transkribiert", "msgVoz": "Sprachnachricht von", "resumoTitulo": "Zusammenfassung in 3 Punkten", "resumo": ["Ana hat den Dorfladen 2019 eröffnet.", "Sie nimmt jetzt auch Bestellungen für Feste an.", "Sie macht lieber an einem Ort alles gut."], "visto": "Transkribiert!"}};
  var L = TEXTOS[(document.documentElement.lang || "pt").slice(0, 2).toLowerCase()] || TEXTOS.pt;
  var FALAS = L.falas;
  var calmo = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
  var temFala = "speechSynthesis" in window;
  var voz = null, corrida = 0;
  var vazioHTML = lista.innerHTML;

  function escolherVoz() {
    voz = temFala ? (speechSynthesis.getVoices().filter(function (v) { return new RegExp(L.voz, "i").test(v.lang); })[0] || null) : null;
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
    li.setAttribute("aria-label", L.msgVoz + " " + quem + ", " + t);
    lista.appendChild(li);
    return li;
  }
  function bolhaTexto(t, quem) {
    var li = document.createElement("li");
    li.className = "oe-texto oe-entra";
    li.innerHTML = '<span class="oe-texto__autor"></span><span class="oe-texto__frase"></span>' +
      '<span class="oe-texto__rodape"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20l4-1 10-10-3-3L5 16l-1 4zM14 7l3 3" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>' + L.transcrito + ' · ' + t + "</span>";
    li.querySelector(".oe-texto__autor").textContent = quem;
    lista.appendChild(li);
    return li.querySelector(".oe-texto__frase");
  }
  function resumoFinal() {
    var li = document.createElement("li");
    li.className = "oe-resumo-bolha oe-entra";
    li.innerHTML = "<h3></h3><ol><li></li><li></li><li></li></ol>";
    li.querySelector("h3").textContent = L.resumoTitulo;
    [].forEach.call(li.querySelectorAll("ol li"), function (x, i) { x.textContent = L.resumo[i]; });
    lista.appendChild(li); descer();
    botao("outra");
  }
  // Instrumento que escreve a mensagem: giz na ardósia, caneta no caderno e na folha (segue as letras).
  var GIZ = '<svg viewBox="0 0 64 16"><path d="M9 1.5H58.5a6.5 6.5 0 0 1 0 13H9C5 14.2 1.6 11.6 1 8.6.6 6.2 3.6 2.6 9 1.5Z" transform="translate(2.5 4.5)" fill="#000" opacity=".28"/><path d="M9 1.5H58.5a6.5 6.5 0 0 1 0 13H9C5 14.2 1.6 11.6 1 8.6.6 6.2 3.6 2.6 9 1.5Z" fill="#f4f6f0"/><ellipse cx="58.5" cy="8" rx="3.2" ry="6.5" fill="#d6dbd0"/><path d="M12 4.2H55" stroke="#fff" stroke-width="1.2" stroke-linecap="round"/></svg>';
  var CANETA = '<svg viewBox="0 0 64 16"><path d="M0 8 9 4.5v7z" fill="#8f9aa6"/><circle cx="1.2" cy="8" r="1.3" fill="#1e3fa8"/><rect x="9" y="3.5" width="44" height="9" rx="2" fill="#eef4fc" stroke="#9fb7d6"/><rect x="12" y="6.6" width="38" height="2.8" rx="1.4" fill="#1e3fa8" opacity=".55"/><rect x="52" y="3" width="11" height="10" rx="2.5" fill="#1e3fa8"/><rect x="30" y="1" width="20" height="2.6" rx="1.2" fill="#1e3fa8"/></svg>';
  var inst = null, alvo = { x: 0, y: 0, a: -30 }, atual = { x: 0, y: 0, a: -30 }, raf = 0, tAnt = 0, letras = 0;
  function tipoInstrumento() {
    var d = document.documentElement.getAttribute("data-design");
    return d === "ardosia" ? "giz" : (d === "caderno" || d === "folha") ? "caneta" : "";
  }
  function passo(t) {
    var dt = Math.min(64, t - (tAnt || t)); tAnt = t;
    var k = 1 - Math.exp(-dt / 40);
    atual.x += (alvo.x - atual.x) * k; atual.y += (alvo.y - atual.y) * k; atual.a += (alvo.a - atual.a) * k;
    inst.style.transform = "translate3d(" + atual.x.toFixed(1) + "px," + atual.y.toFixed(1) + "px,0) rotate(" + atual.a.toFixed(1) + "deg)";
    raf = requestAnimationFrame(passo);
  }
  function moverInstrumento(cursor) {
    var tipo = tipoInstrumento();
    if (calmo || !tipo) return;
    if (!inst || !inst.isConnected) {
      inst = document.createElement("li");
      inst.className = "oe-instrumento"; inst.setAttribute("aria-hidden", "true");
      lista.appendChild(inst);
    }
    if (inst.getAttribute("data-tipo") !== tipo) { inst.innerHTML = tipo === "giz" ? GIZ : CANETA; inst.setAttribute("data-tipo", tipo); }
    var l = lista.getBoundingClientRect(), c = cursor.getBoundingClientRect();
    letras++;
    alvo.x = c.left - l.left + lista.scrollLeft;
    alvo.y = c.bottom - l.top + lista.scrollTop - 5 + Math.sin(letras * 2.1);
    alvo.a = (tipo === "giz" ? -30 : -38) + Math.sin(letras * 1.7) * 3;
    if (!raf) { atual.x = alvo.x; atual.y = alvo.y; atual.a = alvo.a; tAnt = 0; raf = requestAnimationFrame(passo); inst.classList.add("ativo"); }
    if (tipo === "giz" && letras % 4 === 0) {
      var po = document.createElement("li"); po.className = "oe-po"; po.setAttribute("aria-hidden", "true");
      po.style.left = alvo.x + "px"; po.style.top = (alvo.y + 4) + "px";
      po.addEventListener("animationend", function () { po.remove(); });
      lista.appendChild(po);
    }
  }
  function largarInstrumento() {
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
    if (inst) inst.classList.remove("ativo");
  }

  var aTocar = false;
  function botao(estado) {
    aTocar = estado === "parar";
    ouvir.querySelector("span").textContent = aTocar ? L.parar : estado === "outra" ? L.outra : L.ouvir;
    var icone = ouvir.querySelector("path");
    if (icone) icone.setAttribute("d", aTocar ? "M6 6h12v12H6z" : "M7 4.5v15l12-7.5z");
    ouvir.setAttribute("aria-pressed", aTocar ? "true" : "false");
  }
  function limpar() { corrida++; if (temFala) speechSynthesis.cancel(); largarInstrumento(); inst = null; lista.innerHTML = ""; }

  // Caderno e Ardósia: em vez de bolhas, apontamentos em linhas (minuto · quem · texto), como nas demos.
  function design() { return document.documentElement.getAttribute("data-design") || ""; }
  function emLinhas() { var d = design(); return d === "caderno" || d === "ardosia"; }
  function linha(t, quem) {
    var li = document.createElement("li");
    li.className = "oe-linha";
    li.innerHTML = '<time></time><span class="oe-linha__fala"><b></b> <span class="oe-linha__txt"></span></span>';
    li.querySelector("time").textContent = t;
    li.querySelector("b").textContent = quem + ":";
    lista.appendChild(li);
    return li;
  }
  // Fim: no caderno o professor circula e dá o visto; na ardósia o resumo aparece numa caixa a giz.
  function fechoLinhas() {
    var li = document.createElement("li");
    if (design() === "caderno") {
      li.className = "oe-visto"; li.setAttribute("aria-hidden", "true");
      li.innerHTML = '<svg viewBox="0 0 208 90"><path style="--c:560" d="M30 18C70 2 168 4 196 30c14 14-6 40-58 50-50 9-112 2-126-22C2 40 26 22 64 14"/><path class="tick" style="--c:70" d="M170 62c5 4 9 9 12 15 8-16 16-27 24-34"/></svg><span>' + L.visto + '</span>';
    } else {
      li.className = "oe-resumo-giz";
      li.innerHTML = '<svg class="caixa" viewBox="0 0 400 120" preserveAspectRatio="none" aria-hidden="true"><path d="M4 8C120 3 290 5 396 6c2 30 1 70-2 108C270 117 130 116 6 113 3 80 2 40 6 4"/></svg>' +
        "<h3></h3><ol><li></li><li></li><li></li></ol>";
      li.querySelector("h3").textContent = L.resumoTitulo;
      [].forEach.call(li.querySelectorAll("ol li"), function (x, i) { x.textContent = L.resumo[i]; });
    }
    lista.appendChild(li); descer();
    void li.offsetWidth; li.classList.add("feito");
    botao("outra");
  }
  // Na ardósia, antes de voltar a escrever, o apagador passa pelo quadro.
  async function apagar() {
    if (design() !== "ardosia" || calmo || !lista.querySelector(".oe-linha")) return;
    lista.classList.add("a-apagar"); await pausa(700); lista.classList.remove("a-apagar");
  }
  async function tocarLinhas() {
    var antes = corrida;
    await apagar();
    if (antes !== corrida) return; // carregou em «Parar» enquanto o apagador passava
    limpar();
    var esta = corrida;
    if (aviso) aviso.textContent = voz ? "" : L.semVoz;
    for (var k = 0; k < FALAS.length; k++) {
      if (esta !== corrida) return;
      var li = linha(FALAS[k][0], FALAS[k][1]), alvoTxt = li.querySelector(".oe-linha__txt"), texto = FALAS[k][2];
      [].forEach.call(lista.querySelectorAll(".agora"), function (e) { e.classList.remove("agora"); });
      li.classList.add("agora"); descer();
      var fala = falar(texto);
      if (calmo) alvoTxt.textContent = texto;
      else {
        var no = document.createTextNode(""), cur = document.createElement("span");
        cur.className = "oe-cursor"; alvoTxt.appendChild(no); alvoTxt.appendChild(cur);
        for (var j = 1; j <= texto.length; j++) {
          if (esta !== corrida) return;
          no.data = texto.slice(0, j); moverInstrumento(cur);
          await pausa(voz ? 34 : 24);
        }
        largarInstrumento(); alvoTxt.textContent = texto;
      }
      await fala;
      await pausa(calmo ? 0 : 220);
    }
    if (esta !== corrida) return;
    [].forEach.call(lista.querySelectorAll(".agora"), function (e) { e.classList.remove("agora"); });
    fechoLinhas();
  }

  function mostrarTudo() {
    limpar();
    if (emLinhas()) {
      FALAS.forEach(function (f) { linha(f[0], f[1]).querySelector(".oe-linha__txt").textContent = f[2]; });
      fechoLinhas();
      return;
    }
    FALAS.forEach(function (f) {
      var b = bolhaVoz(f[0], f[1], f[2]);
      [].forEach.call(b.querySelectorAll(".oe-voz__onda i"), function (i) { i.classList.add("tocada"); });
      bolhaTexto(f[0], f[1]).textContent = f[2];
    });
    [].forEach.call(lista.querySelectorAll(".oe-entra"), function (e) { e.classList.remove("oe-entra"); });
    resumoFinal();
  }

  async function tocar() {
    if (emLinhas()) return tocarLinhas();
    limpar();
    var esta = corrida;
    if (aviso) aviso.textContent = voz ? "" : L.semVoz;
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
      else {
        // O texto cresce num nó próprio e o cursor (invisível) fica no fim, para o giz/caneta seguir as letras.
        var no = document.createTextNode(""), cur = document.createElement("span");
        cur.className = "oe-cursor"; frase.appendChild(no); frase.appendChild(cur);
        var comInst = !!tipoInstrumento();
        for (var j = 1; j <= texto.length; j += comInst ? 1 : 2) {
          if (esta !== corrida) return;
          no.data = texto.slice(0, j);
          if (comInst) moverInstrumento(cur);
          await pausa(comInst ? 24 : 14);
        }
        largarInstrumento();
      }
      frase.textContent = texto; descer();
      await pausa(calmo ? 0 : 250);
    }
    if (esta === corrida) resumoFinal();
  }

  // Enquanto o exemplo corre, o botão passa a «Parar»: cala a voz e deixa o que já foi escrito.
  function parar() {
    corrida++; if (temFala) speechSynthesis.cancel(); largarInstrumento();
    [].forEach.call(lista.querySelectorAll(".a-tocar"), function (e) { e.classList.remove("a-tocar"); });
    [].forEach.call(lista.querySelectorAll(".oe-digitar, .oe-instrumento"), function (e) { e.remove(); });
    botao(lista.querySelector(".oe-texto, .oe-linha") ? "outra" : "inicio");
  }
  ouvir.addEventListener("click", function () {
    if (aTocar) parar();
    else { botao("parar"); tocar(); }
  });
  lista.addEventListener("click", function (e) { if (e.target && e.target.id === "oeConversaTudo") mostrarTudo(); });
  // Quando aparece o progresso ou o resultado real, o exemplo sai (nas páginas EN/ES/FR/DE não há outro código que o esconda).
  var painel = document.getElementById("homeOutputPlaceholder");
  if (painel && "MutationObserver" in window) {
    var vigiar = ["loading", "output"].map(function (id) { return document.getElementById(id); }).filter(Boolean);
    var ver = function () {
      var ativo = vigiar.some(function (el) { return !el.classList.contains("hidden") && !el.hidden; });
      if (ativo && !painel.hidden) { parar(); painel.hidden = true; }
    };
    vigiar.forEach(function (el) { new MutationObserver(ver).observe(el, { attributes: true, attributeFilter: ["class", "hidden"] }); });
  }
  if ("MutationObserver" in window) {
    new MutationObserver(function () {
      limpar(); lista.innerHTML = vazioHTML;
      botao("inicio");
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-design"] });
  }

  // Ao fazer scroll, os cartões de exemplos entram como mensagens.
  if (!calmo && "IntersectionObserver" in window) {
    var alvos = document.querySelectorAll(".oe-home-usecases__card, .oe-home-usecases__title, .oe-exemplo-tr");
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
