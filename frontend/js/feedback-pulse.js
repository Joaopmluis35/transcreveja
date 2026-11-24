/**
 * Pedido leve de CSAT após uma transcrição bem-sucedida.
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "oe_csat_cooldown_until";
  var COOLDOWN_MS = 7 * 24 * 60 * 60 * 1000;
  var SHOW_DELAY_MS = 1800;
  var selectedRating = 0;
  var showTimer = null;

  var MSGS = {
    pt: {
      title: "Como correu a transcrição?",
      commentPh: "O que podemos melhorar?",
      send: "Enviar",
      dismiss: "Agora não",
      thanks: "Obrigado pelo feedback!",
      pick: "Escolhe uma classificação de 1 a 5.",
      error: "Não foi possível enviar. Tenta de novo.",
      offline: "Erro de ligação. Verifica a internet.",
    },
    en: {
      title: "How was the transcription?",
      commentPh: "What can we improve?",
      send: "Send",
      dismiss: "Not now",
      thanks: "Thanks for your feedback!",
      pick: "Please choose a rating from 1 to 5.",
      error: "Could not send. Please try again.",
      offline: "Connection error. Check your internet.",
    },
    es: {
      title: "¿Cómo fue la transcripción?",
      commentPh: "¿Qué podemos mejorar?",
      send: "Enviar",
      dismiss: "Ahora no",
      thanks: "¡Gracias por tu opinión!",
      pick: "Elige una valoración de 1 a 5.",
      error: "No se pudo enviar. Inténtalo de nuevo.",
      offline: "Error de conexión. Comprueba tu internet.",
    },
    fr: {
      title: "Comment s’est passée la transcription ?",
      commentPh: "Que pouvons-nous améliorer ?",
      send: "Envoyer",
      dismiss: "Pas maintenant",
      thanks: "Merci pour votre avis !",
      pick: "Choisissez une note de 1 à 5.",
      error: "Envoi impossible. Réessayez.",
      offline: "Erreur de connexion. Vérifiez internet.",
    },
    de: {
      title: "Wie war die Transkription?",
      commentPh: "Was können wir verbessern?",
      send: "Senden",
      dismiss: "Nicht jetzt",
      thanks: "Danke für dein Feedback!",
      pick: "Bitte wähle eine Bewertung von 1 bis 5.",
      error: "Senden fehlgeschlagen. Bitte erneut versuchen.",
      offline: "Verbindungsfehler. Prüfe deine Internetverbindung.",
    },
  };

  function lang() {
    var code = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    return MSGS[code] ? code : "pt";
  }

  function t(key) {
    return (MSGS[lang()] || MSGS.pt)[key] || MSGS.pt[key];
  }

  function apiBase() {
    if (global.OuviescreviAPI && global.OuviescreviAPI.getBase) {
      return global.OuviescreviAPI.getBase();
    }
    if (global.OuviescreviAPI && global.OuviescreviAPI.detectApiBase) {
      return global.OuviescreviAPI.detectApiBase();
    }
    return "https://api.ouviescrevi.pt";
  }

  function toast(text, type) {
    if (global.OuviescreviUI && global.OuviescreviUI.toast) {
      global.OuviescreviUI.toast(text, type || "success");
    }
  }

  function els() {
    return {
      root: document.getElementById("oeCsatPulse"),
      title: document.getElementById("oeCsatTitle"),
      comment: document.getElementById("oeCsatComment"),
      send: document.getElementById("oeCsatSend"),
      dismiss: document.getElementById("oeCsatDismiss"),
      stars: document.querySelectorAll("#oeCsatPulse .oe-csat__star"),
      pub: document.getElementById("oeCsatPublic"),
      publish: document.getElementById("oeCsatPublish"),
      name: document.getElementById("oeCsatName"),
    };
  }

  function inCooldown() {
    try {
      var until = Number(localStorage.getItem(STORAGE_KEY) || 0);
      return until > Date.now();
    } catch (e) {
      return false;
    }
  }

  function setCooldown() {
    try {
      localStorage.setItem(STORAGE_KEY, String(Date.now() + COOLDOWN_MS));
    } catch (e) {}
  }

  function hide() {
    var e = els();
    if (!e.root) return;
    e.root.classList.add("hidden");
    e.root.hidden = true;
  }

  function show() {
    var e = els();
    if (!e.root) return;
    if (e.title) e.title.textContent = t("title");
    if (e.comment) {
      e.comment.placeholder = t("commentPh");
      e.comment.value = "";
    }
    if (e.send) {
      e.send.textContent = t("send");
      e.send.disabled = true;
    }
    if (e.dismiss) e.dismiss.textContent = t("dismiss");
    selectedRating = 0;
    e.stars.forEach(function (btn) {
      btn.classList.remove("is-active");
    });
    if (e.pub) e.pub.hidden = true;
    if (e.publish) e.publish.checked = false;
    if (e.name) e.name.value = "";
    e.root.classList.remove("hidden");
    e.root.hidden = false;
  }

  function paintStars(rating) {
    selectedRating = rating;
    var e = els();
    e.stars.forEach(function (btn) {
      var n = Number(btn.getAttribute("data-rating") || 0);
      btn.classList.toggle("is-active", n <= rating);
    });
    if (e.send) e.send.disabled = rating < 1;
    // Com 4-5 estrelas, pedir autorização para mostrar o comentário no site (só existe na PT).
    if (e.pub) e.pub.hidden = rating < 4;
    if (e.comment && lang() === "pt") {
      e.comment.placeholder = rating >= 4 ? "O que te ajudou mais?" : t("commentPh");
    }
  }

  function submit() {
    var e = els();
    if (selectedRating < 1 || selectedRating > 5) {
      toast(t("pick"), "warning");
      return;
    }
    var comment = ((e.comment && e.comment.value) || "").trim();
    var mensagem = comment || "Sem comentário.";
    if (e.send) e.send.disabled = true;

    fetch(apiBase() + "/api/suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "omit",
      body: JSON.stringify({
        nome: "CSAT",
        mensagem: mensagem,
        lang: lang(),
        source: "csat",
        rating: selectedRating,
        publicar: !!(e.publish && e.publish.checked && selectedRating >= 4 && comment),
        nome_publico: ((e.name && e.name.value) || "").trim(),
      }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error("fail");
        toast(t("thanks"), "success");
        setCooldown();
        hide();
      })
      .catch(function () {
        toast(t("offline"), "error");
        if (e.send) e.send.disabled = false;
      });
  }

  function dismiss() {
    setCooldown();
    hide();
  }

  function maybeAskAfterTranscription() {
    if (showTimer) {
      clearTimeout(showTimer);
      showTimer = null;
    }
    if (!document.getElementById("oeCsatPulse")) return;
    if (inCooldown()) return;
    showTimer = setTimeout(function () {
      showTimer = null;
      if (inCooldown()) return;
      show();
    }, SHOW_DELAY_MS);
  }

  /** Mostra na página inicial as opiniões aprovadas no backoffice (texto via textContent). */
  function loadTestimonials() {
    var box = document.getElementById("oeUserTestimonials");
    if (!box) return;
    fetch(apiBase() + "/api/testimonials?lang=" + lang(), { credentials: "omit" })
      .then(function (res) { return res.ok ? res.json() : { items: [] }; })
      .then(function (data) {
        var items = (data && data.items) || [];
        if (!items.length) return;
        var h = document.createElement("h2");
        h.className = "oe-home-usecases__title";
        h.textContent = "O que dizem os utilizadores";
        var grid = document.createElement("div");
        grid.className = "oe-home-usecases__grid";
        items.forEach(function (it) {
          var card = document.createElement("figure");
          card.className = "oe-home-usecases__card";
          card.style.margin = "0";
          var stars = document.createElement("span");
          stars.setAttribute("aria-label", it.rating + " de 5 estrelas");
          stars.style.color = "#b45309";
          stars.textContent = "★★★★★".slice(0, it.rating);
          var q = document.createElement("blockquote");
          q.style.margin = "6px 0";
          q.textContent = "«" + it.texto + "»";
          card.appendChild(stars);
          card.appendChild(q);
          if (it.nome) {
            var who = document.createElement("figcaption");
            who.textContent = "— " + it.nome;
            card.appendChild(who);
          }
          grid.appendChild(card);
        });
        box.appendChild(h);
        box.appendChild(grid);
      })
      .catch(function () {});
  }

  function init() {
    loadTestimonials();
    var e = els();
    if (!e.root) return;
    e.stars.forEach(function (btn) {
      btn.addEventListener("click", function () {
        paintStars(Number(btn.getAttribute("data-rating") || 0));
      });
    });
    if (e.send) e.send.addEventListener("click", submit);
    if (e.dismiss) e.dismiss.addEventListener("click", dismiss);
    hide();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  global.OuviescreviFeedback = {
    maybeAskAfterTranscription: maybeAskAfterTranscription,
    hide: hide,
  };
})(window);
