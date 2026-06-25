(function (global) {
  "use strict";

  var MSGS = {
    pt: {
      empty: "Escreve algo antes de enviar.",
      sending: "A enviar...",
      thanks: "Obrigado pela tua sugestão!",
      error: "Erro ao enviar. Tenta novamente.",
      offline: "Erro de ligação. Verifica a internet.",
    },
    es: {
      empty: "Escribe algo antes de enviar.",
      sending: "Enviando...",
      thanks: "¡Gracias por tu sugerencia!",
      error: "Error al enviar. Inténtalo de nuevo.",
      offline: "Error de conexión. Comprueba tu internet.",
    },
    fr: {
      empty: "Écris quelque chose avant d'envoyer.",
      sending: "Envoi...",
      thanks: "Merci pour ta suggestion !",
      error: "Erreur lors de l'envoi. Réessaie.",
      offline: "Erreur de connexion. Vérifie ta connexion internet.",
    },
    de: {
      empty: "Schreib etwas, bevor du sendest.",
      sending: "Senden...",
      thanks: "Danke für deinen Vorschlag!",
      error: "Fehler beim Senden. Bitte erneut versuchen.",
      offline: "Verbindungsfehler. Prüfe deine Internetverbindung.",
    },
    en: {
      empty: "Write something before sending.",
      sending: "Sending...",
      thanks: "Thanks for your suggestion!",
      error: "Could not send. Please try again.",
      offline: "Connection error. Check your internet.",
    },
  };

  function lang() {
    var code = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    return MSGS[code] ? code : "pt";
  }

  function msg(key) {
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
    } else {
      global.alert(text);
    }
  }

  function init() {
    var fab = document.getElementById("fabSugestao");
    var painel = document.getElementById("sugestaoPanel");
    var enviarBtn = document.getElementById("enviarSugestao");
    var fecharBtn = document.getElementById("fecharSugestao");
    var textarea = document.getElementById("sugestaoTexto");
    if (!fab || !painel || !enviarBtn || !textarea) return;

    fab.addEventListener("click", function () {
      var open = painel.classList.toggle("show");
      fab.setAttribute("aria-expanded", open ? "true" : "false");
    });

    if (fecharBtn) {
      fecharBtn.addEventListener("click", function () {
        painel.classList.remove("show");
        fab.setAttribute("aria-expanded", "false");
      });
    }

    enviarBtn.addEventListener("click", function () {
      var txt = textarea.value.trim();
      if (!txt) {
        toast(msg("empty"), "warning");
        return;
      }
      if (global.OuviescreviUI && global.OuviescreviUI.setButtonLoading) {
        global.OuviescreviUI.setButtonLoading(enviarBtn, true, msg("sending"));
      } else {
        enviarBtn.disabled = true;
      }

      fetch(apiBase() + "/api/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "omit",
        body: JSON.stringify({ mensagem: txt, lang: lang() }),
      })
        .then(function (response) {
          if (response.ok) {
            toast(msg("thanks"), "success");
            textarea.value = "";
            painel.classList.remove("show");
            fab.setAttribute("aria-expanded", "false");
          } else {
            toast(msg("error"), "error");
          }
        })
        .catch(function () {
          toast(msg("offline"), "error");
        })
        .finally(function () {
          if (global.OuviescreviUI && global.OuviescreviUI.setButtonLoading) {
            global.OuviescreviUI.setButtonLoading(enviarBtn, false);
          } else {
            enviarBtn.disabled = false;
          }
        });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(window);
