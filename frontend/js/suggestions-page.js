/**
 * Página pública de sugestões (PT + locales).
 */
(function (global) {
  "use strict";

  var MAX_LEN = 2000;
  var CAT_LABEL = {
    ideia: { pt: "Ideia", en: "Idea", es: "Idea", fr: "Idée", de: "Idee" },
    bug: { pt: "Problema", en: "Issue", es: "Problema", fr: "Problème", de: "Problem" },
    ux: { pt: "Usabilidade", en: "Usability", es: "Usabilidad", fr: "Ergonomie", de: "Bedienung" },
    outro: { pt: "Outro", en: "Other", es: "Otro", fr: "Autre", de: "Sonstiges" },
  };

  var COPY = {
    pt: {
      sending: "A enviar...",
      thanks: "Obrigado pela tua sugestão!",
      empty: "Escreve a tua sugestão antes de enviar.",
      error: "Erro ao enviar. Tenta novamente.",
      offline: "Erro de ligação. Verifica a internet.",
    },
    en: {
      sending: "Sending...",
      thanks: "Thanks for your suggestion!",
      empty: "Write your suggestion before sending.",
      error: "Could not send. Please try again.",
      offline: "Connection error. Check your internet.",
    },
    es: {
      sending: "Enviando...",
      thanks: "¡Gracias por tu sugerencia!",
      empty: "Escribe tu sugerencia antes de enviar.",
      error: "Error al enviar. Inténtalo de nuevo.",
      offline: "Error de conexión. Comprueba tu internet.",
    },
    fr: {
      sending: "Envoi...",
      thanks: "Merci pour ta suggestion !",
      empty: "Écris ta suggestion avant d’envoyer.",
      error: "Erreur lors de l’envoi. Réessaie.",
      offline: "Erreur de connexion. Vérifie internet.",
    },
    de: {
      sending: "Senden...",
      thanks: "Danke für deinen Vorschlag!",
      empty: "Schreib deinen Vorschlag vor dem Senden.",
      error: "Senden fehlgeschlagen. Bitte erneut versuchen.",
      offline: "Verbindungsfehler. Prüfe deine Internetverbindung.",
    },
  };

  function lang() {
    var code = (document.documentElement.lang || "pt").slice(0, 2).toLowerCase();
    return COPY[code] ? code : "pt";
  }

  function t(key) {
    return (COPY[lang()] || COPY.pt)[key] || COPY.pt[key];
  }

  function apiBase() {
    if (global.OuviescreviAPI && global.OuviescreviAPI.getBase) {
      return global.OuviescreviAPI.getBase() || global.OuviescreviAPI.detectApiBase();
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

  function selectedCategory(form) {
    var checked = form.querySelector('input[name="categoria"]:checked');
    return checked ? checked.value : "ideia";
  }

  function formatMessage(cat, text) {
    var labels = CAT_LABEL[cat] || CAT_LABEL.ideia;
    var label = labels[lang()] || labels.pt;
    return "[" + label + "] " + text;
  }

  function updateCounter(ta, meta) {
    if (!ta || !meta) return;
    var n = (ta.value || "").length;
    meta.textContent = n + " / " + MAX_LEN;
    meta.classList.toggle("is-warn", n > MAX_LEN - 120);
  }

  function showSuccess(form, success) {
    if (form) form.hidden = true;
    if (success) {
      success.hidden = false;
      success.classList.remove("hidden");
    }
  }

  function init() {
    var form = document.getElementById("formSugestao");
    if (!form) return;
    var btn = document.getElementById("btnSugestao");
    var nomeEl = document.getElementById("nome");
    var msgEl = document.getElementById("mensagem");
    var hpEl = document.getElementById("sugWebsite");
    var meta = document.getElementById("sugCharCount");
    var success = document.getElementById("sugSuccess");
    var again = document.getElementById("sugAgain");

    if (msgEl && meta) {
      updateCounter(msgEl, meta);
      msgEl.addEventListener("input", function () {
        updateCounter(msgEl, meta);
      });
    }

    if (again && success && form) {
      again.addEventListener("click", function () {
        success.hidden = true;
        success.classList.add("hidden");
        form.hidden = false;
        if (msgEl) msgEl.focus();
      });
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var mensagem = ((msgEl && msgEl.value) || "").trim();
      if (!mensagem) {
        toast(t("empty"), "warning");
        return;
      }
      if (mensagem.length > MAX_LEN) {
        mensagem = mensagem.slice(0, MAX_LEN);
      }
      var nome = ((nomeEl && nomeEl.value) || "").trim();
      var honeypot = ((hpEl && hpEl.value) || "").trim();
      var cat = selectedCategory(form);

      if (global.OuviescreviUI && global.OuviescreviUI.setButtonLoading) {
        global.OuviescreviUI.setButtonLoading(btn, true, t("sending"));
      } else if (btn) {
        btn.disabled = true;
      }

      var ready = Promise.resolve();
      if (global.OuviescreviAPI && global.OuviescreviAPI.init) {
        ready = global.OuviescreviAPI.init().catch(function () {});
      }

      ready.then(function () {
        return fetch(apiBase() + "/api/suggestions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "omit",
          body: JSON.stringify({
            nome: nome || null,
            mensagem: formatMessage(cat, mensagem),
            lang: lang(),
            source: "suggestions_page",
            honeypot: honeypot || null,
          }),
        });
      })
        .then(function (response) {
          if (response.ok) {
            toast(t("thanks"), "success");
            form.reset();
            var defaultCat = form.querySelector('input[name="categoria"][value="ideia"]');
            if (defaultCat) defaultCat.checked = true;
            updateCounter(msgEl, meta);
            showSuccess(form, success);
          } else {
            toast(t("error"), "error");
          }
        })
        .catch(function () {
          toast(t("offline"), "error");
        })
        .finally(function () {
          if (global.OuviescreviUI && global.OuviescreviUI.setButtonLoading) {
            global.OuviescreviUI.setButtonLoading(btn, false);
          } else if (btn) {
            btn.disabled = false;
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
