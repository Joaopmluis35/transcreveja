/**
 * Jogo de quiz animado — usa as perguntas de escolha múltipla (Aula Pronta).
 * OuviescreviQuizGame.open(questions, { title, lang })
 * questions: [{ prompt, options: [{ letter, text }], answer: "A".."D", explanation }]
 */
(function (global) {
  var SECONDS = 20;
  var BASE_POINTS = 100;
  var SHAPES = ["▲", "◆", "●", "■"];

  var STRINGS = {
    pt: {
      play: "🎮 Jogar quiz",
      start: "Começar",
      questionsCount: "{n} perguntas · {s} s cada",
      rules: "Responde depressa para ganhar mais pontos. Acertos seguidos dão bónus 🔥",
      question: "Pergunta {i} de {n}",
      correct: "Certo!",
      wrong: "Errado",
      timeout: "O tempo acabou",
      answerWas: "Resposta certa:",
      next: "Seguinte",
      finish: "Ver resultado",
      score: "Pontos",
      streak: "Sequência",
      resultTitle: "Resultado",
      resultLine: "Acertaste {c} de {n}",
      review: "Rever erros",
      noMistakes: "Sem erros — perfeito! 🏆",
      again: "Jogar outra vez",
      close: "Fechar",
      share: "🔗 Partilhar jogo",
      shareCopied: "Link copiado — envia-o a quem quiseres desafiar!",
      shareFail: "Não foi possível criar o link. Tenta outra vez.",
      shareText: "Desafio-te para este quiz: «{t}»",
      sound: "Som",
      keysHint: "Teclas 1–4 para responder · Enter para avançar · Esc para sair",
      nameLabel: "O teu nome",
      namePlaceholder: "Ex.: Maria",
      ranking: "🏆 Ranking",
      rankingLocal: "🏆 Ranking neste dispositivo",
      rankingOnlineHint: "Partilha o jogo para ter um ranking com outras pessoas.",
      rankingEmpty: "Ainda ninguém jogou — sê o primeiro!",
      position: "Ficaste em {p}.º de {n}",
      you: "tu",
      saving: "A guardar o resultado…",
      saveFail: "Não foi possível guardar no ranking online.",
      praise: ["Continua a estudar 📚", "Bom começo 👍", "Muito bem! 👏", "Excelente! 🌟", "Perfeito! 🏆"],
    },
    en: {
      play: "🎮 Play quiz",
      start: "Start",
      questionsCount: "{n} questions · {s} s each",
      rules: "Answer fast to score more. Answer streaks give a bonus 🔥",
      question: "Question {i} of {n}",
      correct: "Correct!",
      wrong: "Wrong",
      timeout: "Time's up",
      answerWas: "Correct answer:",
      next: "Next",
      finish: "See result",
      score: "Score",
      streak: "Streak",
      resultTitle: "Result",
      resultLine: "You got {c} of {n}",
      review: "Review mistakes",
      noMistakes: "No mistakes — perfect! 🏆",
      again: "Play again",
      close: "Close",
      share: "🔗 Share game",
      shareCopied: "Link copied — send it to whoever you want to challenge!",
      shareFail: "Could not create the link. Try again.",
      shareText: "I challenge you to this quiz: “{t}”",
      sound: "Sound",
      keysHint: "Keys 1–4 to answer · Enter to continue · Esc to exit",
      nameLabel: "Your name",
      namePlaceholder: "e.g. Maria",
      ranking: "🏆 Leaderboard",
      rankingLocal: "🏆 Leaderboard on this device",
      rankingOnlineHint: "Share the game to get a leaderboard with other people.",
      rankingEmpty: "Nobody has played yet — be the first!",
      position: "You are #{p} of {n}",
      you: "you",
      saving: "Saving your result…",
      saveFail: "Could not save to the online leaderboard.",
      praise: ["Keep studying 📚", "Good start 👍", "Well done! 👏", "Excellent! 🌟", "Perfect! 🏆"],
    },
    es: {
      play: "🎮 Jugar quiz",
      start: "Empezar",
      questionsCount: "{n} preguntas · {s} s cada una",
      rules: "Responde rápido para ganar más puntos. Los aciertos seguidos dan bonus 🔥",
      question: "Pregunta {i} de {n}",
      correct: "¡Correcto!",
      wrong: "Incorrecto",
      timeout: "Se acabó el tiempo",
      answerWas: "Respuesta correcta:",
      next: "Siguiente",
      finish: "Ver resultado",
      score: "Puntos",
      streak: "Racha",
      resultTitle: "Resultado",
      resultLine: "Acertaste {c} de {n}",
      review: "Revisar errores",
      noMistakes: "Sin errores — ¡perfecto! 🏆",
      again: "Jugar otra vez",
      close: "Cerrar",
      share: "🔗 Compartir juego",
      shareCopied: "Enlace copiado — ¡envíalo a quien quieras retar!",
      shareFail: "No se pudo crear el enlace. Inténtalo de nuevo.",
      shareText: "Te reto a este quiz: «{t}»",
      sound: "Sonido",
      keysHint: "Teclas 1–4 para responder · Enter para seguir · Esc para salir",
      nameLabel: "Tu nombre",
      namePlaceholder: "Ej.: María",
      ranking: "🏆 Clasificación",
      rankingLocal: "🏆 Clasificación en este dispositivo",
      rankingOnlineHint: "Comparte el juego para tener una clasificación con otras personas.",
      rankingEmpty: "Nadie ha jugado aún — ¡sé el primero!",
      position: "Quedaste {p}.º de {n}",
      you: "tú",
      saving: "Guardando el resultado…",
      saveFail: "No se pudo guardar en la clasificación online.",
      praise: ["Sigue estudiando 📚", "Buen comienzo 👍", "¡Muy bien! 👏", "¡Excelente! 🌟", "¡Perfecto! 🏆"],
    },
    fr: {
      play: "🎮 Jouer au quiz",
      start: "Commencer",
      questionsCount: "{n} questions · {s} s chacune",
      rules: "Réponds vite pour gagner plus de points. Les bonnes réponses d'affilée donnent un bonus 🔥",
      question: "Question {i} sur {n}",
      correct: "Correct !",
      wrong: "Faux",
      timeout: "Temps écoulé",
      answerWas: "Bonne réponse :",
      next: "Suivant",
      finish: "Voir le résultat",
      score: "Points",
      streak: "Série",
      resultTitle: "Résultat",
      resultLine: "{c} bonnes réponses sur {n}",
      review: "Revoir les erreurs",
      noMistakes: "Aucune erreur — parfait ! 🏆",
      again: "Rejouer",
      close: "Fermer",
      share: "🔗 Partager le jeu",
      shareCopied: "Lien copié — envoie-le à qui tu veux défier !",
      shareFail: "Impossible de créer le lien. Réessaie.",
      shareText: "Je te défie sur ce quiz : « {t} »",
      sound: "Son",
      keysHint: "Touches 1–4 pour répondre · Entrée pour continuer · Échap pour quitter",
      nameLabel: "Ton prénom",
      namePlaceholder: "Ex. : Marie",
      ranking: "🏆 Classement",
      rankingLocal: "🏆 Classement sur cet appareil",
      rankingOnlineHint: "Partage le jeu pour avoir un classement avec d'autres personnes.",
      rankingEmpty: "Personne n'a encore joué — sois le premier !",
      position: "Tu es {p}e sur {n}",
      you: "toi",
      saving: "Enregistrement du résultat…",
      saveFail: "Impossible d'enregistrer dans le classement en ligne.",
      praise: ["Continue à réviser 📚", "Bon début 👍", "Très bien ! 👏", "Excellent ! 🌟", "Parfait ! 🏆"],
    },
    de: {
      play: "🎮 Quiz spielen",
      start: "Starten",
      questionsCount: "{n} Fragen · je {s} s",
      rules: "Antworte schnell für mehr Punkte. Richtige Antworten in Folge geben Bonus 🔥",
      question: "Frage {i} von {n}",
      correct: "Richtig!",
      wrong: "Falsch",
      timeout: "Zeit abgelaufen",
      answerWas: "Richtige Antwort:",
      next: "Weiter",
      finish: "Ergebnis ansehen",
      score: "Punkte",
      streak: "Serie",
      resultTitle: "Ergebnis",
      resultLine: "{c} von {n} richtig",
      review: "Fehler ansehen",
      noMistakes: "Keine Fehler — perfekt! 🏆",
      again: "Nochmal spielen",
      close: "Schließen",
      share: "🔗 Spiel teilen",
      shareCopied: "Link kopiert — schick ihn an alle, die du herausfordern willst!",
      shareFail: "Link konnte nicht erstellt werden. Versuch es nochmal.",
      shareText: "Ich fordere dich zu diesem Quiz heraus: „{t}“",
      sound: "Ton",
      keysHint: "Tasten 1–4 zum Antworten · Enter zum Weitermachen · Esc zum Beenden",
      nameLabel: "Dein Name",
      namePlaceholder: "z. B. Maria",
      ranking: "🏆 Rangliste",
      rankingLocal: "🏆 Rangliste auf diesem Gerät",
      rankingOnlineHint: "Teile das Spiel, um eine Rangliste mit anderen zu haben.",
      rankingEmpty: "Noch niemand hat gespielt — sei der Erste!",
      position: "Du bist Platz {p} von {n}",
      you: "du",
      saving: "Ergebnis wird gespeichert…",
      saveFail: "Konnte nicht in der Online-Rangliste speichern.",
      praise: ["Weiter lernen 📚", "Guter Anfang 👍", "Sehr gut! 👏", "Ausgezeichnet! 🌟", "Perfekt! 🏆"],
    },
  };

  var state = null;
  var root = null;
  var lastFocus = null;
  var audioCtx = null;
  var soundOn = true;

  try {
    soundOn = localStorage.getItem("oe_quiz_sound") !== "0";
  } catch (e) {}

  function s(key, vars) {
    var loc = STRINGS[(state && state.lang) || "pt"] || STRINGS.pt;
    var out = loc[key] != null ? loc[key] : STRINGS.pt[key];
    if (typeof out === "string" && vars) {
      Object.keys(vars).forEach(function (k) {
        out = out.replace("{" + k + "}", vars[k]);
      });
    }
    return out;
  }

  function esc(v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function reducedMotion() {
    return global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function shuffle(list) {
    var a = list.slice();
    for (var i = a.length - 1; i > 0; i--) {
      var j = Math.floor(Math.random() * (i + 1));
      var tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
    }
    return a;
  }

  // ── Som (WebAudio, sem ficheiros) ────────────────────────────────────────
  function beep(freqs, dur) {
    if (!soundOn) return;
    try {
      audioCtx = audioCtx || new (global.AudioContext || global.webkitAudioContext)();
      var t0 = audioCtx.currentTime;
      freqs.forEach(function (f, i) {
        var o = audioCtx.createOscillator();
        var g = audioCtx.createGain();
        o.type = "triangle";
        o.frequency.value = f;
        var start = t0 + i * dur;
        g.gain.setValueAtTime(0.0001, start);
        g.gain.exponentialRampToValueAtTime(0.18, start + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
        o.connect(g).connect(audioCtx.destination);
        o.start(start);
        o.stop(start + dur + 0.02);
      });
    } catch (e) {}
  }

  // ── Confetti (canvas) ────────────────────────────────────────────────────
  function confetti(amount) {
    if (reducedMotion() || !root) return;
    var canvas = root.querySelector(".oe-qg__confetti");
    var ctx = canvas.getContext("2d");
    var w = (canvas.width = root.clientWidth);
    var h = (canvas.height = root.clientHeight);
    var colors = ["#a855f7", "#22c55e", "#facc15", "#3b82f6", "#ef4444", "#f472b6"];
    var parts = [];
    for (var i = 0; i < amount; i++) {
      parts.push({
        x: w / 2 + (Math.random() - 0.5) * w * 0.3,
        y: h * 0.45,
        vx: (Math.random() - 0.5) * 14,
        vy: -Math.random() * 14 - 4,
        r: Math.random() * 6 + 4,
        c: colors[i % colors.length],
        a: Math.random() * Math.PI,
        va: (Math.random() - 0.5) * 0.3,
      });
    }
    var frames = 0;
    (function tick() {
      ctx.clearRect(0, 0, w, h);
      parts.forEach(function (p) {
        p.vy += 0.35;
        p.x += p.vx;
        p.y += p.vy;
        p.a += p.va;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.a);
        ctx.fillStyle = p.c;
        ctx.fillRect(-p.r / 2, -p.r / 2, p.r, p.r * 0.6);
        ctx.restore();
      });
      frames++;
      if (frames < 110 && root) requestAnimationFrame(tick);
      else ctx.clearRect(0, 0, w, h);
    })();
  }

  // ── Ecrãs ────────────────────────────────────────────────────────────────
  function render(html) {
    var stage = root.querySelector(".oe-qg__stage");
    stage.innerHTML = html;
    stage.classList.remove("oe-qg__stage--in");
    void stage.offsetWidth;
    stage.classList.add("oe-qg__stage--in");
    var focusTarget = stage.querySelector("[data-qg-focus]") || stage.querySelector("button");
    if (focusTarget) focusTarget.focus();
  }

  function hud() {
    return (
      '<div class="oe-qg__hud">' +
      '<span class="oe-qg__pill">' + esc(s("score")) + ' <strong data-qg-score>' + state.score + "</strong></span>" +
      '<span class="oe-qg__pill oe-qg__pill--streak' + (state.streak >= 2 ? " is-hot" : "") + '">🔥 ' +
      '<strong>' + state.streak + "</strong></span>" +
      "</div>"
    );
  }

  function showIntro() {
    clearTimer();
    render(
      '<div class="oe-qg__intro">' +
      '<div class="oe-qg__logo" aria-hidden="true">🎮</div>' +
      '<h2 class="oe-qg__title">' + esc(state.title) + "</h2>" +
      '<p class="oe-qg__meta">' + esc(s("questionsCount", { n: state.questions.length, s: SECONDS })) + "</p>" +
      '<p class="oe-qg__rules">' + esc(s("rules")) + "</p>" +
      '<form class="oe-qg__name" data-qg-name-form novalidate>' +
      '<label class="oe-qg__name-label" for="oeQgName">' + esc(s("nameLabel")) + "</label>" +
      '<input class="oe-qg__name-input" id="oeQgName" type="text" maxlength="24" autocomplete="nickname" ' +
      'placeholder="' + esc(s("namePlaceholder")) + '" value="' + esc(state.player || savedName()) + '" data-qg-focus>' +
      '<button type="submit" class="oe-qg__btn oe-qg__btn--go" data-qg-start>' + esc(s("start")) + " ▶</button>" +
      "</form>" +
      '<p class="oe-qg__keys">' + esc(s("keysHint")) + "</p>" +
      "</div>"
    );
    var input = root.querySelector("#oeQgName");
    var go = root.querySelector("[data-qg-start]");
    var sync = function () {
      go.disabled = !input.value.trim();
    };
    input.addEventListener("input", sync);
    sync();
    root.querySelector("[data-qg-name-form]").addEventListener("submit", function (e) {
      e.preventDefault();
      var name = input.value.replace(/\s+/g, " ").trim().slice(0, 24);
      if (!name) return input.focus();
      state.player = name;
      try {
        localStorage.setItem("oe_quiz_name", name);
      } catch (err) {}
      beep([523, 659], 0.09);
      showQuestion(0);
    });
  }

  function savedName() {
    try {
      return localStorage.getItem("oe_quiz_name") || "";
    } catch (e) {
      return "";
    }
  }

  function showQuestion(i) {
    state.index = i;
    state.locked = false;
    var q = state.questions[i];
    var n = state.questions.length;
    var dots = state.questions
      .map(function (_, k) {
        var cls = k < i ? (state.results[k] ? " is-ok" : " is-bad") : k === i ? " is-now" : "";
        return '<span class="oe-qg__dot' + cls + '"></span>';
      })
      .join("");
    var opts = q.shuffled
      .map(function (o, k) {
        return (
          '<button type="button" class="oe-qg__btn oe-qg__opt oe-qg__opt--' + k + '" data-qg-letter="' + esc(o.letter) + '">' +
          '<span class="oe-qg__shape" aria-hidden="true">' + SHAPES[k] + "</span>" +
          '<span class="oe-qg__key" aria-hidden="true">' + (k + 1) + "</span>" +
          '<span class="oe-qg__opt-text">' + esc(o.text) + "</span>" +
          "</button>"
        );
      })
      .join("");
    render(
      hud() +
      '<div class="oe-qg__progress" aria-hidden="true">' + dots + "</div>" +
      '<p class="oe-qg__count">' + esc(s("question", { i: i + 1, n: n })) + "</p>" +
      '<h2 class="oe-qg__prompt" tabindex="-1" data-qg-focus>' + esc(q.prompt) + "</h2>" +
      '<div class="oe-qg__timer" aria-hidden="true"><div class="oe-qg__timer-bar"></div></div>' +
      '<div class="oe-qg__opts">' + opts + "</div>" +
      '<div class="oe-qg__feedback" aria-live="assertive"></div>'
    );
    root.querySelectorAll("[data-qg-letter]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        answer(btn.getAttribute("data-qg-letter"));
      });
    });
    startTimer();
  }

  function startTimer() {
    clearTimer();
    var bar = root.querySelector(".oe-qg__timer-bar");
    state.deadline = Date.now() + SECONDS * 1000;
    state.timer = setInterval(function () {
      var left = Math.max(0, state.deadline - Date.now());
      var pct = (left / (SECONDS * 1000)) * 100;
      if (bar) {
        bar.style.width = pct + "%";
        bar.classList.toggle("is-low", pct < 30);
      }
      if (left <= 0) answer(null);
    }, 100);
  }

  function clearTimer() {
    if (state && state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
  }

  function answer(letter) {
    if (state.locked) return;
    state.locked = true;
    clearTimer();
    var q = state.questions[state.index];
    var ok = letter != null && letter === q.answer;
    var left = Math.max(0, state.deadline - Date.now()) / 1000;
    var gained = 0;
    if (ok) {
      state.streak += 1;
      // Rapidez dá até +100; a sequência multiplica (máx. x2).
      gained = Math.round((BASE_POINTS + (left / SECONDS) * 100) * Math.min(2, 1 + (state.streak - 1) * 0.25));
      state.score += gained;
      state.correct += 1;
    } else {
      state.streak = 0;
      state.mistakes.push({ q: q, picked: letter });
    }
    state.results[state.index] = ok;

    root.querySelectorAll("[data-qg-letter]").forEach(function (btn) {
      var l = btn.getAttribute("data-qg-letter");
      btn.disabled = true;
      if (l === q.answer) btn.classList.add("is-correct");
      else if (l === letter) btn.classList.add("is-wrong");
      else btn.classList.add("is-dim");
    });

    var correctOpt = q.options.filter(function (o) { return o.letter === q.answer; })[0];
    var last = state.index === state.questions.length - 1;
    var fb = root.querySelector(".oe-qg__feedback");
    fb.className = "oe-qg__feedback " + (ok ? "is-ok" : "is-bad");
    fb.innerHTML =
      '<p class="oe-qg__verdict">' +
      (ok ? "✅ " + esc(s("correct")) : letter == null ? "⏰ " + esc(s("timeout")) : "❌ " + esc(s("wrong"))) +
      (ok ? ' <span class="oe-qg__gain">+' + gained + "</span>" : "") +
      "</p>" +
      (!ok && correctOpt ? '<p class="oe-qg__answer">' + esc(s("answerWas")) + " <strong>" + esc(correctOpt.text) + "</strong></p>" : "") +
      (q.explanation ? '<p class="oe-qg__expl">' + esc(q.explanation) + "</p>" : "") +
      '<button type="button" class="oe-qg__btn oe-qg__btn--next" data-qg-next>' + esc(last ? s("finish") : s("next")) + " →</button>";

    var scoreEl = root.querySelector("[data-qg-score]");
    if (scoreEl) {
      scoreEl.textContent = state.score;
      scoreEl.parentNode.classList.add("is-bump");
    }
    var streakPill = root.querySelector(".oe-qg__pill--streak");
    if (streakPill) {
      streakPill.querySelector("strong").textContent = state.streak;
      streakPill.classList.toggle("is-hot", state.streak >= 2);
    }

    if (ok) {
      beep([660, 880, 1175], 0.08);
      confetti(state.streak >= 3 ? 90 : 45);
    } else {
      beep([220, 165], 0.14);
      var stage = root.querySelector(".oe-qg__stage");
      stage.classList.remove("oe-qg__stage--shake");
      void stage.offsetWidth;
      stage.classList.add("oe-qg__stage--shake");
    }

    var next = fb.querySelector("[data-qg-next]");
    next.addEventListener("click", function () {
      if (last) showResult();
      else showQuestion(state.index + 1);
    });
    next.focus();
  }

  function showResult() {
    clearTimer();
    var n = state.questions.length;
    var ratio = n ? state.correct / n : 0;
    var stars = ratio >= 0.95 ? 3 : ratio >= 0.6 ? 2 : ratio >= 0.3 ? 1 : 0;
    var praise = s("praise")[ratio >= 0.95 ? 4 : ratio >= 0.7 ? 3 : ratio >= 0.5 ? 2 : ratio >= 0.25 ? 1 : 0];
    var starsHtml = [0, 1, 2]
      .map(function (k) {
        return '<span class="oe-qg__star' + (k < stars ? " is-on" : "") + '" style="animation-delay:' + (0.25 + k * 0.25) + 's">★</span>';
      })
      .join("");
    var review = state.mistakes.length
      ? state.mistakes
          .map(function (m) {
            var c = m.q.options.filter(function (o) { return o.letter === m.q.answer; })[0];
            return (
              '<li class="oe-qg__review-item"><p class="oe-qg__review-q">' + esc(m.q.prompt) + "</p>" +
              (c ? '<p class="oe-qg__review-a">✅ ' + esc(c.text) + "</p>" : "") +
              (m.q.explanation ? '<p class="oe-qg__review-e">' + esc(m.q.explanation) + "</p>" : "") +
              "</li>"
            );
          })
          .join("")
      : "";
    render(
      '<div class="oe-qg__result">' +
      '<p class="oe-qg__count">' + esc(s("resultTitle")) + "</p>" +
      '<div class="oe-qg__stars" aria-hidden="true">' + starsHtml + "</div>" +
      '<p class="oe-qg__big"><span data-qg-total>0</span> <small>' + esc(s("score").toLowerCase()) + "</small></p>" +
      '<p class="oe-qg__meta">' + esc(s("resultLine", { c: state.correct, n: n })) + " · " + esc(praise) + "</p>" +
      '<section class="oe-qg__ranking" data-qg-ranking aria-live="polite"></section>' +
      '<div class="oe-qg__actions">' +
      '<button type="button" class="oe-qg__btn oe-qg__btn--go" data-qg-again data-qg-focus>↻ ' + esc(s("again")) + "</button>" +
      '<button type="button" class="oe-qg__btn oe-qg__btn--ghost" data-qg-share>' + esc(s("share")) + "</button>" +
      '<button type="button" class="oe-qg__btn oe-qg__btn--ghost" data-qg-close>' + esc(s("close")) + "</button>" +
      "</div>" +
      (review
        ? '<details class="oe-qg__review"><summary>' + esc(s("review")) + " (" + state.mistakes.length + ")</summary><ol>" + review + "</ol></details>"
        : '<p class="oe-qg__perfect">' + esc(s("noMistakes")) + "</p>") +
      "</div>"
    );
    // Contagem animada dos pontos.
    var totalEl = root.querySelector("[data-qg-total]");
    var target = state.score;
    if (reducedMotion()) totalEl.textContent = target;
    else {
      var t0 = performance.now();
      (function step(now) {
        var p = Math.min(1, (now - t0) / 1200);
        totalEl.textContent = Math.round(target * (1 - Math.pow(1 - p, 3)));
        if (p < 1 && root) requestAnimationFrame(step);
      })(t0);
    }
    if (stars >= 2) {
      confetti(140);
      beep([523, 659, 784, 1047], 0.11);
    }
    root.querySelector("[data-qg-again]").addEventListener("click", function () {
      start(state.source, { title: state.title, lang: state.lang, shareUrl: state.shareUrl, onLink: state.onLink, quizId: state.quizId, onClose: state.onClose });
    });
    root.querySelector("[data-qg-share]").addEventListener("click", function () {
      shareQuiz(state.source, { title: state.title, lang: state.lang, shareUrl: state.shareUrl, onLink: state.onLink, button: this });
    });
    root.querySelector("[data-qg-close]").addEventListener("click", close);
    saveAndShowRanking();
  }

  // ── Ranking ──────────────────────────────────────────────────────────────
  // Jogo partilhado (tem id): ranking online. Sem link: ranking guardado neste dispositivo.
  function quizIdOf(st) {
    if (st.quizId) return st.quizId;
    var m = /[?&]id=([^&#]+)/.exec(st.shareUrl || "");
    return m ? decodeURIComponent(m[1]) : null;
  }

  function renderRanking(box, rows, opts) {
    if (!box || !root) return;
    var me = (state.player || "").toLowerCase();
    var medals = ["🥇", "🥈", "🥉"];
    var list = rows.length
      ? "<ol>" +
        rows
          .map(function (r, i) {
            var mine = String(r.name).toLowerCase() === me;
            return (
              '<li class="oe-qg__rank-row' + (mine ? " is-me" : "") + '">' +
              '<span class="oe-qg__rank-pos">' + (medals[i] || i + 1) + "</span>" +
              '<span class="oe-qg__rank-name">' + esc(r.name) + (mine ? " <em>(" + esc(s("you")) + ")</em>" : "") + "</span>" +
              '<span class="oe-qg__rank-hits">' + r.correct + "/" + r.total + "</span>" +
              '<strong class="oe-qg__rank-score">' + r.score + "</strong>" +
              "</li>"
            );
          })
          .join("") +
        "</ol>"
      : '<p class="oe-qg__rank-empty">' + esc(s("rankingEmpty")) + "</p>";
    box.innerHTML =
      "<h3>" + esc(opts.local ? s("rankingLocal") : s("ranking")) + "</h3>" +
      (opts.position ? '<p class="oe-qg__rank-me">' + esc(s("position", { p: opts.position, n: opts.players })) + "</p>" : "") +
      list +
      (opts.note ? '<p class="oe-qg__rank-note">' + esc(opts.note) + "</p>" : "");
  }

  function localRankingKey(st) {
    var src = st.title + "|" + st.questions.map(function (q) { return q.prompt; }).join("|");
    var h = 0;
    for (var i = 0; i < src.length; i++) h = (h * 31 + src.charCodeAt(i)) | 0;
    return "oe_quiz_rank_" + (h >>> 0).toString(36);
  }

  function saveLocalRanking(st, entry) {
    var key = localRankingKey(st);
    var rows = [];
    try {
      rows = JSON.parse(localStorage.getItem(key) || "[]");
    } catch (e) {}
    rows.push(entry);
    // Melhor resultado de cada nome.
    var best = {};
    rows.forEach(function (r) {
      var k = String(r.name).toLowerCase();
      if (!best[k] || r.score > best[k].score) best[k] = r;
    });
    var sorted = Object.keys(best)
      .map(function (k) { return best[k]; })
      .sort(function (a, b) { return b.score - a.score; });
    try {
      localStorage.setItem(key, JSON.stringify(sorted.slice(0, 50)));
    } catch (e) {}
    var pos = sorted.findIndex(function (r) { return String(r.name).toLowerCase() === String(entry.name).toLowerCase(); }) + 1;
    return { rows: sorted.slice(0, 10), position: pos, players: sorted.length };
  }

  async function saveAndShowRanking() {
    var st = state;
    var box = root && root.querySelector("[data-qg-ranking]");
    if (!box || !st.player) return;
    var entry = { name: st.player, score: st.score, correct: st.correct, total: st.questions.length };
    var quizId = quizIdOf(st);
    if (!quizId) {
      var local = saveLocalRanking(st, entry);
      renderRanking(box, local.rows, { local: true, position: local.position, players: local.players, note: s("rankingOnlineHint") });
      return;
    }
    box.innerHTML = '<p class="oe-qg__rank-note">' + esc(s("saving")) + "</p>";
    try {
      var api = global.OuviescreviAPI;
      await api.init();
      var res = await fetch(api.getBase() + "/api/share/quiz/" + encodeURIComponent(quizId) + "/scores", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(entry),
      });
      var data = await res.json();
      if (!res.ok) throw new Error(data.detail || "score");
      if (state !== st) return; // o jogador já fechou ou recomeçou
      renderRanking(box, data.ranking || [], { position: data.position, players: data.players });
    } catch (e) {
      if (state !== st) return;
      var fallback = saveLocalRanking(st, entry);
      renderRanking(box, fallback.rows, { local: true, position: fallback.position, players: fallback.players, note: s("saveFail") });
    }
  }

  // ── Abrir / fechar ───────────────────────────────────────────────────────
  function onKey(e) {
    if (!root) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
      return;
    }
    if (/^[1-4]$/.test(e.key)) {
      var btn = root.querySelectorAll("[data-qg-letter]")[parseInt(e.key, 10) - 1];
      if (btn && !btn.disabled) {
        e.preventDefault();
        btn.click();
      }
    }
    if (e.key === "Tab") {
      // Manter o foco dentro do jogo.
      var f = root.querySelectorAll("button:not([disabled]), input, summary, [tabindex='0']");
      if (!f.length) return;
      var first = f[0];
      var lastEl = f[f.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        first.focus();
      }
    }
  }

  function start(questions, opts) {
    var playable = (questions || []).filter(function (q) {
      return q && q.prompt && q.options && q.options.length >= 2 && q.answer;
    });
    state = {
      source: questions,
      lang: STRINGS[(opts && opts.lang) || "pt"] ? opts.lang : "pt",
      title: (opts && opts.title) || "Quiz",
      shareUrl: (opts && opts.shareUrl) || null,
      onLink: (opts && opts.onLink) || null,
      quizId: (opts && opts.quizId) || null,
      onClose: (opts && opts.onClose) || null,
      player: "",
      questions: playable.map(function (q) {
        return Object.assign({}, q, { shuffled: shuffle(q.options) });
      }),
      index: 0,
      score: 0,
      streak: 0,
      correct: 0,
      results: [],
      mistakes: [],
      timer: null,
      locked: false,
    };
    if (!root) build();
    showIntro();
  }

  function build() {
    lastFocus = document.activeElement;
    root = document.createElement("div");
    root.className = "oe-qg";
    root.setAttribute("role", "dialog");
    root.setAttribute("aria-modal", "true");
    root.setAttribute("aria-label", "Quiz");
    root.innerHTML =
      '<div class="oe-qg__bg" aria-hidden="true"><span></span><span></span><span></span></div>' +
      '<canvas class="oe-qg__confetti" aria-hidden="true"></canvas>' +
      '<div class="oe-qg__top">' +
      '<button type="button" class="oe-qg__btn oe-qg__icon" data-qg-sound aria-pressed="' + soundOn + '" title="' + esc(s("sound")) + '">' + (soundOn ? "🔊" : "🔇") + "</button>" +
      '<button type="button" class="oe-qg__btn oe-qg__icon" data-qg-x aria-label="' + esc(s("close")) + '" title="' + esc(s("close")) + '">✕</button>' +
      "</div>" +
      '<div class="oe-qg__stage"></div>';
    document.body.appendChild(root);
    document.documentElement.classList.add("oe-qg-open");
    root.querySelector("[data-qg-x]").addEventListener("click", close);
    root.querySelector("[data-qg-sound]").addEventListener("click", function () {
      soundOn = !soundOn;
      this.textContent = soundOn ? "🔊" : "🔇";
      this.setAttribute("aria-pressed", String(soundOn));
      try {
        localStorage.setItem("oe_quiz_sound", soundOn ? "1" : "0");
      } catch (e) {}
    });
    document.addEventListener("keydown", onKey);
  }

  function close() {
    var onClose = state && state.onClose;
    clearTimer();
    document.removeEventListener("keydown", onKey);
    if (root) root.remove();
    root = null;
    state = null;
    document.documentElement.classList.remove("oe-qg-open");
    if (lastFocus && lastFocus.focus) lastFocus.focus();
    if (onClose) onClose();
  }

  // ── Partilha ─────────────────────────────────────────────────────────────
  function toast(msg, type) {
    if (global.OuviescreviUI && global.OuviescreviUI.toast) global.OuviescreviUI.toast(msg, type);
    else global.alert(msg);
  }

  async function sendLink(url, title, lang) {
    var text = (STRINGS[lang] || STRINGS.pt).shareText.replace("{t}", title);
    if (navigator.share && /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent)) {
      try {
        await navigator.share({ title: title, text: text, url: url });
        return;
      } catch (e) {
        if (e && e.name === "AbortError") return;
      }
    }
    var copied = false;
    try {
      await navigator.clipboard.writeText(url);
      copied = true;
    } catch (e) {}
    if (copied) toast((STRINGS[lang] || STRINGS.pt).shareCopied, "success");
    else global.prompt(text, url);
  }

  // Guarda as perguntas na API (link curto, 30 dias) e partilha. Reaproveita o link se já existir.
  async function shareQuiz(questions, opts) {
    opts = opts || {};
    var lang = STRINGS[opts.lang] ? opts.lang : "pt";
    var title = opts.title || "Quiz";
    if (opts.shareUrl) return sendLink(opts.shareUrl, title, lang);
    var btn = opts.button;
    if (btn) btn.disabled = true;
    try {
      var api = global.OuviescreviAPI;
      await api.init();
      var res = await fetch(api.getBase() + "/api/share/quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title,
          locale: lang,
          questions: (questions || []).map(function (q) {
            return { prompt: q.prompt, options: q.options, answer: q.answer, explanation: q.explanation };
          }),
        }),
      });
      var data = await res.json();
      if (!res.ok || !data.url) throw new Error(data.detail || "share");
      if (state) state.shareUrl = data.url;
      rememberMyGame({ id: data.id, url: data.url, title: title, lang: lang, count: (questions || []).length });
      if (opts.onLink) opts.onLink(data.url);
      await sendLink(data.url, title, lang);
      return data.url;
    } catch (e) {
      toast((STRINGS[lang] || STRINGS.pt).shareFail, "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  // ── Jogos que partilhei (guardados neste dispositivo, para voltar ao ranking) ──
  var MY_GAMES_KEY = "oe_my_quiz_games";

  function myGames() {
    try {
      var list = JSON.parse(localStorage.getItem(MY_GAMES_KEY) || "[]");
      return Array.isArray(list) ? list : [];
    } catch (e) {
      return [];
    }
  }

  function saveMyGames(list) {
    try {
      localStorage.setItem(MY_GAMES_KEY, JSON.stringify(list.slice(0, 30)));
    } catch (e) {}
  }

  function rememberMyGame(game) {
    if (!game || !game.id) return;
    var list = myGames().filter(function (g) { return g.id !== game.id; });
    list.unshift(Object.assign({ created: new Date().toISOString() }, game));
    saveMyGames(list);
    try {
      document.dispatchEvent(new CustomEvent("oe-quiz-shared", { detail: game }));
    } catch (e) {}
  }

  function forgetMyGame(id) {
    saveMyGames(myGames().filter(function (g) { return g.id !== id; }));
  }

  global.OuviescreviQuizGame = {
    open: start,
    close: close,
    share: shareQuiz,
    myGames: myGames,
    forgetMyGame: forgetMyGame,
    shareLabel: function (lang) {
      return (STRINGS[lang] || STRINGS.pt).share;
    },
    label: function (lang) {
      return (STRINGS[lang] || STRINGS.pt).play;
    },
  };
})(typeof window !== "undefined" ? window : this);
