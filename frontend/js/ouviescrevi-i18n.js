/**
 * Locales do site — PT (raiz), EN/ES/FR/DE em subpastas.
 */
(function (global) {
  "use strict";

  var LOCALES = ["pt", "en", "es", "fr", "de"];

  var LOCALE_META = {
    pt: { label: "Português", flag: "/icons/pt.png?v=2", og: "pt_PT", html: "pt" },
    en: { label: "English", flag: "/icons/en.png?v=2", og: "en_GB", html: "en" },
    es: { label: "Español", flag: "/icons/es.png?v=2", og: "es_ES", html: "es" },
    fr: { label: "Français", flag: "/icons/fr.png?v=2", og: "fr_FR", html: "fr" },
    de: { label: "Deutsch", flag: "/icons/de.png?v=2", og: "de_DE", html: "de" },
  };

  /** Páginas equivalentes entre locales (slug lógico → ficheiro por locale). */
  var PAGES = {
    index: { pt: "index.html", en: "index.html", es: "index.html", fr: "index.html", de: "index.html" },
    ajuda: { pt: "ajuda.html", en: "ajuda.html", es: "ajuda.html", fr: "ajuda.html", de: "ajuda.html" },
    conversor: { pt: "conversor.html", en: "conversor.html", es: "conversor.html", fr: "conversor.html", de: "conversor.html" },
    "conversor-imagens": { pt: "conversor-imagens.html", en: "conversor-imagens.html", es: "conversor-imagens.html", fr: "conversor-imagens.html", de: "conversor-imagens.html" },
    sugestoes: { pt: "sugestoes.html", en: "sugestoes.html", es: "sugestoes.html", fr: "sugestoes.html", de: "sugestoes.html" },
    resumo: { pt: "resumo.html", en: "resumo.html", es: "resumo.html", fr: "resumo.html", de: "resumo.html" },
    "url-resumo": { pt: "url-resumo.html", en: "url-resumo.html", es: "url-resumo.html", fr: "url-resumo.html", de: "url-resumo.html" },
    perguntas: { pt: "perguntas.html", en: "perguntas.html", es: "perguntas.html", fr: "perguntas.html", de: "perguntas.html" },
    "aula-pronta": { pt: "aula-pronta.html", en: "aula-pronta.html", es: "aula-pronta.html", fr: "aula-pronta.html", de: "aula-pronta.html" },
    capitulos: { pt: "capitulos.html", en: "capitulos.html", es: "capitulos.html", fr: "capitulos.html", de: "capitulos.html" },
    flashcards: { pt: "flashcards.html", en: "flashcards.html", es: "flashcards.html", fr: "flashcards.html", de: "flashcards.html" },
    "aula-completa": { pt: "aula-completa.html", en: "aula-completa.html", es: "aula-completa.html", fr: "aula-completa.html", de: "aula-completa.html" },
    "podcast-youtube": { pt: "podcast-youtube.html", en: "podcast-youtube.html", es: "podcast-youtube.html", fr: "podcast-youtube.html", de: "podcast-youtube.html" },
    "descricao-youtube": { pt: "descricao-youtube.html", en: "descricao-youtube.html", es: "descricao-youtube.html", fr: "descricao-youtube.html", de: "descricao-youtube.html" },
    corretor: { pt: "corretor.html", en: "corretor.html", es: "corretor.html", fr: "corretor.html", de: "corretor.html" },
    professores: { pt: "professores.html", en: "professores.html" },
    podcasts: { pt: "podcasts.html", en: "podcasts.html" },
    aulas: { pt: "aulas.html", en: "aulas.html" },
    jornalistas: { pt: "jornalistas.html", en: "jornalistas.html" },
    reunioes: { pt: "reunioes.html", en: "reunioes.html" },
    testemunhos: { pt: "testemunhos.html", en: "testemunhos.html" },
    cookies: { pt: "cookies.html", en: "cookies.html", es: "cookies.html", fr: "cookies.html", de: "cookies.html" },
    privacy: { pt: "privacidade.html", en: "privacy.html", es: "privacy.html", fr: "privacy.html", de: "privacy.html" },
    terms: { pt: "termos.html", en: "terms.html", es: "terms.html", fr: "terms.html", de: "terms.html" },
    precos: { pt: "precos.html", en: "precos.html" },
  };

  function localeFromPath(path) {
    path = path || (global.location && global.location.pathname) || "/";
    var m = path.match(/^\/(en|es|fr|de)(\/|$)/);
    return m ? m[1] : "pt";
  }

  function localePrefix(locale) {
    if (!locale || locale === "pt") return "";
    return "/" + locale;
  }

  function pageSlugFromPath(path) {
    path = (path || "/").replace(/\/$/, "");
    var file = path.split("/").pop() || "index.html";
    if (file === "" || file === "index.html") return "index";
    var name = file.replace(".html", "");
    if (PAGES[name]) return name;
    if (name === "privacidade" || name === "privacy") return "privacy";
    if (name === "termos" || name === "terms") return "terms";
    if (name === "precos") return "precos";
    return null;
  }

  function pathFor(locale, slug) {
    var map = PAGES[slug];
    if (!map) return localePrefix(locale) + "/index.html";
    var useLocale = locale;
    if (!map[locale]) {
      if (map.en) useLocale = "en";
      else if (map.pt) useLocale = "pt";
    }
    var file = map[useLocale] || map.en || map.pt;
    var prefix = localePrefix(useLocale);
    return (prefix || "") + "/" + file;
  }

  function currentPagePath() {
    var path = global.location.pathname.replace(/\/$/, "") || "/index.html";
    if (path === "" || path === "/") return "/index.html";
    return path;
  }

  function hreflangMapForPath(path) {
    var slug = pageSlugFromPath(path);
    if (!slug) return null;
    var out = {};
    LOCALES.forEach(function (loc) {
      out[loc] = pathFor(loc, slug);
    });
    return out;
  }

  function switchLanguage(target) {
    if (LOCALES.indexOf(target) === -1) return;
    try {
      localStorage.setItem("lang", target);
    } catch (e) {}
    var slug = pageSlugFromPath(global.location.pathname) || "index";
    global.location.href = pathFor(target, slug);
  }

  function uiStrings(locale) {
    var L = {
      pt: {
        openMenu: "Abrir menu",
        closeMenu: "Fechar menu",
        cookieAria: "Aviso de cookies",
        cookieText: "Utilizamos armazenamento essencial no browser para o serviço funcionar. ",
        cookieLink: "Política de Cookies",
        cookieBtn: "Compreendi",
        cookiesPath: "/cookies.html",
      },
      en: {
        openMenu: "Open menu",
        closeMenu: "Close menu",
        cookieAria: "Cookie notice",
        cookieText: "We use essential browser storage for the service to work. ",
        cookieLink: "Cookie Policy",
        cookieBtn: "OK",
        cookiesPath: "/en/cookies.html",
      },
      es: {
        openMenu: "Abrir menú",
        closeMenu: "Cerrar menú",
        cookieAria: "Aviso de cookies",
        cookieText: "Usamos almacenamiento esencial en el navegador para que el servicio funcione. ",
        cookieLink: "Política de cookies",
        cookieBtn: "Entendido",
        cookiesPath: "/es/cookies.html",
      },
      fr: {
        openMenu: "Ouvrir le menu",
        closeMenu: "Fermer le menu",
        cookieAria: "Avis sur les cookies",
        cookieText: "Nous utilisons un stockage essentiel dans le navigateur pour faire fonctionner le service. ",
        cookieLink: "Politique de cookies",
        cookieBtn: "Compris",
        cookiesPath: "/fr/cookies.html",
      },
      de: {
        openMenu: "Menü öffnen",
        closeMenu: "Menü schließen",
        cookieAria: "Cookie-Hinweis",
        cookieText: "Wir verwenden wesentlichen Browserspeicher, damit der Dienst funktioniert. ",
        cookieLink: "Cookie-Richtlinie",
        cookieBtn: "Verstanden",
        cookiesPath: "/de/cookies.html",
      },
    };
    return L[locale] || L.en;
  }

  function authStrings(locale) {
    var L = {
      pt: {
        close: "Fechar",
        titleLogin: "Entrar na conta",
        titleRegister: "Criar conta",
        titleAdmin: "Entrar como administrador",
        tabLogin: "Entrar",
        tabRegister: "Registar",
        tabAdmin: "Admin",
        email: "Email",
        password: "Palavra-passe",
        passwordMin: "Palavra-passe (mín. 8)",
        nameOptional: "Nome (opcional)",
        username: "Utilizador",
        loginBtn: "Entrar",
        registerBtn: "Criar conta",
        adminBtn: "Entrar como admin",
        registerHint: "Ao registares-te podes usar o site com a tua conta. Atividade normal envia notificação ao administrador.",
        adminHint: "Conta de equipa — atividade no site não envia emails de notificação.",
        accountLabel: "Conta",
        staffSuffix: " (equipa)",
        logoutToast: "Sessão terminada.",
        welcomeBack: "Bem-vindo de volta!",
        accountCreated: "Conta criada — 20 transcrições por dia!",
        adminSession: "Sessão de administrador ativa.",
        loginFail: "Não foi possível entrar.",
        registerFail: "Não foi possível registar.",
        invalidCreds: "Credenciais inválidas.",
        loginError: "Erro ao entrar.",
        registerError: "Erro ao registar.",
      },
      en: {
        close: "Close",
        titleLogin: "Log in to your account",
        titleRegister: "Create account",
        titleAdmin: "Admin sign in",
        tabLogin: "Log in",
        tabRegister: "Sign up",
        tabAdmin: "Admin",
        email: "Email",
        password: "Password",
        passwordMin: "Password (min. 8)",
        nameOptional: "Name (optional)",
        username: "Username",
        loginBtn: "Log in",
        registerBtn: "Create account",
        adminBtn: "Sign in as admin",
        registerHint: "Create an account to use the site. Normal activity sends a notification to the administrator.",
        adminHint: "Team account — site activity does not send notification emails.",
        accountLabel: "Account",
        staffSuffix: " (team)",
        logoutToast: "Signed out.",
        welcomeBack: "Welcome back!",
        accountCreated: "Account created — 20 transcriptions per day!",
        adminSession: "Admin session active.",
        loginFail: "Could not log in.",
        registerFail: "Could not sign up.",
        invalidCreds: "Invalid credentials.",
        loginError: "Error signing in.",
        registerError: "Error signing up.",
      },
      es: {
        close: "Cerrar",
        titleLogin: "Iniciar sesión",
        titleRegister: "Crear cuenta",
        titleAdmin: "Entrar como administrador",
        tabLogin: "Entrar",
        tabRegister: "Registrarse",
        tabAdmin: "Admin",
        email: "Email",
        password: "Contraseña",
        passwordMin: "Contraseña (mín. 8)",
        nameOptional: "Nombre (opcional)",
        username: "Usuario",
        loginBtn: "Entrar",
        registerBtn: "Crear cuenta",
        adminBtn: "Entrar como admin",
        registerHint: "Al registrarte puedes usar el sitio con tu cuenta. La actividad normal notifica al administrador.",
        adminHint: "Cuenta de equipo — la actividad en el sitio no envía emails de notificación.",
        accountLabel: "Cuenta",
        staffSuffix: " (equipo)",
        logoutToast: "Sesión cerrada.",
        welcomeBack: "¡Bienvenido de nuevo!",
        accountCreated: "Cuenta creada — ¡20 transcripciones al día!",
        adminSession: "Sesión de administrador activa.",
        loginFail: "No se pudo iniciar sesión.",
        registerFail: "No se pudo registrar.",
        invalidCreds: "Credenciales inválidas.",
        loginError: "Error al entrar.",
        registerError: "Error al registrarse.",
      },
      fr: {
        close: "Fermer",
        titleLogin: "Se connecter",
        titleRegister: "Créer un compte",
        titleAdmin: "Connexion administrateur",
        tabLogin: "Connexion",
        tabRegister: "Inscription",
        tabAdmin: "Admin",
        email: "Email",
        password: "Mot de passe",
        passwordMin: "Mot de passe (min. 8)",
        nameOptional: "Nom (optionnel)",
        username: "Utilisateur",
        loginBtn: "Se connecter",
        registerBtn: "Créer un compte",
        adminBtn: "Connexion admin",
        registerHint: "Créez un compte pour utiliser le site. L'activité normale notifie l'administrateur.",
        adminHint: "Compte équipe — l'activité sur le site n'envoie pas d'emails de notification.",
        accountLabel: "Compte",
        staffSuffix: " (équipe)",
        logoutToast: "Session terminée.",
        welcomeBack: "Bon retour !",
        accountCreated: "Compte créé — 20 transcriptions par jour !",
        adminSession: "Session administrateur active.",
        loginFail: "Connexion impossible.",
        registerFail: "Inscription impossible.",
        invalidCreds: "Identifiants invalides.",
        loginError: "Erreur de connexion.",
        registerError: "Erreur d'inscription.",
      },
      de: {
        close: "Schließen",
        titleLogin: "Anmelden",
        titleRegister: "Konto erstellen",
        titleAdmin: "Admin-Anmeldung",
        tabLogin: "Anmelden",
        tabRegister: "Registrieren",
        tabAdmin: "Admin",
        email: "E-Mail",
        password: "Passwort",
        passwordMin: "Passwort (min. 8)",
        nameOptional: "Name (optional)",
        username: "Benutzer",
        loginBtn: "Anmelden",
        registerBtn: "Konto erstellen",
        adminBtn: "Als Admin anmelden",
        registerHint: "Mit einem Konto kannst du die Seite nutzen. Normale Aktivität benachrichtigt den Administrator.",
        adminHint: "Team-Konto — Aktivität auf der Seite sendet keine Benachrichtigungs-E-Mails.",
        accountLabel: "Konto",
        staffSuffix: " (Team)",
        logoutToast: "Abgemeldet.",
        welcomeBack: "Willkommen zurück!",
        accountCreated: "Konto erstellt — 20 Transkriptionen pro Tag!",
        adminSession: "Admin-Sitzung aktiv.",
        loginFail: "Anmeldung fehlgeschlagen.",
        registerFail: "Registrierung fehlgeschlagen.",
        invalidCreds: "Ungültige Anmeldedaten.",
        loginError: "Fehler bei der Anmeldung.",
        registerError: "Fehler bei der Registrierung.",
      },
    };
    return L[locale] || L.en;
  }

  function upsellStrings(locale) {
    var L = {
      pt: {
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
      },
      en: {
        close: "Close",
        titleDefault: "Want more?",
        proCta: "View Pro plan",
        registerCta: "Create free account",
        dismiss: "Not now",
        limitTitle: "Daily limit reached",
        freeLimitTitle: "Free account limit",
        proLimitTitle: "Pro daily limit",
        anonLimitHidden: "You've used today's free transcriptions. Sign up for more or try again tomorrow.",
        anonLimit: "You've used today's free transcriptions. Sign up for more or upgrade to Pro for DOCX export and higher limits.",
        regLimitHidden: "You've reached your daily account limit. Try again tomorrow.",
        regLimit: "You've reached the free account daily limit. Pro includes more transcriptions, DOCX export and extended history.",
        proLimit: "Try again tomorrow.",
        savedHistory: "Transcription saved to history.",
        saved: "Transcription saved",
        savedInHistory: " to history",
        proPitch: "Pro ({price}): DOCX export, more transcriptions/day. ",
        proSoon: "Coming soon: Pro plan with DOCX and more transcriptions. ",
        viewPlans: "View plans",
        createAccount: "Create account",
      },
      es: {
        close: "Cerrar",
        titleDefault: "¿Quieres más?",
        proCta: "Ver plan Pro",
        registerCta: "Crear cuenta gratis",
        dismiss: "Ahora no",
        limitTitle: "Límite diario alcanzado",
        freeLimitTitle: "Límite de cuenta gratis",
        proLimitTitle: "Límite Pro de hoy",
        anonLimitHidden: "Has usado las transcripciones gratis de hoy. Regístrate para más o inténtalo mañana.",
        anonLimit: "Has usado las transcripciones gratis de hoy. Regístrate para más o pasa a Pro para exportar DOCX y límites mayores.",
        regLimitHidden: "Has alcanzado el límite diario de tu cuenta. Inténtalo mañana.",
        regLimit: "Has alcanzado el límite diario de la cuenta gratis. Pro incluye más transcripciones, exportación DOCX e historial ampliado.",
        proLimit: "Inténtalo mañana.",
        savedHistory: "Transcripción guardada en el historial.",
        saved: "Transcripción guardada",
        savedInHistory: " en el historial",
        proPitch: "Pro ({price}): exportación DOCX, más transcripciones/día. ",
        proSoon: "Pronto: plan Pro con DOCX y más transcripciones. ",
        viewPlans: "Ver planes",
        createAccount: "Crear cuenta",
      },
      fr: {
        close: "Fermer",
        titleDefault: "Envie de plus ?",
        proCta: "Voir l'offre Pro",
        registerCta: "Créer un compte gratuit",
        dismiss: "Pas maintenant",
        limitTitle: "Limite quotidienne atteinte",
        freeLimitTitle: "Limite du compte gratuit",
        proLimitTitle: "Limite Pro du jour",
        anonLimitHidden: "Vous avez utilisé les transcriptions gratuites d'aujourd'hui. Inscrivez-vous pour plus ou réessayez demain.",
        anonLimit: "Vous avez utilisé les transcriptions gratuites d'aujourd'hui. Inscrivez-vous pour plus ou passez à Pro pour l'export DOCX et des limites plus élevées.",
        regLimitHidden: "Vous avez atteint la limite quotidienne de votre compte. Réessayez demain.",
        regLimit: "Vous avez atteint la limite quotidienne du compte gratuit. Pro inclut plus de transcriptions, l'export DOCX et un historique étendu.",
        proLimit: "Réessayez demain.",
        savedHistory: "Transcription enregistrée dans l'historique.",
        saved: "Transcription enregistrée",
        savedInHistory: " dans l'historique",
        proPitch: "Pro ({price}) : export DOCX, plus de transcriptions/jour. ",
        proSoon: "Bientôt : offre Pro avec DOCX et plus de transcriptions. ",
        viewPlans: "Voir les offres",
        createAccount: "Créer un compte",
      },
      de: {
        close: "Schließen",
        titleDefault: "Mehr gewünscht?",
        proCta: "Pro-Plan ansehen",
        registerCta: "Kostenloses Konto erstellen",
        dismiss: "Nicht jetzt",
        limitTitle: "Tageslimit erreicht",
        freeLimitTitle: "Limit des Gratis-Kontos",
        proLimitTitle: "Pro-Tageslimit",
        anonLimitHidden: "Du hast die kostenlosen Transkriptionen für heute genutzt. Registriere dich für mehr oder versuche es morgen erneut.",
        anonLimit: "Du hast die kostenlosen Transkriptionen für heute genutzt. Registriere dich für mehr oder wechsle zu Pro für DOCX-Export und höhere Limits.",
        regLimitHidden: "Du hast das Tageslimit deines Kontos erreicht. Versuche es morgen erneut.",
        regLimit: "Du hast das Tageslimit des Gratis-Kontos erreicht. Pro bietet mehr Transkriptionen, DOCX-Export und erweiterten Verlauf.",
        proLimit: "Versuche es morgen erneut.",
        savedHistory: "Transkription im Verlauf gespeichert.",
        saved: "Transkription gespeichert",
        savedInHistory: " im Verlauf",
        proPitch: "Pro ({price}): DOCX-Export, mehr Transkriptionen/Tag. ",
        proSoon: "Demnächst: Pro-Plan mit DOCX und mehr Transkriptionen. ",
        viewPlans: "Pläne ansehen",
        createAccount: "Konto erstellen",
      },
    };
    return L[locale] || L.en;
  }

  function langMenuHtml(current) {
    return LOCALES.map(function (loc) {
      var meta = LOCALE_META[loc];
      return (
        '<button type="button" data-lang="' + loc + '" role="menuitem">' +
        '<img src="' + meta.flag + '" alt="" width="22" height="16"> ' +
        meta.label +
        "</button>"
      );
    }).join("\n");
  }

  global.OuviescreviI18n = {
    LOCALES: LOCALES,
    LOCALE_META: LOCALE_META,
    PAGES: PAGES,
    localeFromPath: localeFromPath,
    localePrefix: localePrefix,
    pageSlugFromPath: pageSlugFromPath,
    pathFor: pathFor,
    currentPagePath: currentPagePath,
    hreflangMapForPath: hreflangMapForPath,
    switchLanguage: switchLanguage,
    uiStrings: uiStrings,
    authStrings: authStrings,
    upsellStrings: upsellStrings,
    langMenuHtml: langMenuHtml,
  };
})(window);
