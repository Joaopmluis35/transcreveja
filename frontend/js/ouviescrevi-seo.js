/**
 * SEO — canonical, Open Graph, Twitter, hreflang e JSON-LD.
 */
(function (global) {
  var SITE = "https://www.ouviescrevi.pt";
  var OG_IMAGE = SITE + "/logos/ouviescrevi-logo-pro.png";
  var ORG_ID = SITE + "/#organization";

  var HREFLANG_SLUGS = {
    index: { pt: "index.html", en: "index.html", es: "index.html", fr: "index.html", de: "index.html" },
    ajuda: { pt: "ajuda.html", en: "ajuda.html", es: "ajuda.html", fr: "ajuda.html", de: "ajuda.html" },
    conversor: { pt: "conversor.html", en: "conversor.html", es: "conversor.html", fr: "conversor.html", de: "conversor.html" },
    "conversor-imagens": { pt: "conversor-imagens.html", en: "conversor-imagens.html" },
    sugestoes: { pt: "sugestoes.html", en: "sugestoes.html", es: "sugestoes.html", fr: "sugestoes.html", de: "sugestoes.html" },
    resumo: { pt: "resumo.html", en: "resumo.html", es: "resumo.html", fr: "resumo.html", de: "resumo.html" },
    "url-resumo": { pt: "url-resumo.html", en: "url-resumo.html", es: "url-resumo.html", fr: "url-resumo.html", de: "url-resumo.html" },
    perguntas: { pt: "perguntas.html", en: "perguntas.html", es: "perguntas.html", fr: "perguntas.html", de: "perguntas.html" },
    corretor: { pt: "corretor.html", en: "corretor.html", es: "corretor.html", fr: "corretor.html", de: "corretor.html" },
    "aula-pronta": { pt: "aula-pronta.html", en: "aula-pronta.html" },
    flashcards: { pt: "flashcards.html", en: "flashcards.html", es: "flashcards.html", fr: "flashcards.html", de: "flashcards.html" },
    capitulos: { pt: "capitulos.html", en: "capitulos.html", es: "capitulos.html", fr: "capitulos.html", de: "capitulos.html" },
    "aula-completa": { pt: "aula-completa.html", en: "aula-completa.html", es: "aula-completa.html", fr: "aula-completa.html", de: "aula-completa.html" },
    "descricao-youtube": { pt: "descricao-youtube.html", en: "descricao-youtube.html", es: "descricao-youtube.html", fr: "descricao-youtube.html", de: "descricao-youtube.html" },
    "podcast-youtube": { pt: "podcast-youtube.html", en: "podcast-youtube.html", es: "podcast-youtube.html", fr: "podcast-youtube.html", de: "podcast-youtube.html" },
    cookies: { pt: "cookies.html", en: "cookies.html", es: "cookies.html", fr: "cookies.html", de: "cookies.html" },
    privacy: { pt: "privacidade.html", en: "privacy.html", es: "privacy.html", fr: "privacy.html", de: "privacy.html" },
    terms: { pt: "termos.html", en: "terms.html", es: "terms.html", fr: "terms.html", de: "terms.html" },
  };

  function hrefPath(locale, slug) {
    var file = HREFLANG_SLUGS[slug][locale];
    if (!file) return null;
    return (locale === "pt" ? "" : "/" + locale) + "/" + file;
  }

  function slugFromPath(path) {
    path = (path || "").replace(/\/$/, "");
    var file = path.split("/").pop() || "index.html";
    if (file === "index.html") return "index";
    var name = file.replace(".html", "");
    if (HREFLANG_SLUGS[name]) return name;
    if (name === "privacidade" || name === "privacy") return "privacy";
    if (name === "termos" || name === "terms") return "terms";
    return null;
  }

  function buildHreflangMap() {
    var map = {};
    Object.keys(HREFLANG_SLUGS).forEach(function (slug) {
      var group = {};
      Object.keys(HREFLANG_SLUGS[slug]).forEach(function (loc) {
        group[loc] = hrefPath(loc, slug);
      });
      Object.keys(group).forEach(function (loc) {
        map[group[loc]] = group;
      });
    });
    return map;
  }

  var HREFLANG = buildHreflangMap();

  var PAGES = {
    "/index.html": {
      title: "Ouviescrevi — Transcrição de Áudio e Vídeo com IA Grátis",
      description:
        "Transcreve áudio e vídeo online com IA, grátis e sem registo. Resumos, tradução, legendas SRT, perguntas de estudo e conversão de ficheiros. Feito em Portugal.",
      type: "home",
    },
    "/en/index.html": {
      title: "Ouviescrevi — Free AI Audio & Video Transcription",
      description:
        "Transcribe audio and video online with AI for free. Summaries, translation, SRT subtitles, study questions and file conversion. No sign-up required.",
      type: "home",
      lang: "en",
    },
    "/es/index.html": {
      title: "Ouviescrevi — Transcripción de audio y vídeo con IA gratis",
      description:
        "Transcribe audio y vídeo online con IA, gratis y sin registro. Resúmenes, traducción, subtítulos SRT, preguntas de estudio y conversión de archivos.",
      type: "home",
      lang: "es",
    },
    "/fr/index.html": {
      title: "Ouviescrevi — Transcription audio et vidéo IA gratuite",
      description:
        "Transcrivez audio et vidéo en ligne avec l'IA, gratuitement et sans inscription. Résumés, traduction, sous-titres SRT, questions d'étude et conversion de fichiers.",
      type: "home",
      lang: "fr",
    },
    "/de/index.html": {
      title: "Ouviescrevi — Kostenlose KI-Audio- und Video-Transkription",
      description:
        "Transkribiere Audio und Video online mit KI, kostenlos und ohne Anmeldung. Zusammenfassungen, Übersetzung, SRT-Untertitel, Lernfragen und Dateikonvertierung.",
      type: "home",
      lang: "de",
    },

    "/conversor.html": {
      title: "Conversor de Ficheiros Online Grátis — Word, PDF, Imagem | Ouviescrevi",
      description:
        "Converte Word para PDF, PDF para texto e imagens para PDF no browser. Rápido, gratuito e sem instalação.",
    },
    "/en/conversor.html": {
      title: "Free Online File Converter — Word, PDF, Image | Ouviescrevi",
      description:
        "Convert Word to PDF, PDF to text and images to PDF in your browser. Free, fast and no installation.",
      lang: "en",
    },
    "/es/conversor.html": {
      title: "Conversor de archivos online gratis — Word, PDF e imagen | Ouviescrevi",
      description:
        "Convierte Word a PDF, PDF a texto e imágenes a PDF en el navegador. Rápido, gratis y sin instalación.",
      lang: "es",
    },
    "/fr/conversor.html": {
      title: "Convertisseur de fichiers en ligne gratuit — Word, PDF, Image | Ouviescrevi",
      description:
        "Convertissez Word en PDF, PDF en texte et images en PDF dans le navigateur. Rapide, gratuit et sans installation.",
      lang: "fr",
    },
    "/de/conversor.html": {
      title: "Kostenloser Online-Dateikonverter — Word, PDF, Bild | Ouviescrevi",
      description:
        "Word in PDF, PDF in Text und Bilder in PDF im Browser konvertieren. Schnell, kostenlos und ohne Installation.",
      lang: "de",
    },

    "/conversor-imagens.html": {
      title: "Conversor de Imagens — Converter, Comprimir e Redimensionar | Ouviescrevi",
      description:
        "Converte, comprime e redimensiona imagens no browser. PNG, JPEG, WebP, AVIF, BMP e GIF — grátis e privado.",
    },
    "/en/conversor-imagens.html": {
      title: "Image Tool — Convert, Compress & Resize | Ouviescrevi",
      description:
        "Convert, compress and resize images in your browser. PNG, JPEG, WebP, AVIF, BMP and GIF — free and private.",
      lang: "en",
    },

    "/resumo.html": {
      title: "Resumo Automático com IA — PDF, Word e Texto | Ouviescrevi",
      description:
        "Gera resumos inteligentes com IA a partir de PDF, Word ou texto. Estilos formal, simples, tópicos ou minuta de reunião.",
    },
    "/en/resumo.html": {
      title: "AI Summary Generator — PDF, Word & Text | Ouviescrevi",
      description:
        "Generate smart AI summaries from PDF, Word or plain text. Formal, simple, bullet points or meeting minutes.",
      lang: "en",
    },
    "/es/resumo.html": {
      title: "Generador de resúmenes con IA — PDF, Word y texto | Ouviescrevi",
      description:
        "Genera resúmenes inteligentes con IA a partir de PDF, Word o texto. Estilos formal, simple, puntos clave o acta de reunión.",
      lang: "es",
    },
    "/fr/resumo.html": {
      title: "Résumé automatique par IA — PDF, Word et texte | Ouviescrevi",
      description:
        "Générez des résumés intelligents par IA à partir de PDF, Word ou texte. Styles formel, simple, points clés ou compte-rendu.",
      lang: "fr",
    },
    "/de/resumo.html": {
      title: "KI-Zusammenfassung — PDF, Word und Text | Ouviescrevi",
      description:
        "Erstelle intelligente KI-Zusammenfassungen aus PDF, Word oder Text. Formell, einfach, Stichpunkte oder Sitzungsprotokoll.",
      lang: "de",
    },

    "/ajuda.html": {
      title: "Ajuda e FAQ — Como Usar o Ouviescrevi",
      description:
        "Respostas às perguntas frequentes sobre transcrição com IA, formatos suportados, privacidade, limites gratuitos e ferramentas do Ouviescrevi.",
      faq: true,
    },
    "/en/ajuda.html": {
      title: "Help & FAQ — How to Use Ouviescrevi",
      description:
        "Frequently asked questions about AI transcription, supported formats, privacy, free limits and Ouviescrevi features.",
      lang: "en",
      faq: true,
    },
    "/es/ajuda.html": {
      title: "Ayuda y FAQ — Cómo usar Ouviescrevi",
      description:
        "Preguntas frecuentes sobre transcripción con IA, formatos compatibles, privacidad, límites gratuitos y herramientas de Ouviescrevi.",
      lang: "es",
      faq: true,
    },
    "/fr/ajuda.html": {
      title: "Aide et FAQ — Comment utiliser Ouviescrevi",
      description:
        "Questions fréquentes sur la transcription IA, formats pris en charge, confidentialité, limites gratuites et outils Ouviescrevi.",
      lang: "fr",
      faq: true,
    },
    "/de/ajuda.html": {
      title: "Hilfe und FAQ — So nutzt du Ouviescrevi",
      description:
        "Häufige Fragen zu KI-Transkription, unterstützten Formaten, Datenschutz, kostenlosen Limits und Ouviescrevi-Funktionen.",
      lang: "de",
      faq: true,
    },

    "/corretor.html": {
      title: "Corretor de Texto com IA — Ortografia e Gramática | Ouviescrevi",
      description:
        "Corrige ortografia, gramática e estilo automaticamente com inteligência artificial. Grátis e direto no browser.",
    },
    "/en/corretor.html": {
      title: "AI Text Proofreader — Spelling and Grammar | Ouviescrevi",
      description:
        "Fix spelling, grammar and style automatically with AI. Free proofreading tool in your browser.",
      lang: "en",
    },
    "/es/corretor.html": {
      title: "Corrector de texto con IA — Ortografía y gramática | Ouviescrevi",
      description:
        "Corrige ortografía, gramática y estilo automáticamente con IA. Herramienta gratuita en el navegador.",
      lang: "es",
    },
    "/fr/corretor.html": {
      title: "Correcteur de texte IA — Orthographe et grammaire | Ouviescrevi",
      description:
        "Corrigez orthographe, grammaire et style automatiquement avec l'IA. Outil gratuit dans le navigateur.",
      lang: "fr",
    },
    "/de/corretor.html": {
      title: "KI-Textkorrektur — Rechtschreibung und Grammatik | Ouviescrevi",
      description:
        "Rechtschreibung, Grammatik und Stil automatisch mit KI korrigieren. Kostenloses Tool im Browser.",
      lang: "de",
    },

    "/perguntas.html": {
      title: "Gerador de Perguntas de Escolha Múltipla com IA | Ouviescrevi",
      description:
        "Cria perguntas de estudo e testes a partir de qualquer texto com IA. Gabarito e explicações incluídos — ideal para professores e alunos.",
    },
    "/en/perguntas.html": {
      title: "AI Multiple-Choice Question Generator | Ouviescrevi",
      description:
        "Generate study questions and quizzes from any text with AI. Answer keys and explanations included — for teachers and students.",
      lang: "en",
    },
    "/es/perguntas.html": {
      title: "Generador de preguntas de opción múltiple con IA | Ouviescrevi",
      description:
        "Crea preguntas de estudio y exámenes a partir de cualquier texto con IA. Incluye respuestas y explicaciones.",
      lang: "es",
    },
    "/fr/perguntas.html": {
      title: "Générateur de questions à choix multiples IA | Ouviescrevi",
      description:
        "Créez des questions d'étude et de test à partir de n'importe quel texte avec l'IA. Corrigé et explications inclus.",
      lang: "fr",
    },
    "/de/perguntas.html": {
      title: "KI-Fragengenerator — Multiple-Choice | Ouviescrevi",
      description:
        "Erstelle Lern- und Testfragen aus beliebigem Text mit KI. Mit Lösungen und Erklärungen — für Lehrkräfte und Schüler.",
      lang: "de",
    },

    "/aula-pronta.html": {
      title: "Aula Pronta — Pacote de Estudo com IA | Ouviescrevi",
      description:
        "Transforma uma transcrição de aula em pacote completo: resumo, glossário, pontos-chave e perguntas com gabarito.",
    },
    "/en/aula-pronta.html": {
      title: "Lesson Ready — AI Study Pack | Ouviescrevi",
      description:
        "Turn a lesson transcript into a complete study pack: summary, glossary, key points and quiz questions with answer key.",
      lang: "en",
    },

    "/capitulos.html": {
      title: "Capítulos e Timestamps — Podcasts e YouTube | Ouviescrevi",
      description:
        "Divide transcrições longas em capítulos com horários. Ideal para podcasts, aulas gravadas e vídeos — exporta para YouTube.",
    },
    "/en/capitulos.html": {
      title: "Chapters & Timestamps — Podcasts & YouTube | Ouviescrevi",
      description:
        "Split long transcripts into chapters with timestamps. Great for podcasts, lessons and videos — export for YouTube.",
      lang: "en",
    },

    "/flashcards.html": {
      title: "Flashcards com IA — Estudo e Memorização | Ouviescrevi",
      description:
        "Gera flashcards de estudo com IA a partir de texto ou transcrição. Frente e verso — ideal para revisão escolar.",
    },
    "/en/flashcards.html": {
      title: "AI Flashcards — Study & Memorization | Ouviescrevi",
      description:
        "Generate AI study flashcards from any text or transcript. Front and back cards for revision and memorization.",
      lang: "en",
    },
    "/descricao-youtube.html": {
      title: "Descrição para YouTube com IA — Títulos e Tags | Ouviescrevi",
      description:
        "Gera título, descrição e tags para YouTube com IA a partir de transcrições e capítulos. Ideal para podcasts e vídeos.",
    },
    "/en/descricao-youtube.html": {
      title: "YouTube Description Generator — Titles & Tags | Ouviescrevi",
      description:
        "Generate YouTube title, description and tags with AI from transcripts and chapters. Great for podcasts and videos.",
      lang: "en",
    },

    "/es/flashcards.html": {
      title: "Flashcards con IA — Estudio y Memorización | Ouviescrevi",
      description:
        "Genera flashcards de estudio con IA a partir de texto o transcripción. Anverso y reverso — ideal para repasar.",
      lang: "es",
    },
    "/fr/flashcards.html": {
      title: "Flashcards IA — Étude et Mémorisation | Ouviescrevi",
      description:
        "Générez des flashcards d'étude avec l'IA à partir de texte ou transcription. Recto et verso pour réviser.",
      lang: "fr",
    },
    "/de/flashcards.html": {
      title: "KI-Karteikarten — Lernen & Merken | Ouviescrevi",
      description:
        "Erstellen Sie Lernkarteikarten mit KI aus Text oder Transkript. Vorder- und Rückseite zum Wiederholen.",
      lang: "de",
    },
    "/es/descricao-youtube.html": {
      title: "Descripción para YouTube con IA — Títulos y Etiquetas | Ouviescrevi",
      description:
        "Genera título, descripción y etiquetas para YouTube con IA a partir de transcripciones y capítulos.",
      lang: "es",
    },
    "/fr/descricao-youtube.html": {
      title: "Description YouTube IA — Titres et Tags | Ouviescrevi",
      description:
        "Générez titre, description et tags YouTube avec l'IA à partir de transcriptions et chapitres.",
      lang: "fr",
    },
    "/de/descricao-youtube.html": {
      title: "YouTube-Beschreibung mit KI — Titel & Tags | Ouviescrevi",
      description:
        "Erstellen Sie YouTube-Titel, Beschreibung und Tags mit KI aus Transkripten und Kapiteln.",
      lang: "de",
    },
    "/podcast-youtube.html": {
      title: "Assistente Podcast & YouTube — Capítulos e Descrição | Ouviescrevi",
      description:
        "Da transcrição aos capítulos e à descrição do YouTube — assistente guiado com IA para podcasts e vídeos.",
    },
    "/en/podcast-youtube.html": {
      title: "Podcast & YouTube Assistant — Chapters & Description | Ouviescrevi",
      description:
        "From transcript to chapters and YouTube description — guided AI workflow for podcasts and videos.",
      lang: "en",
    },
    "/es/podcast-youtube.html": {
      title: "Asistente Podcast y YouTube — Capítulos y Descripción | Ouviescrevi",
      description:
        "De la transcripción a capítulos y descripción de YouTube — flujo guiado con IA para podcasts y vídeos.",
      lang: "es",
    },
    "/fr/podcast-youtube.html": {
      title: "Assistant Podcast & YouTube — Chapitres et Description | Ouviescrevi",
      description:
        "De la transcription aux chapitres et à la description YouTube — parcours guidé avec l'IA.",
      lang: "fr",
    },
    "/de/podcast-youtube.html": {
      title: "Podcast- & YouTube-Assistent — Kapitel & Beschreibung | Ouviescrevi",
      description:
        "Vom Transkript zu Kapiteln und YouTube-Beschreibung — geführter KI-Workflow für Podcasts und Videos.",
      lang: "de",
    },

    "/aula-completa.html": {
      title: "Assistente Aula Completa — Resumo, Perguntas e Flashcards | Ouviescrevi",
      description:
        "Da transcrição ao resumo, perguntas e flashcards — assistente guiado com IA para professores e estudantes.",
    },
    "/en/aula-completa.html": {
      title: "Full Lesson Assistant — Summary, Questions & Flashcards | Ouviescrevi",
      description:
        "From transcript to summary, questions and flashcards — guided AI workflow for teachers and students.",
      lang: "en",
    },
    "/es/aula-completa.html": {
      title: "Asistente Clase Completa — Resumen, Preguntas y Flashcards | Ouviescrevi",
      description:
        "De la transcripción al resumen, preguntas y flashcards — flujo guiado con IA.",
      lang: "es",
    },
    "/fr/aula-completa.html": {
      title: "Assistant Cours Complet — Résumé, Questions et Flashcards | Ouviescrevi",
      description:
        "De la transcription au résumé, questions et flashcards — parcours guidé avec l'IA.",
      lang: "fr",
    },
    "/de/aula-completa.html": {
      title: "Unterrichts-Assistent — Zusammenfassung, Fragen & Karteikarten | Ouviescrevi",
      description:
        "Vom Transkript zur Zusammenfassung, Fragen und Karteikarten — geführter KI-Ablauf.",
      lang: "de",
    },
    "/es/capitulos.html": {
      title: "Capítulos y marcas de tiempo — Podcasts y YouTube | Ouviescrevi",
      description:
        "Divide transcripciones largas en capítulos con marcas de tiempo. Ideal para podcasts, clases y vídeos.",
      lang: "es",
    },
    "/fr/capitulos.html": {
      title: "Chapitres et horodatages — Podcasts et YouTube | Ouviescrevi",
      description:
        "Divisez de longues transcriptions en chapitres avec horodatages. Idéal pour podcasts et vidéos.",
      lang: "fr",
    },
    "/de/capitulos.html": {
      title: "Kapitel & Zeitstempel — Podcasts & YouTube | Ouviescrevi",
      description:
        "Teilen Sie lange Transkripte in Kapitel mit Zeitstempeln. Ideal für Podcasts und Videos.",
      lang: "de",
    },

    "/url-resumo.html": {
      title: "Resumo de Artigo por URL com IA | Ouviescrevi",
      description:
        "Cola o link de um artigo online e obtém um resumo automático com inteligência artificial em segundos.",
    },
    "/en/url-resumo.html": {
      title: "AI Article Summary from URL | Ouviescrevi",
      description:
        "Paste an article link and get an automatic AI-generated summary in seconds. No copy-paste needed.",
      lang: "en",
    },
    "/es/url-resumo.html": {
      title: "Resumen de artículo por URL con IA | Ouviescrevi",
      description:
        "Pega el enlace de un artículo online y obtén un resumen automático con inteligencia artificial en segundos.",
      lang: "es",
    },
    "/fr/url-resumo.html": {
      title: "Résumé d'article par URL avec IA | Ouviescrevi",
      description:
        "Collez le lien d'un article en ligne et obtenez un résumé automatique par intelligence artificielle en quelques secondes.",
      lang: "fr",
    },
    "/de/url-resumo.html": {
      title: "KI-Artikelzusammenfassung per URL | Ouviescrevi",
      description:
        "Füge einen Artikellink ein und erhalte in Sekunden eine automatische KI-Zusammenfassung — ohne Copy-Paste.",
      lang: "de",
    },

    "/aulas.html": {
      title: "Transcrição de Aulas com IA — Estudantes e Professores | Ouviescrevi",
      description:
        "Transforma gravações de aulas em texto editável com IA. Ideal para estudantes, professores e ensino à distância.",
    },
    "/professores.html": {
      title: "IA para Professores — Transcrever e Resumir Aulas | Ouviescrevi",
      description:
        "Ferramentas de IA para educadores: transcrição de aulas, resumos automáticos, perguntas de estudo e exportação DOCX.",
    },
    "/jornalistas.html": {
      title: "Transcrição de Entrevistas para Jornalistas | Ouviescrevi",
      description:
        "Transcreve entrevistas de áudio e vídeo com IA. Poupa horas de trabalho e exporta texto pronto a editar.",
    },
    "/podcasts.html": {
      title: "Transcrever Podcast com IA — Episódios em Texto | Ouviescrevi",
      description:
        "Converte episódios de podcast em transcrições e resumos com inteligência artificial. Grátis e online.",
    },
    "/reunioes.html": {
      title: "Transcrição de Reuniões e Minutas com IA | Ouviescrevi",
      description:
        "Grava e transcreve reuniões automaticamente. Gera minutas e resumos para equipas e empresas.",
    },
    "/testemunhos.html": {
      title: "Transcrição de Testemunhos e Declarações | Ouviescrevi",
      description:
        "Transcreve testemunhos, declarações e gravações com precisão. Útil em contextos jurídicos e administrativos.",
    },
    "/gerar-video.html": {
      title: "Gerar Vídeo com Voz e Legendas — IA | Ouviescrevi",
      description:
        "Cria vídeos com narração automática e legendas a partir de texto com inteligência artificial.",
    },
    "/precos.html": {
      title: "Preços e Planos — Transcrição com IA | Ouviescrevi",
      description:
        "Planos Ouviescrevi para transcrição com IA, legendas SRT e exportação. Opções gratuitas e premium para criadores e educadores.",
    },

    "/sugestoes.html": {
      title: "Enviar Sugestões — Ouviescrevi",
      description: "Partilha ideias para melhorar o Ouviescrevi. O teu feedback ajuda-nos a evoluir a plataforma.",
    },
    "/en/sugestoes.html": {
      title: "Send Suggestions — Ouviescrevi",
      description: "Share ideas to improve Ouviescrevi. Your feedback helps us build better AI tools.",
      lang: "en",
    },
    "/es/sugestoes.html": {
      title: "Enviar sugerencias — Ouviescrevi",
      description: "Comparte ideas para mejorar Ouviescrevi. Tu opinión nos ayuda a crear mejores herramientas de IA.",
      lang: "es",
    },
    "/fr/sugestoes.html": {
      title: "Envoyer des suggestions — Ouviescrevi",
      description: "Partagez vos idées pour améliorer Ouviescrevi. Vos retours nous aident à créer de meilleurs outils IA.",
      lang: "fr",
    },
    "/de/sugestoes.html": {
      title: "Vorschläge senden — Ouviescrevi",
      description: "Teile Ideen zur Verbesserung von Ouviescrevi. Dein Feedback hilft uns, bessere KI-Tools zu entwickeln.",
      lang: "de",
    },

    "/privacidade.html": {
      title: "Política de Privacidade — Ouviescrevi (RGPD)",
      description:
        "Como o Ouviescrevi trata dados pessoais em conformidade com o RGPD. Ficheiros, cookies e os teus direitos.",
    },
    "/en/privacy.html": {
      title: "Privacy Policy — Ouviescrevi (GDPR)",
      description: "How Ouviescrevi processes personal data under GDPR. Files, cookies and your rights.",
      lang: "en",
    },
    "/es/privacy.html": {
      title: "Política de privacidad — Ouviescrevi (RGPD)",
      description: "Cómo Ouviescrevi trata los datos personales conforme al RGPD. Archivos, cookies y tus derechos.",
      lang: "es",
    },
    "/fr/privacy.html": {
      title: "Politique de confidentialité — Ouviescrevi (RGPD)",
      description: "Comment Ouviescrevi traite les données personnelles conformément au RGPD. Fichiers, cookies et vos droits.",
      lang: "fr",
    },
    "/de/privacy.html": {
      title: "Datenschutzerklärung — Ouviescrevi (DSGVO)",
      description: "Wie Ouviescrevi personenbezogene Daten gemäß DSGVO verarbeitet. Dateien, Cookies und deine Rechte.",
      lang: "de",
    },

    "/termos.html": {
      title: "Termos de Utilização — Ouviescrevi",
      description: "Condições de uso do serviço de transcrição, resumos e ferramentas de IA do Ouviescrevi.",
    },
    "/en/terms.html": {
      title: "Terms of Use — Ouviescrevi",
      description: "Terms and conditions for using Ouviescrevi AI transcription, summaries and tools.",
      lang: "en",
    },
    "/es/terms.html": {
      title: "Términos de uso — Ouviescrevi",
      description: "Condiciones de uso del servicio de transcripción, resúmenes y herramientas de IA de Ouviescrevi.",
      lang: "es",
    },
    "/fr/terms.html": {
      title: "Conditions d'utilisation — Ouviescrevi",
      description: "Conditions d'utilisation du service de transcription, résumés et outils IA d'Ouviescrevi.",
      lang: "fr",
    },
    "/de/terms.html": {
      title: "Nutzungsbedingungen — Ouviescrevi",
      description: "Nutzungsbedingungen für Ouviescrevi KI-Transkription, Zusammenfassungen und Tools.",
      lang: "de",
    },

    "/cookies.html": {
      title: "Política de Cookies — Ouviescrevi",
      description: "Informação sobre cookies e armazenamento local utilizados no website Ouviescrevi.",
    },
    "/en/cookies.html": {
      title: "Cookie Policy — Ouviescrevi",
      description: "Information about cookies and local storage used on the Ouviescrevi website.",
      lang: "en",
    },
    "/es/cookies.html": {
      title: "Política de cookies — Ouviescrevi",
      description: "Información sobre cookies y almacenamiento local utilizados en el sitio web Ouviescrevi.",
      lang: "es",
    },
    "/fr/cookies.html": {
      title: "Politique de cookies — Ouviescrevi",
      description: "Informations sur les cookies et le stockage local utilisés sur le site Ouviescrevi.",
      lang: "fr",
    },
    "/de/cookies.html": {
      title: "Cookie-Richtlinie — Ouviescrevi",
      description: "Informationen zu Cookies und lokalem Speicher auf der Ouviescrevi-Website.",
      lang: "de",
    },

    "/404.html": {
      title: "Página não encontrada — Ouviescrevi",
      description: "A página que procuras não existe ou foi movida. Volta ao início do Ouviescrevi.",
      noindex: true,
    },
  };

  var FAQ_PT = [
    { q: "O que é o Ouviescrevi?", a: "É uma ferramenta automática que converte ficheiros de áudio, vídeo ou texto em transcrições, resumos, traduções e muito mais, usando inteligência artificial." },
    { q: "Que formatos de ficheiros são suportados?", a: "Suportamos .mp3, .mp4, .wav, .m4a, .mov e outros formatos comuns." },
    { q: "É necessário criar conta?", a: "Não. O Ouviescrevi está disponível gratuitamente e sem necessidade de registo." },
    { q: "A transcrição é 100% precisa?", a: "A precisão depende da qualidade do áudio. Utilizamos o modelo Whisper da OpenAI para alta qualidade." },
    { q: "Os meus ficheiros são guardados?", a: "Os ficheiros são eliminados após o processamento. Registamos apenas o nome e a data para estatísticas." },
  ];

  var FAQ_EN = [
    { q: "What is Ouviescrevi?", a: "A fully automated tool that converts audio, video, or text files into transcriptions, summaries, translations, and more using AI." },
    { q: "Which file formats are supported?", a: "We support .mp3, .mp4, .wav, .m4a, .mov and other common formats." },
    { q: "Do I need to create an account?", a: "No. Ouviescrevi is completely free and requires no registration." },
    { q: "Is the transcription 100% accurate?", a: "Accuracy depends on audio quality. We use OpenAI's Whisper model for high-quality results." },
    { q: "Are my files stored?", a: "Files are deleted after processing. Only file name and timestamp are stored for statistics." },
  ];

  var FAQ_ES = [
    { q: "¿Qué es Ouviescrevi?", a: "Es una herramienta automática que convierte archivos de audio, vídeo o texto en transcripciones, resúmenes, traducciones y más, usando inteligencia artificial." },
    { q: "¿Qué formatos de archivo se admiten?", a: "Admitimos .mp3, .mp4, .wav, .m4a, .mov y otros formatos habituales." },
    { q: "¿Es necesario crear una cuenta?", a: "No. Ouviescrevi es gratuito y no requiere registro." },
    { q: "¿La transcripción es 100% precisa?", a: "La precisión depende de la calidad del audio. Usamos el modelo Whisper de OpenAI para alta calidad." },
    { q: "¿Se guardan mis archivos?", a: "Los archivos se eliminan tras el procesamiento. Solo registramos el nombre y la fecha para estadísticas." },
  ];

  var FAQ_FR = [
    { q: "Qu'est-ce qu'Ouviescrevi ?", a: "Un outil automatique qui convertit des fichiers audio, vidéo ou texte en transcriptions, résumés, traductions et plus encore grâce à l'IA." },
    { q: "Quels formats de fichiers sont pris en charge ?", a: "Nous prenons en charge .mp3, .mp4, .wav, .m4a, .mov et d'autres formats courants." },
    { q: "Faut-il créer un compte ?", a: "Non. Ouviescrevi est gratuit et ne nécessite aucune inscription." },
    { q: "La transcription est-elle 100 % précise ?", a: "La précision dépend de la qualité audio. Nous utilisons le modèle Whisper d'OpenAI pour une haute qualité." },
    { q: "Mes fichiers sont-ils conservés ?", a: "Les fichiers sont supprimés après traitement. Seuls le nom et la date sont enregistrés pour les statistiques." },
  ];

  var FAQ_DE = [
    { q: "Was ist Ouviescrevi?", a: "Ein automatisches Tool, das Audio-, Video- oder Textdateien mit KI in Transkripte, Zusammenfassungen, Übersetzungen und mehr umwandelt." },
    { q: "Welche Dateiformate werden unterstützt?", a: "Wir unterstützen .mp3, .mp4, .wav, .m4a, .mov und andere gängige Formate." },
    { q: "Muss ich ein Konto erstellen?", a: "Nein. Ouviescrevi ist kostenlos und erfordert keine Registrierung." },
    { q: "Ist die Transkription 100 % genau?", a: "Die Genauigkeit hängt von der Audioqualität ab. Wir nutzen OpenAIs Whisper-Modell für hohe Qualität." },
    { q: "Werden meine Dateien gespeichert?", a: "Dateien werden nach der Verarbeitung gelöscht. Nur Dateiname und Zeitstempel werden für Statistiken gespeichert." },
  ];

  var FAQ_BY_LANG = { pt: FAQ_PT, en: FAQ_EN, es: FAQ_ES, fr: FAQ_FR, de: FAQ_DE };

  function pagePath() {
    var p = (global.location.pathname || "/").replace(/\/$/, "");
    if (!p || p === "") return "/index.html";
    if (p.indexOf(".") === -1) return p + "/index.html";
    return p;
  }

  function upsertMeta(attr, key, value) {
    if (!value) return;
    var sel = "meta[" + attr + '="' + key + '"]';
    var el = document.querySelector(sel);
    if (!el) {
      el = document.createElement("meta");
      el.setAttribute(attr, key);
      document.head.appendChild(el);
    }
    el.setAttribute("content", value);
  }

  function upsertLink(rel, href, extra) {
    var sel = 'link[rel="' + rel + '"]';
    if (extra && extra.hreflang) sel += '[hreflang="' + extra.hreflang + '"]';
    var el = document.querySelector(sel);
    if (!el) {
      el = document.createElement("link");
      el.setAttribute("rel", rel);
      document.head.appendChild(el);
    }
    el.setAttribute("href", href);
    if (extra && extra.hreflang) el.setAttribute("hreflang", extra.hreflang);
  }

  function injectJsonLd(data) {
    var el = document.createElement("script");
    el.type = "application/ld+json";
    el.textContent = JSON.stringify(data);
    document.head.appendChild(el);
  }

  function breadcrumb(path, title) {
    var parts = path.split("/").filter(Boolean);
    var items = [{ "@type": "ListItem", position: 1, name: "Ouviescrevi", item: SITE + "/index.html" }];
    if (parts.length && parts[parts.length - 1] !== "index.html") {
      items.push({
        "@type": "ListItem",
        position: 2,
        name: title,
        item: SITE + path,
      });
    }
    return {
      "@context": "https://schema.org",
      "@type": "BreadcrumbList",
      itemListElement: items,
    };
  }

  function apply() {
    var path = pagePath();
    if (path.indexOf("backoffice") !== -1 || path.indexOf("/archive/") !== -1) return;

    var cfg = PAGES[path] || {};
    var title = cfg.title || document.title || "Ouviescrevi";
    var desc =
      cfg.description ||
      (document.querySelector('meta[name="description"]') || {}).content ||
      "Transcrição de áudio e vídeo com IA — Ouviescrevi";
    var lang = cfg.lang;
    if (!lang) {
      var localeMatch = path.match(/^\/(en|es|fr|de)(\/|$)/);
      lang = localeMatch ? localeMatch[1] : "pt";
    }
    var OG_LOCALES = { pt: "pt_PT", en: "en_GB", es: "es_ES", fr: "fr_FR", de: "de_DE" };
    var IN_LANG = { pt: "pt-PT", en: "en", es: "es", fr: "fr", de: "de" };
    var canonical = SITE + path;

    document.title = title;
    upsertMeta("name", "description", desc);
    upsertMeta("name", "robots", cfg.noindex ? "noindex, follow" : "index, follow");
    upsertMeta("name", "author", "Ouviescrevi");
    upsertLink("canonical", canonical);

    upsertMeta("property", "og:type", cfg.type === "home" ? "website" : "article");
    upsertMeta("property", "og:site_name", "Ouviescrevi");
    upsertMeta("property", "og:title", title);
    upsertMeta("property", "og:description", desc);
    upsertMeta("property", "og:url", canonical);
    upsertMeta("property", "og:image", OG_IMAGE);
    upsertMeta("property", "og:locale", OG_LOCALES[lang] || "pt_PT");

    upsertMeta("name", "twitter:card", "summary_large_image");
    upsertMeta("name", "twitter:title", title);
    upsertMeta("name", "twitter:description", desc);
    upsertMeta("name", "twitter:image", OG_IMAGE);

    var alternates = HREFLANG[path];
    if (alternates) {
      Object.keys(alternates).forEach(function (loc) {
        upsertLink("alternate", SITE + alternates[loc], { hreflang: loc });
      });
      if (alternates.pt) {
        upsertLink("alternate", SITE + alternates.pt, { hreflang: "x-default" });
      }
    }

    injectJsonLd(breadcrumb(path, title));
    injectJsonLd({
      "@context": "https://schema.org",
      "@type": "WebPage",
      name: title,
      description: desc,
      url: canonical,
      inLanguage: IN_LANG[lang] || "pt-PT",
      isPartOf: { "@id": SITE + "/#website" },
    });

    if (cfg.type === "home") {
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "WebSite",
        "@id": SITE + "/#website",
        name: "Ouviescrevi",
        url: SITE + "/",
        description: desc,
        inLanguage: ["pt-PT", "en", "es", "fr", "de"],
        publisher: { "@id": ORG_ID },
      });
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "Organization",
        "@id": ORG_ID,
        name: "Ouviescrevi",
        url: SITE + "/",
        logo: OG_IMAGE,
        email: "ouviescrevi@gmail.com",
        areaServed: "PT",
      });
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
        name: "Ouviescrevi",
        applicationCategory: "BusinessApplication",
        operatingSystem: "Web",
        offers: { "@type": "Offer", price: "0", priceCurrency: "EUR" },
        description: desc,
        url: canonical,
      });
    }

    if (cfg.faq) {
      var faqItems = FAQ_BY_LANG[lang] || FAQ_PT;
      injectJsonLd({
        "@context": "https://schema.org",
        "@type": "FAQPage",
        mainEntity: faqItems.map(function (item) {
          return {
            "@type": "Question",
            name: item.q,
            acceptedAnswer: { "@type": "Answer", text: item.a },
          };
        }),
      });
    }
  }

  function applyOverrides(seoMap) {
    if (!seoMap) return;
    var path = pagePath();
    var o = seoMap[path];
    if (!o) return;
    if (o.title) document.title = o.title;
    if (o.description) upsertMeta("name", "description", o.description);
    if (o.title) upsertMeta("property", "og:title", o.title);
    if (o.description) upsertMeta("property", "og:description", o.description);
  }

  global.OuviescreviSEO = { apply: apply, pagePath: pagePath, applyOverrides: applyOverrides, PAGES: PAGES };
})(window);
