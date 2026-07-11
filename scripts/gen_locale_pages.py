#!/usr/bin/env python3
"""Gera páginas de preços e landings de audiência ES/FR/DE a partir de templates."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

PRICING_STYLE = """    body.oe-page--pricing .oe-pricing-hero { text-align: center; max-width: 720px; margin: 32px auto 24px; padding: 0 16px; }
    body.oe-page--pricing .oe-pricing-hero h1 { font-size: 2rem; margin-bottom: 8px; color: var(--pro-text, #0f172a); }
    body.oe-page--pricing .oe-pricing-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 20px; max-width: 900px; margin: 0 auto 40px; padding: 0 16px;
    }
    body.oe-page--pricing .oe-pricing-card {
      border: 1px solid var(--pro-border, #e2e8f0); border-radius: 16px; padding: 28px 24px;
      background: var(--pro-surface, #fff);
      box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    }
    body.oe-page--pricing .oe-pricing-card--pro {
      border-color: #8b5cf6; box-shadow: 0 8px 32px rgba(109, 40, 217, 0.12);
    }
    body.oe-page--pricing .oe-pricing-card h2 { margin: 0 0 8px; font-size: 1.25rem; color: var(--pro-text, #0f172a); }
    body.oe-page--pricing .oe-pricing-price { font-size: 1.8rem; font-weight: 700; margin: 12px 0; color: #6d28d9; }
    body.oe-page--pricing .oe-pricing-card ul { margin: 16px 0 24px; padding-left: 1.2rem; line-height: 1.7; color: var(--pro-text-secondary, #475569); }
    body.oe-page--pricing .oe-pricing-cta {
      display: flex; align-items: center; justify-content: center;
      width: 100%; min-height: 48px; text-align: center; padding: 12px 20px;
      border-radius: 12px; font-weight: 700; font-size: 1rem;
      text-decoration: none; border: none; cursor: pointer;
      transition: transform .15s ease, box-shadow .15s ease;
    }
    body.oe-page--pricing .oe-pricing-cta:hover { transform: translateY(-1px); }
    body.oe-page--pricing .oe-pricing-cta--primary {
      background: linear-gradient(135deg, #7c3aed, #2563eb); color: #fff;
      box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35);
    }
    body.oe-page--pricing .oe-pricing-cta--ghost {
      background: #fff; color: #6d28d9;
      border: 2px solid #8b5cf6;
      box-shadow: 0 2px 12px rgba(139, 92, 246, 0.12);
    }
    body.oe-page--pricing .oe-pricing-note { text-align: center; opacity: .85; max-width: 560px; margin: 0 auto 48px; padding: 0 16px; color: var(--pro-text-muted, #64748b); }
    body.oe-page--pricing .oe-pricing-badge {
      display: inline-block; font-size: .75rem; font-weight: 600;
      background: #ede9fe; color: #6d28d9; padding: 4px 10px; border-radius: 999px; margin-bottom: 8px;
    }
    body.oe-page--pricing .oe-pricing-free-panel {
      max-width: 520px; margin: 0 auto 48px; padding: 0 16px; text-align: center;
    }
    body.oe-page--pricing .oe-pricing-free-panel .oe-pricing-card { text-align: left; }"""

PRICING = {
    "es": {
        "lang": "es",
        "skip": "Saltar al contenido",
        "desc": "Planes Ouviescrevi para transcripción con IA, subtítulos SRT y exportación. Opciones gratuitas y Pro.",
        "title": "Precios y planes — Transcripción con IA | Ouviescrevi",
        "h1": "Planes simples, sin sorpresas",
        "h1free": "Gratis para empezar",
        "intro": "Empieza gratis. Pasa a Pro cuando necesites más transcripciones, exportación DOCX y límites mayores.",
        "introfree": "Ouviescrevi es gratuito — transcribe clases y grabaciones, exporta en varios formatos y guarda historial en tu cuenta.",
        "free": "Gratis",
        "pro": "Pro",
        "badge": "Recomendado",
        "price": "0 €",
        "proprice": "9,99 €/mes",
        "start": "Empezar gratis",
        "subscribe": "Suscribirse a Pro",
        "included": "Qué incluye",
        "footnote": "Pagos seguros con Stripe. Cancela cuando quieras. Hecho en Portugal 🇵🇹",
        "toastOk": "Suscripción recibida — ¡gracias! Pro se activará pronto.",
        "checkoutSoon": "Checkout Pro en preparación — crea una cuenta para avisarte cuando abra.",
        "checkoutBtn": "Crear cuenta (avisarme)",
        "loginSub": "Inicia sesión para suscribirte.",
        "li1": "3 transcripciones/día (anónimo)",
        "li2": "20/día con cuenta",
        "li3": "Historial en la cuenta",
        "li4": "Exportar TXT, PDF, SRT",
        "liPro1": "transcripciones/día",
        "liPro2": "Exportación <strong>DOCX</strong> profesional",
        "liPro3": "Historial ampliado",
        "liPro4": "Prioridad en el procesamiento",
        "liFree1": "Transcripción con IA",
        "liFree2": "3/día sin cuenta · 20/día con cuenta",
    },
    "fr": {
        "lang": "fr",
        "skip": "Aller au contenu",
        "desc": "Offres Ouviescrevi pour la transcription IA, sous-titres SRT et export. Options gratuites et Pro.",
        "title": "Tarifs et offres — Transcription IA | Ouviescrevi",
        "h1": "Des offres simples, sans surprise",
        "h1free": "Gratuit pour commencer",
        "intro": "Commencez gratuitement. Passez à Pro pour plus de transcriptions, l'export DOCX et des limites plus élevées.",
        "introfree": "Ouviescrevi est gratuit — transcrivez cours et enregistrements, exportez en plusieurs formats et gardez l'historique.",
        "free": "Gratuit",
        "pro": "Pro",
        "badge": "Recommandé",
        "price": "0 €",
        "proprice": "9,99 €/mois",
        "start": "Commencer gratuitement",
        "subscribe": "S'abonner à Pro",
        "included": "Ce qui est inclus",
        "footnote": "Paiements sécurisés via Stripe. Annulez quand vous voulez. Fait au Portugal 🇵🇹",
        "toastOk": "Abonnement reçu — merci ! Pro s'active bientôt.",
        "checkoutSoon": "Checkout Pro bientôt — créez un compte pour être averti à l'ouverture.",
        "checkoutBtn": "Créer un compte (me prévenir)",
        "loginSub": "Connectez-vous pour vous abonner.",
        "li1": "3 transcriptions/jour (anonyme)",
        "li2": "20/jour avec compte",
        "li3": "Historique dans le compte",
        "li4": "Export TXT, PDF, SRT",
        "liPro1": "transcriptions/jour",
        "liPro2": "Export <strong>DOCX</strong> professionnel",
        "liPro3": "Historique étendu",
        "liPro4": "Priorité de traitement",
        "liFree1": "Transcription IA",
        "liFree2": "3/jour sans compte · 20/jour avec compte",
    },
    "de": {
        "lang": "de",
        "skip": "Zum Inhalt springen",
        "desc": "Ouviescrevi-Tarife für KI-Transkription, SRT-Untertitel und Export. Gratis- und Pro-Optionen.",
        "title": "Preise & Pläne — KI-Transkription | Ouviescrevi",
        "h1": "Einfache Pläne, keine Überraschungen",
        "h1free": "Kostenlos starten",
        "intro": "Starte gratis. Wechsle zu Pro für mehr Transkriptionen, DOCX-Export und höhere Limits.",
        "introfree": "Ouviescrevi ist kostenlos — transkribiere Aufnahmen, exportiere in mehreren Formaten und speichere den Verlauf.",
        "free": "Gratis",
        "pro": "Pro",
        "badge": "Empfohlen",
        "price": "0 €",
        "proprice": "9,99 €/Monat",
        "start": "Kostenlos starten",
        "subscribe": "Pro abonnieren",
        "included": "Was enthalten ist",
        "footnote": "Sichere Zahlung via Stripe. Jederzeit kündbar. Made in Portugal 🇵🇹",
        "toastOk": "Abo erhalten — danke! Pro wird bald aktiviert.",
        "checkoutSoon": "Pro-Checkout kommt bald — Konto erstellen für Benachrichtigung.",
        "checkoutBtn": "Konto erstellen (benachrichtigen)",
        "loginSub": "Zum Abonnieren anmelden.",
        "li1": "3 Transkriptionen/Tag (anonym)",
        "li2": "20/Tag mit Konto",
        "li3": "Verlauf im Konto",
        "li4": "Export TXT, PDF, SRT",
        "liPro1": "Transkriptionen/Tag",
        "liPro2": "Professioneller <strong>DOCX</strong>-Export",
        "liPro3": "Erweiterter Verlauf",
        "liPro4": "Verarbeitungspriorität",
        "liFree1": "KI-Transkription",
        "liFree2": "3/Tag ohne Konto · 20/Tag mit Konto",
    },
}

LANDINGS = {
    "professores": {
        "es": {
            "title": "IA para profesores — Transcribir y resumir clases | Ouviescrevi",
            "desc": "Herramientas de IA para educadores: transcripción de clases, resúmenes automáticos y exportación DOCX.",
            "body_class": "oe-page--landing-pro",
            "main": """<section class="oe-lp-hero">
    <p class="oe-lp-badge">🎓 Para profesores y formadores</p>
    <h1>Graba la clase. Obtén texto, resúmenes y materiales en minutos.</h1>
    <p class="oe-lp-lead">Ouviescrevi convierte audio y vídeo de clases en transcripciones editables — ideal para apuntes, accesibilidad y preparación de exámenes.</p>
    <div class="oe-lp-cta-row">
      <a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir clase gratis</a>
      <a href="precos.html" class="oe-lp-btn oe-lp-btn--ghost" data-pricing-only>Ver plan Pro (DOCX)</a>
    </div>
  </section>
  <div class="oe-lp-grid">
    <article class="oe-lp-card"><h2>⏱️ Ahorra horas</h2><p>Una clase de 60 minutos ya no significa horas escuchando y tecleando. Revisa y edita el texto generado por IA.</p></article>
    <article class="oe-lp-card"><h2>📚 Materiales para alumnos</h2><p>Exporta PDF, TXT o <strong><span data-pricing-only>DOCX (Pro)</span><span data-pricing-free-only hidden>DOCX</span></strong>. Combina con resúmenes y preguntas en el mismo sitio.</p></article>
    <article class="oe-lp-card"><h2>🇵🇹 Portugués nativo</h2><p>Detección automática de idioma, acentos portugueses e interfaz hecha en Portugal.</p></article>
    <article class="oe-lp-card"><h2>🔒 Privacidad</h2><p>Archivos temporales eliminados automáticamente. Cuenta gratuita con historial.</p></article>
  </div>
  <section class="oe-lp-steps"><h2 style="text-align:center;margin-bottom:16px">Cómo funciona</h2><ol>
    <li>Graba la clase en el móvil o exporta el vídeo (Zoom, Teams, etc.).</li>
    <li>Sube el archivo a <a href="index.html">Transcribir</a> — hasta 500 MB gratis.</li>
    <li>Usa <a href="aula-completa.html">Clase completa</a> para resúmenes, preguntas y flashcards — o exporta PDF/DOCX.</li>
  </ol></section>
  <div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">Probar ahora — gratis</a><a href="aula-completa.html" class="oe-lp-btn oe-lp-btn--ghost">📚 Clase completa</a></div>""",
        },
        "fr": {
            "title": "IA pour enseignants — Transcrire et résumer les cours | Ouviescrevi",
            "desc": "Outils IA pour éducateurs : transcription de cours, résumés automatiques et export DOCX.",
            "body_class": "oe-page--landing-pro",
            "main": """<section class="oe-lp-hero">
    <p class="oe-lp-badge">🎓 Pour enseignants et formateurs</p>
    <h1>Enregistrez le cours. Obtenez texte, résumés et supports en quelques minutes.</h1>
    <p class="oe-lp-lead">Ouviescrevi transforme l'audio et la vidéo de cours en transcriptions éditables — idéal pour révisions, accessibilité et préparation d'examens.</p>
    <div class="oe-lp-cta-row">
      <a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire un cours gratuitement</a>
      <a href="precos.html" class="oe-lp-btn oe-lp-btn--ghost" data-pricing-only>Voir l'offre Pro (DOCX)</a>
    </div>
  </section>
  <div class="oe-lp-grid">
    <article class="oe-lp-card"><h2>⏱️ Gagnez du temps</h2><p>Un cours de 60 minutes ne signifie plus des heures d'écoute et de saisie. Relisez et éditez le texte généré par l'IA.</p></article>
    <article class="oe-lp-card"><h2>📚 Supports pour élèves</h2><p>Export PDF, TXT ou <strong><span data-pricing-only>DOCX (Pro)</span><span data-pricing-free-only hidden>DOCX</span></strong>. Combinez résumés et questions sur le même site.</p></article>
    <article class="oe-lp-card"><h2>🇵🇹 Portugais natif</h2><p>Détection automatique de langue, accents portugais et interface faite au Portugal.</p></article>
    <article class="oe-lp-card"><h2>🔒 Confidentialité</h2><p>Fichiers temporaires supprimés automatiquement. Compte gratuit avec historique.</p></article>
  </div>
  <section class="oe-lp-steps"><h2 style="text-align:center;margin-bottom:16px">Comment ça marche</h2><ol>
    <li>Enregistrez le cours sur téléphone ou exportez la vidéo (Zoom, Teams, etc.).</li>
    <li>Téléversez sur <a href="index.html">Transcrire</a> — jusqu'à 500 Mo gratuits.</li>
    <li>Utilisez <a href="aula-completa.html">Cours complet</a> pour résumés, questions et flashcards — ou exportez PDF/DOCX.</li>
  </ol></section>
  <div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">Essayer maintenant — gratuit</a><a href="aula-completa.html" class="oe-lp-btn oe-lp-btn--ghost">📚 Cours complet</a></div>""",
        },
        "de": {
            "title": "KI für Lehrkräfte — Unterricht transkribieren & zusammenfassen | Ouviescrevi",
            "desc": "KI-Tools für Bildung: Unterrichtstranskription, automatische Zusammenfassungen und DOCX-Export.",
            "body_class": "oe-page--landing-pro",
            "main": """<section class="oe-lp-hero">
    <p class="oe-lp-badge">🎓 Für Lehrkräfte und Trainer</p>
    <h1>Unterricht aufnehmen. Text, Zusammenfassungen und Material in Minuten.</h1>
    <p class="oe-lp-lead">Ouviescrevi verwandelt Unterrichts-Audio und -Video in bearbeitbare Transkripte — ideal für Lernmaterial, Barrierefreiheit und Prüfungsvorbereitung.</p>
    <div class="oe-lp-cta-row">
      <a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Unterricht kostenlos transkribieren</a>
      <a href="precos.html" class="oe-lp-btn oe-lp-btn--ghost" data-pricing-only>Pro-Plan ansehen (DOCX)</a>
    </div>
  </section>
  <div class="oe-lp-grid">
    <article class="oe-lp-card"><h2>⏱️ Stunden sparen</h2><p>60 Minuten Unterricht bedeuten nicht mehr stundenlanges Abhören und Tippen. Text von der KI prüfen und bearbeiten.</p></article>
    <article class="oe-lp-card"><h2>📚 Material für Schüler</h2><p>Export als PDF, TXT oder <strong><span data-pricing-only>DOCX (Pro)</span><span data-pricing-free-only hidden>DOCX</span></strong>. Zusammen mit Zusammenfassungen und Fragen.</p></article>
    <article class="oe-lp-card"><h2>🇵🇹 Portugiesisch im Fokus</h2><p>Automatische Spracherkennung, portugiesische Akzente und Oberfläche aus Portugal.</p></article>
    <article class="oe-lp-card"><h2>🔒 Datenschutz</h2><p>Temporäre Dateien werden automatisch gelöscht. Gratis-Konto mit Verlauf.</p></article>
  </div>
  <section class="oe-lp-steps"><h2 style="text-align:center;margin-bottom:16px">So funktioniert's</h2><ol>
    <li>Unterricht am Handy aufnehmen oder Video exportieren (Zoom, Teams, etc.).</li>
    <li>Datei bei <a href="index.html">Transkribieren</a> hochladen — bis 500 MB gratis.</li>
    <li><a href="aula-completa.html">Voller Unterricht</a> für Zusammenfassungen, Fragen und Karteikarten — oder PDF/DOCX exportieren.</li>
  </ol></section>
  <div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">Jetzt testen — gratis</a><a href="aula-completa.html" class="oe-lp-btn oe-lp-btn--ghost">📚 Voller Unterricht</a></div>""",
        },
    },
    "podcasts": {
        "es": {"title": "Transcribir podcasts con IA | Ouviescrevi", "desc": "Convierte episodios de podcast en transcripciones y resúmenes con inteligencia artificial.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎙️ Para creadores de podcast</p><h1>Del audio al artículo en minutos</h1><div class="oe-lp-lead"><p>Convierte episodios en texto con un clic. Mejora el SEO, comparte transcripciones y reutiliza contenido.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir episodio gratis</a><a href="podcast-youtube.html" class="oe-lp-btn oe-lp-btn--ghost">🎬 Asistente Podcast/YouTube</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🔍 Mejor SEO</h2><p>Publica la transcripción para que Google indexe palabras clave del audio.</p></article><article class="oe-lp-card"><h2>♻️ Reutiliza contenido</h2><p>Newsletter, LinkedIn o blog a partir del mismo episodio.</p></article><article class="oe-lp-card"><h2>📺 Listo para YouTube</h2><p>De audio a capítulos y descripción — <a href="podcast-youtube.html">asistente en 3 pasos</a>.</p></article></div>"""},
        "fr": {"title": "Transcrire des podcasts avec l'IA | Ouviescrevi", "desc": "Transformez vos épisodes de podcast en transcriptions et résumés avec l'intelligence artificielle.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎙️ Pour créateurs de podcasts</p><h1>De l'audio à l'article en quelques minutes</h1><div class="oe-lp-lead"><p>Transformez vos épisodes en texte en un clic. Améliorez le SEO et réutilisez le contenu.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire un épisode gratuitement</a><a href="podcast-youtube.html" class="oe-lp-btn oe-lp-btn--ghost">🎬 Assistant Podcast/YouTube</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🔍 Meilleur SEO</h2><p>Publiez la transcription pour indexer les mots-clés parlés.</p></article><article class="oe-lp-card"><h2>♻️ Réutiliser le contenu</h2><p>Newsletter, LinkedIn ou blog à partir du même épisode.</p></article><article class="oe-lp-card"><h2>📺 Prêt pour YouTube</h2><p>De l'audio aux chapitres et description — <a href="podcast-youtube.html">assistant en 3 étapes</a>.</p></article></div>"""},
        "de": {"title": "Podcasts mit KI transkribieren | Ouviescrevi", "desc": "Wandle Podcast-Folgen in Transkripte und Zusammenfassungen mit KI um.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎙️ Für Podcast-Ersteller</p><h1>Vom Audio zum Artikel in Minuten</h1><div class="oe-lp-lead"><p>Verwandle Folgen mit einem Klick in Text. Besseres SEO und Content-Wiederverwendung.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Folge kostenlos transkribieren</a><a href="podcast-youtube.html" class="oe-lp-btn oe-lp-btn--ghost">🎬 Podcast/YouTube-Assistent</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🔍 Besseres SEO</h2><p>Transkript veröffentlichen, damit Google gesprochene Keywords indexiert.</p></article><article class="oe-lp-card"><h2>♻️ Content wiederverwenden</h2><p>Newsletter, LinkedIn oder Blog aus derselben Folge.</p></article><article class="oe-lp-card"><h2>📺 YouTube-ready</h2><p>Vom Audio zu Kapiteln und Beschreibung — <a href="podcast-youtube.html">3-Schritte-Assistent</a>.</p></article></div>"""},
    },
    "aulas": {
        "es": {"title": "Transcripción de clases con IA | Ouviescrevi", "desc": "Convierte grabaciones de clases en texto editable con IA.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎥 Para estudiantes y educación</p><h1>Clases en texto, listas para estudiar</h1><div class="oe-lp-lead"><p>Convierte vídeos de clase en texto claro. Útil para alumnos, profesores y plataformas educativas.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir clase gratis</a><a href="aula-pronta.html" class="oe-lp-btn oe-lp-btn--ghost">📦 Clase lista</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📚 Apoyo al estudio</h2><p>Repasa explicaciones grabadas sin ver todo el vídeo — busca en el texto.</p></article><article class="oe-lp-card"><h2>♿ Accesibilidad</h2><p>Comparte transcripciones con alumnos que necesitan apoyo escrito.</p></article><article class="oe-lp-card"><h2>📦 Pack completo</h2><p>Resumen, glosario y preguntas con la herramienta Clase lista.</p></article></div>"""},
        "fr": {"title": "Transcription de cours avec l'IA | Ouviescrevi", "desc": "Transformez les enregistrements de cours en texte éditable avec l'IA.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎥 Pour étudiants et éducation</p><h1>Cours en texte, prêts à réviser</h1><div class="oe-lp-lead"><p>Transformez les vidéos de cours en texte clair. Utile pour élèves, enseignants et plateformes.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire un cours gratuitement</a><a href="aula-pronta.html" class="oe-lp-btn oe-lp-btn--ghost">📦 Cours prêt</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📚 Aide à la révision</h2><p>Revoyez les explications sans revoir toute la vidéo — cherchez dans le texte.</p></article><article class="oe-lp-card"><h2>♿ Accessibilité</h2><p>Partagez des transcriptions avec les élèves qui ont besoin de support écrit.</p></article><article class="oe-lp-card"><h2>📦 Pack complet</h2><p>Résumé, glossaire et questions avec l'outil Cours prêt.</p></article></div>"""},
        "de": {"title": "Unterrichtstranskription mit KI | Ouviescrevi", "desc": "Verwandle Unterrichtsaufnahmen in bearbeitbaren Text mit KI.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🎥 Für Schüler und Bildung</p><h1>Unterricht als Text, bereit zum Lernen</h1><div class="oe-lp-lead"><p>Verwandle Unterrichtsvideos in klaren Text. Nützlich für Schüler, Lehrkräfte und Plattformen.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Unterricht kostenlos transkribieren</a><a href="aula-pronta.html" class="oe-lp-btn oe-lp-btn--ghost">📦 Unterricht fertig</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📚 Lernhilfe</h2><p>Aufnahmen durchsuchen, ohne das ganze Video erneut anzusehen.</p></article><article class="oe-lp-card"><h2>♿ Barrierefreiheit</h2><p>Transkripte für Lernende mit Bedarf an schriftlicher Wiederholung teilen.</p></article><article class="oe-lp-card"><h2>📦 Komplettpaket</h2><p>Zusammenfassung, Glossar und Fragen mit Unterricht fertig.</p></article></div>"""},
    },
    "jornalistas": {
        "es": {"title": "Transcripción de entrevistas para periodistas | Ouviescrevi", "desc": "Transcribe entrevistas con IA. Ahorra horas y exporta texto listo para editar.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">📰 Para periodistas y redacciones</p><h1>Entrevistas transcritas en minutos</h1><div class="oe-lp-lead"><p>Transcribe entrevistas, ruedas de prensa o reportajes con precisión y velocidad.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir entrevista gratis</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Corregir texto</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>⏱️ Plazos ajustados</h2><p>Una hora de audio ya no significa horas escuchando y tecleando.</p></article><article class="oe-lp-card"><h2>📝 Citas fiables</h2><p>Busca pasajes, corrige nombres y exporta PDF o TXT.</p></article><article class="oe-lp-card"><h2>🔒 Discreción</h2><p>Archivos temporales eliminados automáticamente.</p></article></div>"""},
        "fr": {"title": "Transcription d'entretiens pour journalistes | Ouviescrevi", "desc": "Transcrivez des entretiens avec l'IA. Gagnez du temps et exportez du texte prêt à éditer.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">📰 Pour journalistes et rédactions</p><h1>Entretiens transcrits en quelques minutes</h1><div class="oe-lp-lead"><p>Transcrivez entretiens, conférences de presse ou reportages avec précision et rapidité.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire un entretien gratuitement</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Corriger le texte</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>⏱️ Délais serrés</h2><p>Une heure d'audio ne signifie plus des heures d'écoute et de saisie.</p></article><article class="oe-lp-card"><h2>📝 Citations fiables</h2><p>Cherchez des passages, corrigez les noms et exportez PDF ou TXT.</p></article><article class="oe-lp-card"><h2>🔒 Discrétion</h2><p>Fichiers temporaires supprimés automatiquement.</p></article></div>"""},
        "de": {"title": "Interview-Transkription für Journalisten | Ouviescrevi", "desc": "Transkribiere Interviews mit KI. Spare Stunden und exportiere bearbeitbaren Text.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">📰 Für Journalisten und Redaktionen</p><h1>Interviews in Minuten transkribiert</h1><div class="oe-lp-lead"><p>Transkribiere Interviews, Pressekonferenzen oder Reportagen schnell und präzise.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Interview kostenlos transkribieren</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Text korrigieren</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>⏱️ Enge Deadlines</h2><p>Eine Stunde Audio bedeutet nicht mehr stundenlanges Abhören und Tippen.</p></article><article class="oe-lp-card"><h2>📝 Zuverlässige Zitate</h2><p>Passagen suchen, Namen korrigieren und als PDF oder TXT exportieren.</p></article><article class="oe-lp-card"><h2>🔒 Diskretion</h2><p>Temporäre Dateien werden automatisch gelöscht.</p></article></div>"""},
    },
    "reunioes": {
        "es": {"title": "Transcripción de reuniones y actas con IA | Ouviescrevi", "desc": "Graba y transcribe reuniones automáticamente. Genera actas y resúmenes.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🗣️ Para equipos y empresas</p><h1>Reuniones con actas automáticas</h1><div class="oe-lp-lead"><p>Graba y transcribe reuniones presenciales u online. Ideal para equipos y empresas.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir reunión gratis</a><a href="resumo.html" class="oe-lp-btn oe-lp-btn--ghost">📩 Generar acta</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📋 Actas rápidas</h2><p>Exporta el texto y resume puntos clave el mismo día.</p></article><article class="oe-lp-card"><h2>👥 Quién dijo qué</h2><p>Identificación de hablantes para actas con nombres o roles.</p></article><article class="oe-lp-card"><h2>💼 Zoom y Teams</h2><p>Sube la grabación exportada — hasta 500 MB gratis.</p></article></div>"""},
        "fr": {"title": "Transcription de réunions et comptes rendus IA | Ouviescrevi", "desc": "Enregistrez et transcrivez les réunions automatiquement. Générez des comptes rendus.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🗣️ Pour équipes et entreprises</p><h1>Réunions avec compte rendu automatique</h1><div class="oe-lp-lead"><p>Enregistrez et transcrivez réunions présentielles ou en ligne. Idéal pour équipes et entreprises.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire une réunion gratuitement</a><a href="resumo.html" class="oe-lp-btn oe-lp-btn--ghost">📩 Générer un compte rendu</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📋 Comptes rendus rapides</h2><p>Exportez le texte et résumez les points clés le jour même.</p></article><article class="oe-lp-card"><h2>👥 Qui a dit quoi</h2><p>Identification des locuteurs pour des comptes rendus nominatifs.</p></article><article class="oe-lp-card"><h2>💼 Zoom et Teams</h2><p>Téléversez l'enregistrement exporté — jusqu'à 500 Mo gratuits.</p></article></div>"""},
        "de": {"title": "Meeting-Transkription und Protokolle mit KI | Ouviescrevi", "desc": "Meetings aufnehmen und automatisch transkribieren. Protokolle und Zusammenfassungen erstellen.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🗣️ Für Teams und Unternehmen</p><h1>Meetings mit automatischen Protokollen</h1><div class="oe-lp-lead"><p>Nehme Meetings vor Ort oder online auf und transkribiere sie. Ideal für Teams und Unternehmen.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Meeting kostenlos transkribieren</a><a href="resumo.html" class="oe-lp-btn oe-lp-btn--ghost">📩 Protokoll erstellen</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>📋 Schnelle Protokolle</h2><p>Text exportieren und Kernpunkte noch am selben Tag teilen.</p></article><article class="oe-lp-card"><h2>👥 Wer sagte was</h2><p>Sprechererkennung für Protokolle mit Namen oder Rollen.</p></article><article class="oe-lp-card"><h2>💼 Zoom und Teams</h2><p>Exportierte Aufnahme hochladen — bis 500 MB gratis.</p></article></div>"""},
    },
    "testemunhos": {
        "es": {"title": "Transcripción de testimonios y declaraciones | Ouviescrevi", "desc": "Transcribe testimonios y declaraciones con precisión. Útil en contextos legales.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🧑‍⚖️ Para contextos formales</p><h1>Testimonios con registro escrito</h1><div class="oe-lp-lead"><p>Ideal para testimonios legales, audiencias y declaraciones. Garantiza precisión y registro escrito.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcribir grabación gratis</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Revisar texto</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🎯 Precisión</h2><p>Revisa y edita el texto generado — mantienes el control editorial.</p></article><article class="oe-lp-card"><h2>📄 Exportar</h2><p>Guarda PDF o TXT para archivos y equipos.</p></article><article class="oe-lp-card"><h2>🔒 Confidencialidad</h2><p>Archivos temporales eliminados automáticamente.</p></article></div>"""},
        "fr": {"title": "Transcription de témoignages et déclarations | Ouviescrevi", "desc": "Transcrivez témoignages et déclarations avec précision. Utile en contexte juridique.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🧑‍⚖️ Pour contextes formels</p><h1>Témoignages avec trace écrite</h1><div class="oe-lp-lead"><p>Idéal pour témoignages juridiques, audiences et déclarations. Assure précision et trace écrite.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Transcrire un enregistrement gratuitement</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Relire le texte</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🎯 Précision</h2><p>Relisez et éditez le texte généré — vous gardez le contrôle éditorial.</p></article><article class="oe-lp-card"><h2>📄 Export</h2><p>Enregistrez en PDF ou TXT pour archives et équipes.</p></article><article class="oe-lp-card"><h2>🔒 Confidentialité</h2><p>Fichiers temporaires supprimés automatiquement.</p></article></div>"""},
        "de": {"title": "Transkription von Zeugenaussagen | Ouviescrevi", "desc": "Transkribiere Zeugenaussagen und Erklärungen präzise. Nützlich in rechtlichen Kontexten.", "main": """<section class="oe-lp-hero"><p class="oe-lp-badge">🧑‍⚖️ Für formelle Kontexte</p><h1>Aussagen mit schriftlicher Dokumentation</h1><div class="oe-lp-lead"><p>Ideal für rechtliche Zeugenaussagen, Anhörungen und Erklärungen. Präzision und schriftliche Spur.</p></div><div class="oe-lp-cta-row"><a href="index.html" class="oe-lp-btn oe-lp-btn--primary">🎙️ Aufnahme kostenlos transkribieren</a><a href="corretor.html" class="oe-lp-btn oe-lp-btn--ghost">✍️ Text prüfen</a></div></section><div class="oe-lp-grid"><article class="oe-lp-card"><h2>🎯 Genauigkeit</h2><p>Generierten Text prüfen und bearbeiten — volle redaktionelle Kontrolle.</p></article><article class="oe-lp-card"><h2>📄 Export</h2><p>Als PDF oder TXT für Archive und Teams speichern.</p></article><article class="oe-lp-card"><h2>🔒 Vertraulichkeit</h2><p>Temporäre Dateien werden automatisch gelöscht.</p></article></div>"""},
    },
}

SKIP = {"es": "Saltar al contenido", "fr": "Aller au contenu", "de": "Zum Inhalt springen"}


def pricing_html(loc: str, t: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="{t['lang']}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{t['desc']}">
  <title>{t['title']}</title>
  <link rel="icon" href="/logos/ouviescrevi-icon-pro.png" type="image/png">
  <link rel="stylesheet" href="../css/ouviescrevi.css">
  <link rel="stylesheet" href="../css/ouviescrevi-pro.css">
  <link rel="stylesheet" href="../css/index-home.css?v=18">
  <script src="/js/ouviescrevi-seo.js" defer></script>
  <script src="../js/pricing-visibility.js" defer></script>
  <script src="../js/ouviescrevi-ui.js" defer></script>
  <script src="../js/ouviescrevi-api.js" defer></script>
  <script src="../js/auth-ui.js" defer></script>
  <style>
{PRICING_STYLE}
  </style>
</head>
<body class="oe-page oe-pro oe-page--pricing">
<a class="oe-skip" href="#conteudo">{t['skip']}</a>
<div id="header"></div>
<main id="conteudo">
  <section class="oe-pricing-hero">
    <h1 data-pricing-only>{t['h1']}</h1>
    <h1 data-pricing-free-only hidden>{t['h1free']}</h1>
    <p data-pricing-only>{t['intro']}</p>
    <p data-pricing-free-only hidden>{t['introfree']}</p>
  </section>
  <div class="oe-pricing-grid" data-pricing-only>
    <article class="oe-pricing-card">
      <h2>{t['free']}</h2>
      <p class="oe-pricing-price">{t['price']}</p>
      <ul>
        <li>{t['li1']}</li>
        <li>{t['li2']}</li>
        <li>{t['li3']}</li>
        <li>{t['li4']}</li>
      </ul>
      <a href="index.html" class="oe-pricing-cta oe-pricing-cta--ghost">{t['start']}</a>
    </article>
    <article class="oe-pricing-card oe-pricing-card--pro">
      <span class="oe-pricing-badge">{t['badge']}</span>
      <h2>{t['pro']}</h2>
      <p class="oe-pricing-price" id="proPriceLabel">{t['proprice']}</p>
      <ul>
        <li><span id="proQuotaLabel">200</span> {t['liPro1']}</li>
        <li>{t['liPro2']}</li>
        <li>{t['liPro3']}</li>
        <li>{t['liPro4']}</li>
      </ul>
      <button type="button" id="btnCheckoutPro" class="oe-pricing-cta oe-pricing-cta--primary">{t['subscribe']}</button>
      <p class="oe-pricing-note" id="billingDisabledNote" style="margin-top:12px;font-size:.85rem"></p>
    </article>
  </div>
  <section class="oe-pricing-free-panel" data-pricing-free-only hidden>
    <article class="oe-pricing-card">
      <h2>{t['included']}</h2>
      <ul>
        <li>{t['liFree1']}</li>
        <li>{t['liFree2']}</li>
        <li>{t['li3']}</li>
        <li>{t['li4']}</li>
      </ul>
      <a href="index.html" class="oe-pricing-cta oe-pricing-cta--primary">{t['start']}</a>
    </article>
  </section>
  <p class="oe-pricing-note" id="pricingFootnote" data-pricing-only>{t['footnote']}</p>
</main>
<div id="footer"></div>
<script>
  OuviescreviUI.loadHeader();
  OuviescreviUI.loadFooter();
  (async function () {{
    await OuviescreviAPI.init();
    if (OuviescreviPricingVisibility) await OuviescreviPricingVisibility.whenReady();
    var params = new URLSearchParams(location.search);
    if (params.get("ok") === "1") {{
      OuviescreviUI.toast("{t['toastOk']}", "success");
    }}
    if (OuviescreviPricing && OuviescreviPricing.hidden) return;
    var res = await fetch(OuviescreviAPI.getBase() + "/api/billing/status");
    var data = res.ok ? await res.json() : {{}};
    if (data.price_label) document.getElementById("proPriceLabel").textContent = data.price_label;
    if (data.pro_quota_daily) document.getElementById("proQuotaLabel").textContent = data.pro_quota_daily;
    var note = document.getElementById("billingDisabledNote");
    var btn = document.getElementById("btnCheckoutPro");
    if (!btn) return;
    if (!data.enabled || !data.checkout_ready) {{
      note.textContent = "{t['checkoutSoon']}";
      btn.textContent = "{t['checkoutBtn']}";
      btn.addEventListener("click", function () {{
        if (OuviescreviAuth) OuviescreviAuth.openModal("register");
      }});
    }} else {{
      btn.addEventListener("click", async function () {{
        try {{
          var cres = await fetch(OuviescreviAPI.getBase() + "/api/billing/checkout", {{
            method: "POST",
            headers: OuviescreviAPI.authHeaders({{ "Content-Type": "application/json" }}),
            body: JSON.stringify({{}}),
          }});
          var checkout = await cres.json();
          if (!cres.ok) throw new Error(checkout.detail || "Unavailable");
          if (checkout.url) location.href = checkout.url;
        }} catch (e) {{
          OuviescreviUI.toast(e.message || "{t['loginSub']}", "error");
          if (OuviescreviAuth) OuviescreviAuth.openModal("login");
        }}
      }});
    }}
  }})();
</script>
</body>
</html>
"""


def landing_html(loc: str, slug: str, data: dict) -> str:
    body_class = data.get("body_class", "oe-page--landing")
    main_tag = "oe-content" if body_class == "oe-page--landing-pro" else "oe-lp-page"
    return f"""<!DOCTYPE html>
<html lang="{loc}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="{data['desc']}">
  <meta name="robots" content="index, follow">
  <title>{data['title']}</title>
  <link rel="icon" href="/logos/ouviescrevi-icon-pro.png" type="image/png">
  <link rel="stylesheet" href="../css/ouviescrevi.css?v=16">
  <link rel="stylesheet" href="../css/landing-pages.css">
  <script src="/js/ouviescrevi-seo.js" defer></script>
  <script src="../js/ouviescrevi-ui.js" defer></script>
</head>
<body class="oe-page oe-pro {body_class}">
<a class="oe-skip" href="#conteudo">{SKIP[loc]}</a>
<div id="header"></div>
<main class="{main_tag}" id="conteudo">
  {data['main']}
</main>
<div id="footer"></div>
<script>OuviescreviUI.loadHeader(); OuviescreviUI.loadFooter();</script>
</body>
</html>
"""


def main() -> None:
    created = 0
    for loc, t in PRICING.items():
        path = FRONTEND / loc / "precos.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pricing_html(loc, t), encoding="utf-8")
        created += 1
        print(f"Wrote {path.relative_to(ROOT)}")

    for slug, locales in LANDINGS.items():
        for loc, data in locales.items():
            path = FRONTEND / loc / f"{slug}.html"
            path.write_text(landing_html(loc, slug, data), encoding="utf-8")
            created += 1
            print(f"Wrote {path.relative_to(ROOT)}")

    print(f"Done — {created} files.")


if __name__ == "__main__":
    main()
