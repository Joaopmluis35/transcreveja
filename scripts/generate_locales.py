#!/usr/bin/env python3
"""Gera frontend/es, fr, de a partir de frontend/en com traduções."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "frontend" / "en"
PT_INDEX = ROOT / "frontend" / "index.html"

LANG_MENU = """            <button type="button" data-lang="pt" role="menuitem"><img src="/icons/pt.png?v=2" alt=""> Português</button>
            <button type="button" data-lang="en" role="menuitem"><img src="/icons/en.png?v=2" alt=""> English</button>
            <button type="button" data-lang="es" role="menuitem"><img src="/icons/es.png?v=2" alt=""> Español</button>
            <button type="button" data-lang="fr" role="menuitem"><img src="/icons/fr.png?v=2" alt=""> Français</button>
            <button type="button" data-lang="de" role="menuitem"><img src="/icons/de.png?v=2" alt=""> Deutsch</button>"""

LOCALES = {
    "es": {
        "html_lang": "es",
        "flag": "es",
        "open_menu": "Abrir menú",
        "close_menu": "Cerrar menú",
        "nav_main": "Principal",
        "nav_tools": "Herramientas",
        "tools_menu": [
            ("Resumir PDF / Word", "resumo.html", "resumo"),
            ("Resumen por URL", "url-resumo.html", "url-resumo"),
            ("Preguntas con IA", "perguntas.html", "perguntas"),
            ("Conversor de archivos", "conversor.html", "conversor"),
        ],
        "help": "Ayuda",
        "suggestions": "Sugerencias",
        "lang_label": "Idioma",
        "cta": "Transcribir gratis",
        "footer_tagline": "Transcribe, resume y traduce con IA — gratis y hecho en Portugal.",
        "footer_tools": "Herramientas",
        "footer_legal": "Legal",
        "footer_summarize": "Resumir",
        "footer_url": "Resumen URL",
        "footer_questions": "Preguntas",
        "footer_converter": "Conversor",
        "footer_privacy": "Privacidad",
        "footer_terms": "Términos",
        "footer_cookies": "Cookies",
        "footer_help": "Ayuda",
        "footer_suggestions": "Sugerencias",
        "footer_made": "Hecho en Portugal",
        "skip": "Saltar al contenido",
        "back_home": "← Volver al inicio",
        "replacements": [
            ("lang=\"en\"", "lang=\"es\""),
            ("Open menu", "Abrir menú"),
            ("Close menu", "Cerrar menú"),
            ("Main", "Principal"),
            ("Tools", "Herramientas"),
            ("Summarize PDF / Word", "Resumir PDF / Word"),
            ("URL Summary", "Resumen por URL"),
            ("AI Questions", "Preguntas con IA"),
            ("File Converter", "Conversor de archivos"),
            ("Help", "Ayuda"),
            ("Suggestions", "Sugerencias"),
            ("Language", "Idioma"),
            ("Transcribe free", "Transcribir gratis"),
            ("Skip to content", "Saltar al contenido"),
            ("Back to Home", "Volver al inicio"),
            ("Help & Support", "Ayuda y soporte"),
            ("Help & FAQ", "Ayuda y preguntas frecuentes"),
            ("Privacy Policy", "Política de privacidad"),
            ("Terms of Use", "Términos de uso"),
            ("Cookie Policy", "Política de cookies"),
            ("Suggestions & Feedback", "Sugerencias y comentarios"),
            ("Summarize", "Resumir"),
            ("Converter", "Conversor"),
            ("Questions", "Preguntas"),
            ("Made in Portugal", "Hecho en Portugal"),
            ("Transcribe, summarize and translate with AI — free and made in Portugal.", "Transcribe, resume y traduce con IA — gratis y hecho en Portugal."),
            ('alt="EN"', 'alt="ES"'),
            ('src="/icons/en.png"', 'src="/icons/es.png"'),
            ("data-cms-key=\"en_", "data-cms-key=\"es_"),
        ],
        "index_from_pt": True,
        "index_strings": [
            ("Saltar para o conteúdo", "Saltar al contenido"),
            ("é o teu assistente com IA para", "es tu asistente con IA para"),
            ("transcrever", "transcribir"),
            ("traduzir", "traducir"),
            ("resumir", "resumir"),
            (" e converter", " y convertir"),
            (" e ", " y "),
            ("converter ficheiros", "convertir archivos"),
            ("simples, rápido e gratuito", "simple, rápido y gratuito"),
            ("Arrasta o ficheiro aqui ou clica para escolher", "Arrastra el archivo aquí o haz clic para elegir"),
            ("— simples, rápido e gratuito.", "— simple, rápido y gratuito."),
            ("Nenhum ficheiro selecionado", "Ningún archivo seleccionado"),
            ("Limite: 500 MB — para ficheiros maiores, extrai só o áudio.", "Límite: 500 MB — para archivos más grandes, extrae solo el audio."),
            ("Transcrever", "Transcribir"),
            ("Legendar Vídeo (SRT + MP4)", "Subtitular vídeo (SRT + MP4)"),
            ("Gravar Áudio", "Grabar audio"),
            ("Parar Gravação", "Detener grabación"),
            ("Limpar Tudo", "Limpiar todo"),
            ("Identificar falas com nomes de locutores", "Identificar voces con nombres de hablantes"),
            ("Idioma do áudio", "Idioma del audio"),
            ("Deteção automática", "Detección automática"),
            ("Português", "Portugués"),
            ("Inglês", "Inglés"),
            ("Espanhol", "Español"),
            ("Francês", "Francés"),
            ("Alemão", "Alemán"),
            ("A preparar...", "Preparando..."),
            ("Transcrição:", "Transcripción:"),
            ("Copiar Tudo", "Copiar todo"),
            ("Tentar novamente", "Intentar de nuevo"),
            ("A traduzir... por favor aguarde", "Traduciendo... por favor espera"),
            ("Resumo:", "Resumen:"),
            ("Minuta da Reunião:", "Acta de la reunión:"),
            ("Tradução:", "Traducción:"),
            ("Tipo de Conteúdo:", "Tipo de contenido:"),
            ("O que é o Ouviescrevi?", "¿Qué es Ouviescrevi?"),
            ("A tua sugestão", "Tu sugerencia"),
            ("Escreve a tua sugestão...", "Escribe tu sugerencia..."),
            ("Enviar", "Enviar"),
            ("Fechar", "Cerrar"),
            ("Compreendi", "Entendido"),
            ("Política de Cookies", "Política de cookies"),
            ("Transcrição de Áudio e Vídeo com IA Grátis", "Transcripción de audio y vídeo con IA gratis"),
            ("Transcreve áudio e vídeo online", "Transcribe audio y vídeo online"),
        ],
        "index_title": "Ouviescrevi — Transcripción de audio y vídeo con IA gratis",
        "index_desc": "Transcribe audio y vídeo online con inteligencia artificial, gratis y sin registro. Resúmenes, traducción, subtítulos SRT y conversión de archivos.",
    },
    "fr": {
        "html_lang": "fr",
        "flag": "fr",
        "open_menu": "Ouvrir le menu",
        "close_menu": "Fermer le menu",
        "nav_main": "Principal",
        "nav_tools": "Outils",
        "tools_menu": [
            ("Résumer PDF / Word", "resumo.html", "resumo"),
            ("Résumé par URL", "url-resumo.html", "url-resumo"),
            ("Questions IA", "perguntas.html", "perguntas"),
            ("Convertisseur de fichiers", "conversor.html", "conversor"),
        ],
        "help": "Aide",
        "suggestions": "Suggestions",
        "lang_label": "Langue",
        "cta": "Transcrire gratuitement",
        "footer_tagline": "Transcrivez, résumez et traduisez avec l'IA — gratuit et fait au Portugal.",
        "footer_tools": "Outils",
        "footer_legal": "Mentions légales",
        "footer_summarize": "Résumer",
        "footer_url": "Résumé URL",
        "footer_questions": "Questions",
        "footer_converter": "Convertisseur",
        "footer_privacy": "Confidentialité",
        "footer_terms": "Conditions",
        "footer_cookies": "Cookies",
        "footer_help": "Aide",
        "footer_suggestions": "Suggestions",
        "footer_made": "Fait au Portugal",
        "skip": "Aller au contenu",
        "back_home": "← Retour à l'accueil",
        "replacements": [
            ("lang=\"en\"", "lang=\"fr\""),
            ("Open menu", "Ouvrir le menu"),
            ("Close menu", "Fermer le menu"),
            ("Main", "Principal"),
            ("Tools", "Outils"),
            ("Summarize PDF / Word", "Résumer PDF / Word"),
            ("URL Summary", "Résumé par URL"),
            ("AI Questions", "Questions IA"),
            ("File Converter", "Convertisseur de fichiers"),
            ("Help", "Aide"),
            ("Suggestions", "Suggestions"),
            ("Language", "Langue"),
            ("Transcribe free", "Transcrire gratuitement"),
            ("Skip to content", "Aller au contenu"),
            ("Back to Home", "Retour à l'accueil"),
            ("Help & Support", "Aide et support"),
            ("Help & FAQ", "Aide et FAQ"),
            ("Privacy Policy", "Politique de confidentialité"),
            ("Terms of Use", "Conditions d'utilisation"),
            ("Cookie Policy", "Politique de cookies"),
            ("Suggestions & Feedback", "Suggestions et avis"),
            ("Summarize", "Résumer"),
            ("Converter", "Convertisseur"),
            ("Questions", "Questions"),
            ("Made in Portugal", "Fait au Portugal"),
            ("Transcribe, summarize and translate with AI — free and made in Portugal.", "Transcrivez, résumez et traduisez avec l'IA — gratuit et fait au Portugal."),
            ('alt="EN"', 'alt="FR"'),
            ('src="/icons/en.png"', 'src="/icons/fr.png"'),
            ("data-cms-key=\"en_", "data-cms-key=\"fr_"),
        ],
        "index_from_pt": True,
        "index_strings": [
            ("Saltar para o conteúdo", "Aller au contenu"),
            ("é o teu assistente com IA para", "est votre assistant IA pour"),
            ("transcrever", "transcrire"),
            ("traduzir", "traduire"),
            ("resumir", "résumer"),
            (" e converter", " et convertir"),
            (" e ", " et "),
            ("converter ficheiros", "convertir des fichiers"),
            ("simples, rápido e gratuito", "simple, rapide et gratuit"),
            ("Arrasta o ficheiro aqui ou clica para escolher", "Glissez le fichier ici ou cliquez pour choisir"),
            ("— simples, rápido e gratuito.", "— simple, rapide et gratuit."),
            ("Nenhum ficheiro selecionado", "Aucun fichier sélectionné"),
            ("Limite: 500 MB — para ficheiros maiores, extrai só o áudio.", "Limite : 500 Mo — pour les fichiers plus volumineux, extrayez l'audio uniquement."),
            ("Transcrever", "Transcrire"),
            ("Legendar Vídeo (SRT + MP4)", "Sous-titrer la vidéo (SRT + MP4)"),
            ("Gravar Áudio", "Enregistrer l'audio"),
            ("Parar Gravação", "Arrêter l'enregistrement"),
            ("Limpar Tudo", "Tout effacer"),
            ("Identificar falas com nomes de locutores", "Identifier les voix par nom de locuteur"),
            ("Idioma do áudio", "Langue de l'audio"),
            ("Deteção automática", "Détection automatique"),
            ("Português", "Portugais"),
            ("Inglês", "Anglais"),
            ("Espanhol", "Espagnol"),
            ("Francês", "Français"),
            ("Alemão", "Allemand"),
            ("A preparar...", "Préparation..."),
            ("Transcrição:", "Transcription :"),
            ("Copiar Tudo", "Tout copier"),
            ("Tentar novamente", "Réessayer"),
            ("A traduzir... por favor aguarde", "Traduction en cours... veuillez patienter"),
            ("Resumo:", "Résumé :"),
            ("Minuta da Reunião:", "Procès-verbal :"),
            ("Tradução:", "Traduction :"),
            ("Tipo de Conteúdo:", "Type de contenu :"),
            ("O que é o Ouviescrevi?", "Qu'est-ce qu'Ouviescrevi ?"),
            ("A tua sugestão", "Votre suggestion"),
            ("Escreve a tua sugestão...", "Écrivez votre suggestion..."),
            ("Enviar", "Envoyer"),
            ("Fechar", "Fermer"),
            ("Compreendi", "Compris"),
            ("Política de Cookies", "Politique de cookies"),
            ("Transcrição de Áudio e Vídeo com IA Grátis", "Transcription audio et vidéo IA gratuite"),
            ("Transcreve áudio e vídeo online", "Transcrivez audio et vidéo en ligne"),
        ],
        "index_title": "Ouviescrevi — Transcription audio et vidéo IA gratuite",
        "index_desc": "Transcrivez audio et vidéo en ligne avec l'intelligence artificielle, gratuitement et sans inscription. Résumés, traduction, sous-titres SRT et conversion de fichiers.",
    },
    "de": {
        "html_lang": "de",
        "flag": "de",
        "open_menu": "Menü öffnen",
        "close_menu": "Menü schließen",
        "nav_main": "Hauptmenü",
        "nav_tools": "Werkzeuge",
        "tools_menu": [
            ("PDF / Word zusammenfassen", "resumo.html", "resumo"),
            ("URL-Zusammenfassung", "url-resumo.html", "url-resumo"),
            ("KI-Fragen", "perguntas.html", "perguntas"),
            ("Dateikonverter", "conversor.html", "conversor"),
        ],
        "help": "Hilfe",
        "suggestions": "Vorschläge",
        "lang_label": "Sprache",
        "cta": "Kostenlos transkribieren",
        "footer_tagline": "Transkribieren, zusammenfassen und übersetzen mit KI — kostenlos und aus Portugal.",
        "footer_tools": "Werkzeuge",
        "footer_legal": "Rechtliches",
        "footer_summarize": "Zusammenfassen",
        "footer_url": "URL-Zusammenfassung",
        "footer_questions": "Fragen",
        "footer_converter": "Konverter",
        "footer_privacy": "Datenschutz",
        "footer_terms": "Nutzungsbedingungen",
        "footer_cookies": "Cookies",
        "footer_help": "Hilfe",
        "footer_suggestions": "Vorschläge",
        "footer_made": "Made in Portugal",
        "skip": "Zum Inhalt springen",
        "back_home": "← Zur Startseite",
        "replacements": [
            ("lang=\"en\"", "lang=\"de\""),
            ("Open menu", "Menü öffnen"),
            ("Close menu", "Menü schließen"),
            ("Main", "Hauptmenü"),
            ("Tools", "Werkzeuge"),
            ("Summarize PDF / Word", "PDF / Word zusammenfassen"),
            ("URL Summary", "URL-Zusammenfassung"),
            ("AI Questions", "KI-Fragen"),
            ("File Converter", "Dateikonverter"),
            ("Help", "Hilfe"),
            ("Suggestions", "Vorschläge"),
            ("Language", "Sprache"),
            ("Transcribe free", "Kostenlos transkribieren"),
            ("Skip to content", "Zum Inhalt springen"),
            ("Back to Home", "Zur Startseite"),
            ("Help & Support", "Hilfe & Support"),
            ("Help & FAQ", "Hilfe & FAQ"),
            ("Privacy Policy", "Datenschutzerklärung"),
            ("Terms of Use", "Nutzungsbedingungen"),
            ("Cookie Policy", "Cookie-Richtlinie"),
            ("Suggestions & Feedback", "Vorschläge & Feedback"),
            ("Summarize", "Zusammenfassen"),
            ("Converter", "Konverter"),
            ("Questions", "Fragen"),
            ("Made in Portugal", "Made in Portugal"),
            ("Transcribe, summarize and translate with AI — free and made in Portugal.", "Transkribieren, zusammenfassen und übersetzen mit KI — kostenlos und aus Portugal."),
            ('alt="EN"', 'alt="DE"'),
            ('src="/icons/en.png"', 'src="/icons/de.png"'),
            ("data-cms-key=\"en_", "data-cms-key=\"de_"),
        ],
        "index_from_pt": True,
        "index_strings": [
            ("Saltar para o conteúdo", "Zum Inhalt springen"),
            ("é o teu assistente com IA para", "ist dein KI-Assistent zum"),
            ("transcrever", "Transkribieren"),
            ("traduzir", "Übersetzen"),
            ("resumir", "Zusammenfassen"),
            (" e converter", " und konvertieren"),
            (" e ", " und "),
            ("converter ficheiros", "Dateien konvertieren"),
            ("simples, rápido e gratuito", "einfach, schnell und kostenlos"),
            ("Arrasta o ficheiro aqui ou clica para escolher", "Datei hierher ziehen oder klicken zum Auswählen"),
            ("— simples, rápido e gratuito.", "— einfach, schnell und kostenlos."),
            ("Nenhum ficheiro selecionado", "Keine Datei ausgewählt"),
            ("Limite: 500 MB — para ficheiros maiores, extrai só o áudio.", "Limit: 500 MB — bei größeren Dateien nur Audio extrahieren."),
            ("Transcrever", "Transkribieren"),
            ("Legendar Vídeo (SRT + MP4)", "Video untertiteln (SRT + MP4)"),
            ("Gravar Áudio", "Audio aufnehmen"),
            ("Parar Gravação", "Aufnahme stoppen"),
            ("Limpar Tudo", "Alles löschen"),
            ("Identificar falas com nomes de locutores", "Sprecher anhand von Namen erkennen"),
            ("Idioma do áudio", "Audiosprache"),
            ("Deteção automática", "Automatische Erkennung"),
            ("Português", "Portugiesisch"),
            ("Inglês", "Englisch"),
            ("Espanhol", "Spanisch"),
            ("Francês", "Französisch"),
            ("Alemão", "Deutsch"),
            ("A preparar...", "Wird vorbereitet..."),
            ("Transcrição:", "Transkription:"),
            ("Copiar Tudo", "Alles kopieren"),
            ("Tentar novamente", "Erneut versuchen"),
            ("A traduzir... por favor aguarde", "Übersetzung läuft... bitte warten"),
            ("Resumo:", "Zusammenfassung:"),
            ("Minuta da Reunião:", "Sitzungsprotokoll:"),
            ("Tradução:", "Übersetzung:"),
            ("Tipo de Conteúdo:", "Inhaltstyp:"),
            ("O que é o Ouviescrevi?", "Was ist Ouviescrevi?"),
            ("A tua sugestão", "Dein Vorschlag"),
            ("Escreve a tua sugestão...", "Schreibe deinen Vorschlag..."),
            ("Enviar", "Senden"),
            ("Fechar", "Schließen"),
            ("Compreendi", "Verstanden"),
            ("Política de Cookies", "Cookie-Richtlinie"),
            ("Transcrição de Áudio e Vídeo com IA Grátis", "Kostenlose KI-Audio- und Video-Transkription"),
            ("Transcreve áudio e vídeo online", "Transkribiere Audio und Video online"),
        ],
        "index_title": "Ouviescrevi — Kostenlose KI-Audio- und Video-Transkription",
        "index_desc": "Transkribiere Audio und Video online mit KI, kostenlos und ohne Anmeldung. Zusammenfassungen, Übersetzung, SRT-Untertitel und Dateikonvertierung.",
    },
}


def apply_replacements(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        text = text.replace(old, new)
    return text


def build_header(locale: str, cfg: dict) -> str:
    tools = "\n".join(
        f'            <a href="{href}" data-nav-page="{slug}" role="menuitem">{label}</a>'
        for label, href, slug in cfg["tools_menu"]
    )
    return f"""<header class="oe-pro-header" id="oeProHeader">
  <div class="oe-pro-header__inner">
    <a class="oe-pro-brand" href="index.html">
      <span class="oe-pro-brand__icon-wrap" aria-hidden="true">
        <img src="/logos/ouviescrevi-icon-pro.png" alt="" class="oe-pro-brand__icon">
      </span>
      <span class="oe-pro-brand__name">Ouviescrevi</span>
    </a>

    <button type="button" class="oe-pro-nav__mobile-toggle" aria-expanded="false" aria-controls="oeProNavPanel" aria-label="{cfg['open_menu']}">
      <span class="oe-pro-nav__burger" aria-hidden="true"></span>
    </button>

    <div class="oe-pro-nav__panel" id="oeProNavPanel">
      <nav class="oe-pro-nav" aria-label="{cfg['nav_main']}">
        <div class="oe-pro-nav__dropdown">
          <button type="button" class="oe-pro-nav__trigger" aria-expanded="false" aria-haspopup="true">
            {cfg['nav_tools']}
            <svg class="oe-pro-nav__chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
          </button>
          <div class="oe-pro-nav__menu" role="menu">
{tools}
          </div>
        </div>

        <a class="oe-pro-nav__link" href="ajuda.html" data-nav-page="ajuda">{cfg['help']}</a>
        <a class="oe-pro-nav__link" href="sugestoes.html" data-nav-page="sugestoes">{cfg['suggestions']}</a>
      </nav>

      <div class="oe-pro-header__actions">
        <div class="oe-pro-lang">
          <button type="button" class="oe-pro-lang__btn" id="oeLangBtn" aria-expanded="false" aria-haspopup="true" aria-label="{cfg['lang_label']}">
            <img src="/icons/{cfg['flag']}.png?v=2" alt="{cfg['flag'].upper()}" width="22" height="16">
          </button>
          <div class="oe-pro-lang__menu" id="oeLangMenu" role="menu">
{LANG_MENU}
          </div>
        </div>
        <a class="oe-pro-btn oe-pro-btn--primary oe-pro-nav__cta" href="index.html">{cfg['cta']}</a>
      </div>
    </div>
  </div>
  <div class="oe-pro-nav__backdrop" aria-hidden="true"></div>
</header>
"""


def build_footer(locale: str, cfg: dict) -> str:
  en_footer = (EN / "footer.html").read_text(encoding="utf-8")
  footer = apply_replacements(en_footer, cfg["replacements"])
  footer = re.sub(
      r'<div class="oe-pro-lang__menu" id="oeLangMenu" role="menu">.*?</div>',
      "",
      footer,
      flags=re.S,
  )
  return footer


def build_index(locale: str, cfg: dict) -> str:
    text = PT_INDEX.read_text(encoding="utf-8")
    text = text.replace('lang="pt"', f'lang="{cfg["html_lang"]}"')
    text = text.replace('href="css/', 'href="../css/')
    text = text.replace('href="js/', 'href="../js/')
    text = text.replace('src="js/', 'src="../js/')
    for old, new in cfg["index_strings"]:
        text = text.replace(old, new)
    text = re.sub(
        r"<title>.*?</title>",
        f"<title>{cfg['index_title']}</title>",
        text,
        count=1,
    )
    text = re.sub(
        r'<meta name="description" content="[^"]*"',
        f'<meta name="description" content="{cfg["index_desc"]}"',
        text,
        count=1,
    )
    text = text.replace('data-cms-key="home_', f'data-cms-key="{locale}_home_')
    return text


def main() -> None:
    if not EN.is_dir():
        raise SystemExit(f"Missing {EN}")

    for locale, cfg in LOCALES.items():
        out = ROOT / "frontend" / locale
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(EN, out)

        for html in out.glob("*.html"):
            if html.name == "index.html":
                continue
            content = html.read_text(encoding="utf-8")
            content = apply_replacements(content, cfg["replacements"])
            html.write_text(content, encoding="utf-8")

        (out / "header.html").write_text(build_header(locale, cfg), encoding="utf-8")
        (out / "footer.html").write_text(
            apply_replacements((EN / "footer.html").read_text(encoding="utf-8"), cfg["replacements"]),
            encoding="utf-8",
        )
        (out / "index.html").write_text(build_index(locale, cfg), encoding="utf-8")
        print(f"Generated frontend/{locale}/ ({len(list(out.glob('*.html')))} pages)")


if __name__ == "__main__":
    main()
