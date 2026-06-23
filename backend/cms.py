"""Conteúdo editável do site (CMS multi-página)."""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from typing import Any

_AJUDA_FAQ_PT = """<h2>1. O que é o Ouviescrevi?</h2>
<p>É uma ferramenta automática que converte ficheiros de áudio, vídeo ou texto em transcrições, resumos, traduções e muito mais, usando inteligência artificial.</p>
<h2>2. Que formatos de ficheiros são suportados?</h2>
<p>Atualmente suportamos <strong>.mp3, .mp4, .wav, .m4a, .mov</strong> e outros formatos comuns.</p>
<h2>3. É necessário criar conta?</h2>
<p>Não. O Ouviescrevi está disponível gratuitamente e sem necessidade de registo.</p>
<h2>4. Que funcionalidades posso usar?</h2>
<ul>
<li>🎧 Transcrição automática de áudio e vídeo</li>
<li>🧠 Geração de resumo com IA (inclui estilo formal, simples, em tópicos ou minuta)</li>
<li>🌍 Tradução de textos para vários idiomas</li>
<li>✍️ Correção automática de ortografia e gramática</li>
<li>📧 Geração automática de e-mails a partir de texto</li>
<li>❓ Criação de perguntas de escolha múltipla com base em conteúdo</li>
<li>🗂️ Classificação automática do tipo de conteúdo (ex: aula, reunião, podcast)</li>
</ul>
<h2>5. A transcrição é 100% precisa?</h2>
<p>A precisão depende da qualidade do áudio e da clareza da fala. Utilizamos o modelo <strong>Whisper da OpenAI</strong> para garantir alta qualidade.</p>
<h2>6. Os meus ficheiros são guardados?</h2>
<p>Os ficheiros de áudio/vídeo são eliminados após o processamento. Registamos apenas o nome do ficheiro e a data para estatísticas. Consulta a <a href="privacidade.html">Política de Privacidade</a> para detalhes completos.</p>
<h2>7. Como posso usar as funcionalidades de IA?</h2>
<p>Basta carregar o ficheiro e, após a transcrição, podes clicar em "Gerar resumo", "Traduzir", "Corrigir", "Classificar" ou "Gerar email". Tudo acontece automaticamente no browser.</p>"""

_AJUDA_FAQ_EN = """<h2>1. What is Ouviescrevi?</h2>
<p>It's a fully automated tool that converts audio, video, or text files into transcriptions, summaries, translations, and more using artificial intelligence.</p>
<h2>2. Which file formats are supported?</h2>
<p>We currently support <strong>.mp3, .mp4, .wav, .m4a, .mov</strong> and other common formats.</p>
<h2>3. Do I need to create an account?</h2>
<p>No. Ouviescrevi is completely free and requires no registration.</p>
<h2>4. What features are available?</h2>
<ul>
<li>🎧 Automatic transcription of audio and video</li>
<li>🧠 AI-generated summaries (formal, simple, bullet points, or meeting minutes)</li>
<li>🌍 Text translation into multiple languages</li>
<li>✍️ Grammar and spelling correction</li>
<li>📧 Email generation from text or transcripts</li>
<li>❓ Multiple-choice question generation</li>
<li>🗂️ Automatic content classification (e.g. class, meeting, podcast)</li>
</ul>
<h2>5. Is the transcription 100% accurate?</h2>
<p>Accuracy depends on audio quality and clarity of speech. We use OpenAI's Whisper model for high-quality results.</p>
<h2>6. Are my files stored?</h2>
<p>No. Files are processed temporarily and deleted after transcription. Only file name and timestamp are stored for statistics.</p>
<h2>7. How do I use the AI features?</h2>
<p>After uploading a file, click "Generate summary", "Translate", "Correct", "Classify", or "Generate Email". Everything runs in your browser.</p>"""

_CONVERSOR_SEO_PT = """<h2>🔄 Conversor Inteligente de Ficheiros</h2>
<p>O <strong>Ouviescrevi</strong> também funciona como um <strong>conversor gratuito de ficheiros online</strong>, rápido e simples de usar. Converte ficheiros entre <strong>Word (.docx)</strong>, <strong>PDF</strong>, <strong>TXT</strong>, <strong>JSON</strong>, <strong>SRT</strong> e até imagens em texto com apoio de Inteligência Artificial.</p>
<p>Ideal para quem precisa de adaptar formatos para envio por email, impressão, edição ou partilha. Tudo direto no navegador, sem instalação.</p>
<p>Basta colar ou carregar o conteúdo, escolher o formato de saída e fazer download com um clique.</p>
<p><strong>Formatos suportados:</strong></p>
<ul>
<li>📄 Word (.docx) para PDF</li>
<li>📃 PDF para texto editável (.txt)</li>
<li>🖼️ Imagens para texto com OCR</li>
<li>📑 Texto para subtítulos (.srt)</li>
<li>🔢 Texto para ficheiro JSON estruturado</li>
</ul>
<p>Um dos conversores mais completos e acessíveis de Portugal. 100% online, gratuito e sem complicações 🇵🇹</p>"""

_CONVERSOR_SEO_EN = """<h2>🔄 Smart File Converter</h2>
<p><strong>Ouviescrevi</strong> also offers a <strong>free online file converter</strong> that is quick and easy to use. Convert between <strong>Word (.docx)</strong>, <strong>PDF</strong>, <strong>TXT</strong>, <strong>JSON</strong>, <strong>SRT</strong>, and even images to text with the help of AI.</p>
<p>Ideal for anyone needing to adapt file formats for email, printing, editing or sharing — directly in your browser, no installation required.</p>
<p>Simply paste or upload your content, choose the output format, and download with one click.</p>
<p><strong>Supported formats:</strong></p>
<ul>
<li>📄 Word (.docx) to PDF</li>
<li>📃 PDF to editable text (.txt)</li>
<li>🖼️ Images to text with OCR</li>
<li>📑 Text to subtitles (.srt)</li>
<li>🔢 Text to structured JSON</li>
</ul>
<p>One of the most complete and accessible converters in Portugal. 100% online, free and hassle-free 🇵🇹</p>"""

PAGE_SCHEMA: list[dict[str, Any]] = [
    {
        "id": "home",
        "label": "Homepage",
        "lang": "pt",
        "path": "/index.html",
        "fields": [
            {"key": "home_intro_html", "label": "Texto de boas-vindas (topo)", "type": "rich"},
            {"key": "seo_title", "label": "Título «O que é o Ouviescrevi?»", "type": "text"},
            {"key": "seo_p1", "label": "Parágrafo 1", "type": "rich"},
            {"key": "seo_p2", "label": "Parágrafo 2", "type": "rich"},
            {"key": "seo_features", "label": "Funcionalidades (uma por linha)", "type": "lines"},
            {"key": "seo_closing", "label": "Texto de fecho", "type": "text"},
        ],
    },
    {
        "id": "ajuda",
        "label": "Ajuda",
        "lang": "pt",
        "path": "/ajuda.html",
        "fields": [
            {"key": "ajuda_title", "label": "Título da página", "type": "text"},
            {"key": "ajuda_intro", "label": "Introdução", "type": "rich"},
            {"key": "ajuda_faq", "label": "Perguntas frequentes", "type": "rich"},
            {"key": "ajuda_contact", "label": "Secção de contacto", "type": "rich"},
        ],
    },
    {
        "id": "conversor",
        "label": "Conversor",
        "lang": "pt",
        "path": "/conversor.html",
        "fields": [
            {"key": "conversor_title", "label": "Título", "type": "text"},
            {"key": "conversor_lead", "label": "Subtítulo", "type": "text"},
            {"key": "conversor_notice", "label": "Aviso", "type": "rich"},
            {"key": "conversor_seo", "label": "Texto SEO (rodapé)", "type": "rich"},
        ],
    },
    {
        "id": "sugestoes",
        "label": "Sugestões",
        "lang": "pt",
        "path": "/sugestoes.html",
        "fields": [
            {"key": "sugestoes_title", "label": "Título", "type": "text"},
            {"key": "sugestoes_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "ajuda_en",
        "label": "Help (EN)",
        "lang": "en",
        "path": "/en/ajuda.html",
        "fields": [
            {"key": "en_ajuda_title", "label": "Page title", "type": "text"},
            {"key": "en_ajuda_intro", "label": "Introduction", "type": "rich"},
            {"key": "en_ajuda_faq", "label": "FAQ", "type": "rich"},
            {"key": "en_ajuda_contact", "label": "Contact section", "type": "rich"},
        ],
    },
    {
        "id": "conversor_en",
        "label": "Converter (EN)",
        "lang": "en",
        "path": "/en/conversor.html",
        "fields": [
            {"key": "en_conversor_title", "label": "Title", "type": "text"},
            {"key": "en_conversor_lead", "label": "Subtitle", "type": "text"},
            {"key": "en_conversor_notice", "label": "Notice", "type": "rich"},
            {"key": "en_conversor_seo", "label": "SEO text (footer)", "type": "rich"},
        ],
    },
    {
        "id": "sugestoes_en",
        "label": "Suggestions (EN)",
        "lang": "en",
        "path": "/en/sugestoes.html",
        "fields": [
            {"key": "en_sugestoes_title", "label": "Title", "type": "text"},
            {"key": "en_sugestoes_lead", "label": "Subtitle", "type": "text"},
        ],
    },
]

DEFAULT_SITE_CONTENT: dict[str, str] = {
    "home_intro_html": (
        "<p><strong>🧠 Ouviescrevi</strong> é o teu assistente com IA para<br>"
        "<strong>transcrever</strong> 🎙️, <strong>traduzir</strong> 🌍, <strong>resumir</strong> 📌 "
        "e <strong>converter ficheiros</strong> 📄<br>— simples, rápido e gratuito.</p>"
    ),
    "seo_title": "🧠 O que é o Ouviescrevi?",
    "seo_p1": (
        "<p>Ouviescrevi é uma ferramenta online que usa Inteligência Artificial (IA) para "
        "<strong>transcrever, resumir, traduzir, corrigir</strong> e <strong>classificar conteúdos</strong>, "
        "além de <strong>gerar emails</strong>, <strong>criar perguntas de estudo</strong> e até "
        "<strong>produzir vídeos com legendas automáticas</strong>.</p>"
    ),
    "seo_p2": (
        "<p>Ideal para professores, jornalistas, estudantes, empresas e qualquer pessoa que precise "
        "transformar áudio, vídeo ou texto em conhecimento útil. 🚀</p>"
    ),
    "seo_features": (
        "🎙️ Transcrição de áudio/vídeo\n"
        "🧠 Resumos automáticos\n"
        "🌍 Tradução de textos\n"
        "✍️ Correção e emails com IA\n"
        "📚 Perguntas de estudo\n"
        "💬 Vídeos com legendas automáticas\n"
        "📁 Conversão de ficheiros"
    ),
    "seo_closing": "Começa já gratuitamente. Simples, rápido e feito em Portugal 🇵🇹",
    "ajuda_title": "Ajuda e Suporte",
    "ajuda_intro": (
        "<p>Bem-vindo à página de ajuda do <strong>Ouviescrevi</strong>. "
        "Aqui encontras respostas às perguntas mais frequentes:</p>"
    ),
    "ajuda_faq": _AJUDA_FAQ_PT,
    "ajuda_contact": (
        "<h2>📩 Contacto de Suporte</h2>"
        "<p>Se precisares de ajuda adicional, envia um email para: "
        "<strong>ouviescrevi@gmail.com</strong></p>"
    ),
    "conversor_title": "📁 Conversor de Ficheiros",
    "conversor_lead": "Converte Word, PDF e imagens — gratuito e no browser.",
    "conversor_notice": "<p>💡 Aproveita esta funcionalidade gratuita enquanto a versão com IA evolui.</p>",
    "conversor_seo": _CONVERSOR_SEO_PT,
    "sugestoes_title": "💡 Sugestões",
    "sugestoes_lead": "A tua ideia ajuda-nos a melhorar o Ouviescrevi.",
    "en_ajuda_title": "Help & Support",
    "en_ajuda_intro": (
        "<p>Welcome to the <strong>Ouviescrevi</strong> help page. "
        "Here are the most frequently asked questions:</p>"
    ),
    "en_ajuda_faq": _AJUDA_FAQ_EN,
    "en_ajuda_contact": (
        "<h2>📩 Support Contact</h2>"
        "<p>If you need additional help, email us at: "
        "<strong>ouviescrevi@gmail.com</strong></p>"
    ),
    "en_conversor_title": "📁 File Converter",
    "en_conversor_lead": "Convert Word, PDF and images — free in your browser.",
    "en_conversor_notice": "<p>💡 Use this free tool while our AI-based version evolves.</p>",
    "en_conversor_seo": _CONVERSOR_SEO_EN,
    "en_sugestoes_title": "💡 Suggestions",
    "en_sugestoes_lead": "Your feedback helps us improve Ouviescrevi.",
}

CONTENT_KEYS = frozenset(DEFAULT_SITE_CONTENT.keys())
_PAGE_KEYS: dict[str, list[str]] = {
    page["id"]: [f["key"] for f in page["fields"]] for page in PAGE_SCHEMA
}


def get_page_schema() -> list[dict[str, Any]]:
    return PAGE_SCHEMA


def keys_for_page(page_id: str) -> list[str]:
    return list(_PAGE_KEYS.get(page_id, []))


def _db_path() -> str:
    return "ouviescrevi.db"


def _sanitize_html(value: str) -> str:
    cleaned = re.sub(
        r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"on\w+\s*=", "", cleaned, flags=re.IGNORECASE)


def _normalize_value(key: str, value: str) -> str:
    text = str(value)
    field_type = "text"
    for page in PAGE_SCHEMA:
        for field in page["fields"]:
            if field["key"] == key:
                field_type = field["type"]
                break
    if field_type == "rich":
        return _sanitize_html(text)
    return text


def get_all_content() -> dict[str, str]:
    out = dict(DEFAULT_SITE_CONTENT)
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM site_content")
        for key, value in cur.fetchall():
            if key in CONTENT_KEYS and value is not None:
                out[key] = value
    finally:
        conn.close()
    return out


def update_content(updates: dict[str, str]) -> dict[str, str]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        for key, value in updates.items():
            if key not in CONTENT_KEYS:
                continue
            cur.execute(
                """
                INSERT INTO site_content (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, _normalize_value(key, value), now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_all_content()


def reset_content(keys: list[str] | None = None) -> dict[str, str]:
    to_drop = keys if keys else list(CONTENT_KEYS)
    conn = sqlite3.connect(_db_path())
    try:
        cur = conn.cursor()
        for key in to_drop:
            if key in CONTENT_KEYS:
                cur.execute("DELETE FROM site_content WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()
    return get_all_content()
