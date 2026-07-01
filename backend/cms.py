"""Conteúdo editável do site (CMS multi-página)."""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from typing import Any

from database import get_connection

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
    {
        "id": "home_en",
        "label": "Homepage (EN)",
        "lang": "en",
        "path": "/en/index.html",
        "fields": [
            {"key": "en_home_intro_html", "label": "Welcome text (top)", "type": "rich"},
        ],
    },
    {
        "id": "resumo_en",
        "label": "Summary tool (EN)",
        "lang": "en",
        "path": "/en/resumo.html",
        "fields": [
            {"key": "en_resumo_title", "label": "Title", "type": "text"},
            {"key": "en_resumo_lead", "label": "Subtitle", "type": "text"},
        ],
    },
    {
        "id": "url_resumo_en",
        "label": "URL summary (EN)",
        "lang": "en",
        "path": "/en/url-resumo.html",
        "fields": [
            {"key": "en_url_resumo_title", "label": "Title", "type": "text"},
            {"key": "en_url_resumo_lead", "label": "Subtitle", "type": "text"},
        ],
    },
    {
        "id": "perguntas_en",
        "label": "Quiz generator (EN)",
        "lang": "en",
        "path": "/en/perguntas.html",
        "fields": [
            {"key": "en_perguntas_title", "label": "Title", "type": "text"},
            {"key": "en_perguntas_lead", "label": "Subtitle", "type": "text"},
        ],
    },
    {
        "id": "capitulos_en",
        "label": "Chapters (EN)",
        "lang": "en",
        "path": "/en/capitulos.html",
        "fields": [
            {"key": "en_capitulos_title", "label": "Title", "type": "text"},
            {"key": "en_capitulos_lead", "label": "Subtitle", "type": "text"},
        ],
    },
    {
        "id": "aulas",
        "label": "Landing — Aulas",
        "lang": "pt",
        "path": "/aulas.html",
        "fields": [
            {"key": "aulas_title", "label": "Título", "type": "text"},
            {"key": "aulas_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "professores",
        "label": "Landing — Professores",
        "lang": "pt",
        "path": "/professores.html",
        "fields": [
            {"key": "professores_title", "label": "Título", "type": "text"},
            {"key": "professores_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "jornalistas",
        "label": "Landing — Jornalistas",
        "lang": "pt",
        "path": "/jornalistas.html",
        "fields": [
            {"key": "jornalistas_title", "label": "Título", "type": "text"},
            {"key": "jornalistas_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "podcasts",
        "label": "Landing — Podcasts",
        "lang": "pt",
        "path": "/podcasts.html",
        "fields": [
            {"key": "podcasts_title", "label": "Título", "type": "text"},
            {"key": "podcasts_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "reunioes",
        "label": "Landing — Reuniões",
        "lang": "pt",
        "path": "/reunioes.html",
        "fields": [
            {"key": "reunioes_title", "label": "Título", "type": "text"},
            {"key": "reunioes_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "testemunhos",
        "label": "Landing — Testemunhos",
        "lang": "pt",
        "path": "/testemunhos.html",
        "fields": [
            {"key": "testemunhos_title", "label": "Título", "type": "text"},
            {"key": "testemunhos_body", "label": "Texto", "type": "rich"},
        ],
    },
    {
        "id": "resumo",
        "label": "Ferramenta — Resumo",
        "lang": "pt",
        "path": "/resumo.html",
        "fields": [
            {"key": "resumo_title", "label": "Título", "type": "text"},
            {"key": "resumo_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "corretor",
        "label": "Ferramenta — Corretor",
        "lang": "pt",
        "path": "/corretor.html",
        "fields": [
            {"key": "corretor_title", "label": "Título", "type": "text"},
            {"key": "corretor_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "perguntas",
        "label": "Ferramenta — Perguntas",
        "lang": "pt",
        "path": "/perguntas.html",
        "fields": [
            {"key": "perguntas_title", "label": "Título", "type": "text"},
            {"key": "perguntas_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "capitulos",
        "label": "Ferramenta — Capítulos",
        "lang": "pt",
        "path": "/capitulos.html",
        "fields": [
            {"key": "capitulos_title", "label": "Título", "type": "text"},
            {"key": "capitulos_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "url_resumo",
        "label": "Ferramenta — Resumo URL",
        "lang": "pt",
        "path": "/url-resumo.html",
        "fields": [
            {"key": "url_resumo_title", "label": "Título", "type": "text"},
            {"key": "url_resumo_lead", "label": "Subtítulo", "type": "text"},
        ],
    },
    {
        "id": "privacidade",
        "label": "Legal — Privacidade",
        "lang": "pt",
        "path": "/privacidade.html",
        "fields": [
            {"key": "privacidade_meta", "label": "Data de atualização", "type": "text"},
            {"key": "privacidade_disclaimer", "label": "Aviso introdutório", "type": "rich"},
        ],
    },
    {
        "id": "termos",
        "label": "Legal — Termos",
        "lang": "pt",
        "path": "/termos.html",
        "fields": [
            {"key": "termos_meta", "label": "Data de atualização", "type": "text"},
            {"key": "termos_intro", "label": "Aviso introdutório", "type": "rich"},
        ],
    },
    {
        "id": "cookies",
        "label": "Legal — Cookies",
        "lang": "pt",
        "path": "/cookies.html",
        "fields": [
            {"key": "cookies_meta", "label": "Data de atualização", "type": "text"},
            {"key": "cookies_intro", "label": "Introdução (secção 1)", "type": "rich"},
        ],
    },
    {
        "id": "privacidade_en",
        "label": "Legal — Privacy (EN)",
        "lang": "en",
        "path": "/en/privacy.html",
        "fields": [
            {"key": "en_privacidade_meta", "label": "Last updated line", "type": "rich"},
            {"key": "en_privacidade_disclaimer", "label": "Intro disclaimer", "type": "rich"},
        ],
    },
    {
        "id": "termos_en",
        "label": "Legal — Terms (EN)",
        "lang": "en",
        "path": "/en/terms.html",
        "fields": [
            {"key": "en_termos_meta", "label": "Last updated line", "type": "rich"},
            {"key": "en_termos_intro", "label": "Intro disclaimer", "type": "rich"},
        ],
    },
    {
        "id": "cookies_en",
        "label": "Legal — Cookies (EN)",
        "lang": "en",
        "path": "/en/cookies.html",
        "fields": [
            {"key": "en_cookies_meta", "label": "Last updated line", "type": "rich"},
            {"key": "en_cookies_intro", "label": "Introduction (section 1)", "type": "rich"},
        ],
    },
    {
        "id": "site_global",
        "label": "Site — Global",
        "lang": "pt",
        "path": "/index.html",
        "fields": [
            {"key": "maintenance_message", "label": "Mensagem de manutenção", "type": "rich"},
            {"key": "home_testimonials", "label": "Testemunhos homepage (HTML)", "type": "rich"},
        ],
    },
    {
        "id": "seo_home",
        "label": "SEO — Homepage",
        "lang": "pt",
        "path": "/index.html",
        "category": "seo",
        "fields": [
            {"key": "meta_home_title", "label": "Meta title", "type": "text"},
            {"key": "meta_home_description", "label": "Meta description", "type": "text"},
        ],
    },
    {
        "id": "seo_conversor",
        "label": "SEO — Conversor",
        "lang": "pt",
        "path": "/conversor.html",
        "category": "seo",
        "fields": [
            {"key": "meta_conversor_title", "label": "Meta title", "type": "text"},
            {"key": "meta_conversor_description", "label": "Meta description", "type": "text"},
        ],
    },
    {
        "id": "seo_resumo",
        "label": "SEO — Resumo",
        "lang": "pt",
        "path": "/resumo.html",
        "category": "seo",
        "fields": [
            {"key": "meta_resumo_title", "label": "Meta title", "type": "text"},
            {"key": "meta_resumo_description", "label": "Meta description", "type": "text"},
        ],
    },
    {
        "id": "seo_ajuda",
        "label": "SEO — Ajuda",
        "lang": "pt",
        "path": "/ajuda.html",
        "category": "seo",
        "fields": [
            {"key": "meta_ajuda_title", "label": "Meta title", "type": "text"},
            {"key": "meta_ajuda_description", "label": "Meta description", "type": "text"},
        ],
    },
]

LOCALE_CMS_LANGS = (("es", "ES"), ("fr", "FR"), ("de", "DE"))


def _locale_cms_pages(lang: str, lang_label: str) -> list[dict[str, Any]]:
    p = lang
    base = f"/{lang}"
    return [
        {
            "id": f"home_{lang}",
            "label": f"Homepage ({lang_label})",
            "lang": lang,
            "path": f"{base}/index.html",
            "fields": [
                {"key": f"{p}_home_intro_html", "label": "Texto de boas-vindas (topo)", "type": "rich"},
            ],
        },
        {
            "id": f"ajuda_{lang}",
            "label": f"Ajuda ({lang_label})",
            "lang": lang,
            "path": f"{base}/ajuda.html",
            "fields": [
                {"key": f"{p}_ajuda_title", "label": "Título da página", "type": "text"},
                {"key": f"{p}_ajuda_intro", "label": "Introdução", "type": "rich"},
                {"key": f"{p}_ajuda_faq", "label": "Perguntas frequentes", "type": "rich"},
                {"key": f"{p}_ajuda_contact", "label": "Secção de contacto", "type": "rich"},
            ],
        },
        {
            "id": f"conversor_{lang}",
            "label": f"Conversor ({lang_label})",
            "lang": lang,
            "path": f"{base}/conversor.html",
            "fields": [
                {"key": f"{p}_conversor_title", "label": "Título", "type": "text"},
                {"key": f"{p}_conversor_lead", "label": "Subtítulo", "type": "text"},
                {"key": f"{p}_conversor_notice", "label": "Aviso", "type": "rich"},
                {"key": f"{p}_conversor_seo", "label": "Texto SEO (rodapé)", "type": "rich"},
            ],
        },
        {
            "id": f"sugestoes_{lang}",
            "label": f"Sugestões ({lang_label})",
            "lang": lang,
            "path": f"{base}/sugestoes.html",
            "fields": [
                {"key": f"{p}_sugestoes_title", "label": "Título", "type": "text"},
                {"key": f"{p}_sugestoes_lead", "label": "Subtítulo", "type": "text"},
            ],
        },
        {
            "id": f"privacidade_{lang}",
            "label": f"Legal — Privacidade ({lang_label})",
            "lang": lang,
            "path": f"{base}/privacy.html",
            "fields": [
                {"key": f"{p}_privacidade_meta", "label": "Linha «última atualização»", "type": "rich"},
                {"key": f"{p}_privacidade_disclaimer", "label": "Aviso introdutório", "type": "rich"},
            ],
        },
        {
            "id": f"termos_{lang}",
            "label": f"Legal — Termos ({lang_label})",
            "lang": lang,
            "path": f"{base}/terms.html",
            "fields": [
                {"key": f"{p}_termos_meta", "label": "Linha «última atualização»", "type": "rich"},
                {"key": f"{p}_termos_intro", "label": "Aviso introdutório", "type": "rich"},
            ],
        },
        {
            "id": f"cookies_{lang}",
            "label": f"Legal — Cookies ({lang_label})",
            "lang": lang,
            "path": f"{base}/cookies.html",
            "fields": [
                {"key": f"{p}_cookies_meta", "label": "Linha «última atualização»", "type": "rich"},
                {"key": f"{p}_cookies_intro", "label": "Introdução (secção 1)", "type": "rich"},
            ],
        },
        {
            "id": f"resumo_{lang}",
            "label": f"Resumo ({lang_label})",
            "lang": lang,
            "path": f"{base}/resumo.html",
            "fields": [
                {"key": f"{p}_resumo_title", "label": "Título", "type": "text"},
                {"key": f"{p}_resumo_lead", "label": "Subtítulo", "type": "text"},
            ],
        },
        {
            "id": f"url_resumo_{lang}",
            "label": f"Resumo URL ({lang_label})",
            "lang": lang,
            "path": f"{base}/url-resumo.html",
            "fields": [
                {"key": f"{p}_url_resumo_title", "label": "Título", "type": "text"},
                {"key": f"{p}_url_resumo_lead", "label": "Subtítulo", "type": "text"},
            ],
        },
        {
            "id": f"perguntas_{lang}",
            "label": f"Perguntas ({lang_label})",
            "lang": lang,
            "path": f"{base}/perguntas.html",
            "fields": [
                {"key": f"{p}_perguntas_title", "label": "Título", "type": "text"},
                {"key": f"{p}_perguntas_lead", "label": "Subtítulo", "type": "text"},
            ],
        },
    ]


for _lc_lang, _lc_label in LOCALE_CMS_LANGS:
    PAGE_SCHEMA.extend(_locale_cms_pages(_lc_lang, _lc_label))


def _locale_seo_pages(lang: str, lang_label: str) -> list[dict[str, Any]]:
    base = f"/{lang}"
    p = lang
    return [
        {
            "id": f"seo_home_{lang}",
            "label": f"SEO — Homepage ({lang_label})",
            "lang": lang,
            "path": f"{base}/index.html",
            "category": "seo",
            "fields": [
                {"key": f"meta_home_title_{lang}", "label": "Meta title", "type": "text"},
                {"key": f"meta_home_description_{lang}", "label": "Meta description", "type": "text"},
            ],
        },
        {
            "id": f"seo_ajuda_{lang}",
            "label": f"SEO — Ajuda ({lang_label})",
            "lang": lang,
            "path": f"{base}/ajuda.html",
            "category": "seo",
            "fields": [
                {"key": f"meta_ajuda_title_{lang}", "label": "Meta title", "type": "text"},
                {"key": f"meta_ajuda_description_{lang}", "label": "Meta description", "type": "text"},
            ],
        },
        {
            "id": f"seo_conversor_{lang}",
            "label": f"SEO — Conversor ({lang_label})",
            "lang": lang,
            "path": f"{base}/conversor.html",
            "category": "seo",
            "fields": [
                {"key": f"meta_conversor_title_{lang}", "label": "Meta title", "type": "text"},
                {"key": f"meta_conversor_description_{lang}", "label": "Meta description", "type": "text"},
            ],
        },
        {
            "id": f"seo_resumo_{lang}",
            "label": f"SEO — Resumo ({lang_label})",
            "lang": lang,
            "path": f"{base}/resumo.html",
            "category": "seo",
            "fields": [
                {"key": f"meta_resumo_title_{lang}", "label": "Meta title", "type": "text"},
                {"key": f"meta_resumo_description_{lang}", "label": "Meta description", "type": "text"},
            ],
        },
    ]


for _lc_lang, _lc_label in LOCALE_CMS_LANGS:
    PAGE_SCHEMA.extend(_locale_seo_pages(_lc_lang, _lc_label))

PAGE_SCHEMA.extend(_locale_seo_pages("en", "EN"))

# Reordenar: PT → EN → ES → FR → DE (facilita backoffice e API)

def _nav_link(label: str, href: str, page: str = "", pricing_only: bool = False) -> dict:
    item: dict[str, Any] = {"label": label, "href": href}
    if page:
        item["page"] = page
    if pricing_only:
        item["pricingOnly"] = True
    return item


def default_nav_config(lang: str) -> dict[str, Any]:
    if lang == "en":
        return {
            "menuToolsLabel": "Tools",
            "menuAudienceLabel": "For",
            "tools": [
                _nav_link("Summarize PDF / Word", "resumo.html", "resumo"),
                _nav_link("URL Summary", "url-resumo.html", "url-resumo"),
                _nav_link("AI Questions", "perguntas.html", "perguntas"),
                _nav_link("Lesson Ready", "aula-pronta.html", "aula-pronta"),
                _nav_link("Chapters & timestamps", "capitulos.html", "capitulos"),
                _nav_link("File Converter", "conversor.html", "conversor"),
                _nav_link("Text proofreader", "corretor.html", "corretor"),
            ],
            "audience": [
                _nav_link("Classes", "aulas.html", "aulas"),
                _nav_link("Teachers", "professores.html", "professores"),
                _nav_link("Journalists", "jornalistas.html", "jornalistas"),
                _nav_link("Podcasts", "podcasts.html", "podcasts"),
                _nav_link("Meetings", "reunioes.html", "reunioes"),
                _nav_link("Testimonials", "testemunhos.html", "testemunhos"),
            ],
            "topLinks": [
                _nav_link("Help", "ajuda.html", "ajuda"),
                _nav_link("Pricing", "precos.html", "precos", pricing_only=True),
                _nav_link("Suggestions", "sugestoes.html", "sugestoes"),
            ],
            "ctaLabel": "Transcribe free",
            "ctaHref": "index.html",
            "footerTagline": "Transcribe, summarize and translate with AI — free and made in Portugal.",
            "footerEmail": "ouviescrevi@gmail.com",
            "footerCopyright": "© 2026 Ouviescrevi · Made in Portugal",
            "footerColumns": [
                {
                    "title": "Tools",
                    "links": [
                        _nav_link("Summary", "resumo.html"),
                        _nav_link("URL Summary", "url-resumo.html"),
                        _nav_link("Questions", "perguntas.html"),
                        _nav_link("Lesson Ready", "aula-pronta.html"),
                        _nav_link("Chapters", "capitulos.html"),
                        _nav_link("Converter", "conversor.html"),
                        _nav_link("Proofreader", "corretor.html"),
                    ],
                },
                {
                    "title": "For",
                    "links": [
                        _nav_link("Teachers", "professores.html"),
                        _nav_link("Journalists", "jornalistas.html"),
                        _nav_link("Podcasts", "podcasts.html"),
                        _nav_link("Classes", "aulas.html"),
                    ],
                },
                {
                    "title": "Legal",
                    "links": [
                        _nav_link("Privacy", "privacy.html"),
                        _nav_link("Terms", "terms.html"),
                        _nav_link("Cookies", "cookies.html"),
                        _nav_link("Help", "ajuda.html"),
                        _nav_link("Suggestions", "sugestoes.html"),
                    ],
                },
            ],
        }
    return {
        "menuToolsLabel": "Ferramentas",
        "menuAudienceLabel": "Para quem",
        "tools": [
            _nav_link("Resumo PDF / Word", "resumo.html", "resumo"),
            _nav_link("Resumo por URL", "url-resumo.html", "url-resumo"),
            _nav_link("Perguntas com IA", "perguntas.html", "perguntas"),
            _nav_link("Aula Pronta", "aula-pronta.html", "aula-pronta"),
            _nav_link("Capítulos & timestamps", "capitulos.html", "capitulos"),
            _nav_link("Conversor de ficheiros", "conversor.html", "conversor"),
            _nav_link("Corretor de texto", "corretor.html", "corretor"),
        ],
        "audience": [
            _nav_link("Aulas", "aulas.html", "aulas"),
            _nav_link("Professores", "professores.html", "professores"),
            _nav_link("Jornalistas", "jornalistas.html", "jornalistas"),
            _nav_link("Podcasts", "podcasts.html", "podcasts"),
            _nav_link("Reuniões", "reunioes.html", "reunioes"),
            _nav_link("Testemunhos", "testemunhos.html", "testemunhos"),
        ],
        "topLinks": [
            _nav_link("Ajuda", "ajuda.html", "ajuda"),
            _nav_link("Preços", "precos.html", "precos", pricing_only=True),
            _nav_link("Sugestões", "sugestoes.html", "sugestoes"),
        ],
        "ctaLabel": "Transcrever grátis",
        "ctaHref": "index.html",
        "footerTagline": "Transcreve, resume e traduz com IA — grátis e feito em Portugal.",
        "footerEmail": "ouviescrevi@gmail.com",
        "footerCopyright": "© 2026 Ouviescrevi · Feito em Portugal",
        "footerColumns": [
            {
                "title": "Ferramentas",
                "links": [
                    _nav_link("Resumo", "resumo.html"),
                    _nav_link("Resumo URL", "url-resumo.html"),
                    _nav_link("Perguntas", "perguntas.html"),
                    _nav_link("Aula Pronta", "aula-pronta.html"),
                    _nav_link("Capítulos", "capitulos.html"),
                    _nav_link("Conversor", "conversor.html"),
                    _nav_link("Corretor", "corretor.html"),
                ],
            },
            {
                "title": "Para quem",
                "links": [
                    _nav_link("Professores", "professores.html"),
                    _nav_link("Jornalistas", "jornalistas.html"),
                    _nav_link("Podcasts", "podcasts.html"),
                    _nav_link("Aulas", "aulas.html"),
                ],
            },
            {
                "title": "Legal",
                "links": [
                    _nav_link("Privacidade", "privacidade.html"),
                    _nav_link("Termos", "termos.html"),
                    _nav_link("Cookies", "cookies.html"),
                    _nav_link("Ajuda", "ajuda.html"),
                    _nav_link("Sugestões", "sugestoes.html"),
                ],
            },
        ],
    }


def nav_config_key(lang: str) -> str:
    return f"nav_config_{lang}" if lang != "pt" else "nav_config_pt"


def _nav_base_lang(lang: str) -> str:
    return "pt" if lang == "pt" else "en"


def merge_nav_config(data: dict[str, Any], lang: str) -> dict[str, Any]:
    """Preenche secções vazias com os defaults (evita menu em branco)."""
    base = default_nav_config(_nav_base_lang(lang))
    out = dict(base)
    for key, val in data.items():
        if val is not None and val != "":
            out[key] = val
    for key in ("tools", "audience", "topLinks", "footerColumns"):
        if not out.get(key):
            out[key] = list(base.get(key) or [])
    return out


def parse_nav_config(raw: str | None, lang: str = "pt") -> dict[str, Any]:
    base_lang = _nav_base_lang(lang)
    if not raw:
        return default_nav_config(base_lang)
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(data, dict):
            return merge_nav_config(data, lang)
    except (json.JSONDecodeError, TypeError):
        pass
    return default_nav_config(base_lang)


def nav_defaults_for_admin() -> dict[str, dict[str, Any]]:
    return {lang: default_nav_config(_nav_base_lang(lang)) for lang in ("pt", "en", "es", "fr", "de")}


_LANG_ORDER = {"pt": 0, "en": 1, "es": 2, "fr": 3, "de": 4}
PAGE_SCHEMA.sort(
    key=lambda p: (
        _LANG_ORDER.get(p.get("lang", "pt"), 9),
        1 if p.get("category") == "seo" else 0,
        str(p.get("label", "")),
    )
)

DEFAULT_SITE_CONTENT: dict[str, str] = {
    "home_intro_html": (
        "<p><strong>Cola áudio ou vídeo e obtém o texto em minutos.</strong><br>"
        "Transcrever 🎙️ · Traduzir 🌍 · Resumir 📌 · Converter ficheiros 📄 "
        "— grátis, em português.</p>"
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
    "en_home_intro_html": (
        "<p><strong>Upload audio or video and get text in minutes.</strong><br>"
        "Transcribe 🎙️ · Translate 🌍 · Summarise 📌 · Convert files 📄 "
        "— simple, fast and free.</p>"
    ),
    "es_home_intro_html": (
        "<p><strong>Sube audio o vídeo y obtén el texto en minutos.</strong><br>"
        "Transcribir 🎙️ · Traducir 🌍 · Resumir 📌 · Convertir archivos 📄 "
        "— simple, rápido y gratis.</p>"
    ),
    "fr_home_intro_html": (
        "<p><strong>Déposez un audio ou une vidéo et obtenez le texte en quelques minutes.</strong><br>"
        "Transcrire 🎙️ · Traduire 🌍 · Résumer 📌 · Convertir des fichiers 📄 "
        "— simple, rapide et gratuit.</p>"
    ),
    "de_home_intro_html": (
        "<p><strong>Lade Audio oder Video hoch und erhalte den Text in wenigen Minuten.</strong><br>"
        "Transkribieren 🎙️ · Übersetzen 🌍 · Zusammenfassen 📌 · Dateien konvertieren 📄 "
        "— einfach, schnell und kostenlos.</p>"
    ),
    "en_resumo_title": "📌 Smart Summary",
    "en_resumo_lead": "Paste your text or upload a PDF or Word file, then choose a summary style.",
    "en_url_resumo_title": "🔗 Smart Summary from URL",
    "en_url_resumo_lead": "Paste an article link to generate an automatic AI summary.",
    "en_perguntas_title": "📘 AI Quiz Generator",
    "en_perguntas_lead": (
        "Paste your text here to generate multiple-choice questions with answers and explanations."
    ),
    "en_capitulos_title": "⏱️ Chapters & timestamps",
    "en_capitulos_lead": "Paste a timestamped transcript and get organized chapters — ready for YouTube or podcasts.",
    "aulas_title": "🎥 Aulas",
    "aulas_body": (
        "<p>Transforma vídeos de aulas em texto claro, bem formatado e pronto a partilhar. "
        "Útil para estudantes, professores, tutores e plataformas educativas.</p>"
    ),
    "professores_title": "🎓 Professores",
    "professores_body": (
        "<h2>Transforma as tuas aulas em texto com IA</h2>"
        "<p>Grava explicações, aulas ou feedback e converte tudo automaticamente em texto estruturado. "
        "Ideal para criar resumos, materiais de apoio ou registos para alunos.</p>"
        "<p>Podes ainda traduzir, resumir ou exportar em vários formatos (PDF, Word, TXT, etc.).</p>"
    ),
    "jornalistas_title": "📰 Jornalistas",
    "jornalistas_body": (
        "<p>Transcreve entrevistas, conferências de imprensa ou reportagens com precisão e rapidez. "
        "Facilita a produção de conteúdos e permite focar no que realmente importa: contar histórias.</p>"
    ),
    "podcasts_title": "🎙️ Podcasts",
    "podcasts_body": (
        "<p>Converte episódios de podcast em texto com um clique. Melhora o SEO, partilha transcrições "
        "com o público e reutiliza os conteúdos em newsletters, artigos ou redes sociais.</p>"
    ),
    "reunioes_title": "🗣️ Reuniões",
    "reunioes_body": (
        "<p>Grava e transcreve automaticamente reuniões presenciais ou online. Garante que nenhuma decisão, "
        "ideia ou compromisso é esquecido. Ideal para equipas, empresas e freelancers.</p>"
    ),
    "testemunhos_title": "🧑‍⚖️ Testemunhos",
    "testemunhos_body": (
        "<p>Ideal para transcrever testemunhos jurídicos, audiências e declarações. Garante precisão e "
        "registo textual para processos legais ou administrativos.</p>"
    ),
    "resumo_title": "📌 Resumo Inteligente",
    "resumo_lead": "Cola o teu texto, ou carrega um ficheiro PDF ou Word (.docx), e escolhe o estilo de resumo",
    "corretor_title": "✍️ Corretor de Texto com IA",
    "corretor_lead": "Cola o teu texto para corrigir erros ortográficos e gramaticais automaticamente.",
    "perguntas_title": "📘 Gerador de Perguntas com IA",
    "perguntas_lead": "Cola aqui o teu texto e gera perguntas de escolha múltipla com respostas e explicações.",
    "capitulos_title": "⏱️ Capítulos & timestamps",
    "capitulos_lead": "Cola uma transcrição com timestamps e obtém capítulos organizados — prontos para YouTube ou podcasts.",
    "url_resumo_title": "🔗 Resumo Inteligente por URL",
    "url_resumo_lead": "Insere o link de um artigo ou página online para gerar um resumo automático com IA.",
    "privacidade_meta": "Última atualização: 22 de junho de 2026",
    "privacidade_disclaimer": (
        "<p>Este documento descreve como o serviço <strong>Ouviescrevi</strong> trata dados pessoais. "
        "Recomendamos revisão por um advogado antes de uso comercial em larga escala.</p>"
    ),
    "termos_meta": "Última atualização: 22 de junho de 2026",
    "termos_intro": (
        "<p>Ao utilizar o <strong>Ouviescrevi</strong> (ouviescrevi.pt), aceitas estes termos. "
        "Se não concordares, não utilizes o serviço.</p>"
    ),
    "cookies_meta": "Última atualização: 22 de junho de 2026",
    "cookies_intro": (
        "<h2>1. O que são cookies?</h2>"
        "<p>Cookies são pequenos ficheiros guardados no teu dispositivo. Também utilizamos tecnologias "
        "semelhantes, como <strong>localStorage</strong> e <strong>sessionStorage</strong> do browser.</p>"
        "<p>Esta política explica o que usamos no <strong>ouviescrevi.pt</strong> e como podes gerir as tuas preferências.</p>"
    ),
    "en_privacidade_meta": (
        'Last updated: 22 June 2026 · <a href="../privacidade.html">Português</a>'
    ),
    "en_privacidade_disclaimer": (
        "<p>This policy describes how <strong>Ouviescrevi</strong> (ouviescrevi.pt) processes personal data. "
        "We recommend legal review before large-scale commercial use.</p>"
    ),
    "en_termos_meta": (
        'Last updated: 22 June 2026 · <a href="../termos.html">Português</a>'
    ),
    "en_termos_intro": (
        "<p>By using <strong>Ouviescrevi</strong>, you agree to these terms. "
        "If you do not agree, do not use the service.</p>"
    ),
    "en_cookies_meta": (
        'Last updated: 22 June 2026 · <a href="../cookies.html">Português</a>'
    ),
    "en_cookies_intro": (
        "<h2>1. What are cookies?</h2>"
        "<p>Cookies are small files stored on your device. We also use similar technologies such as "
        "<strong>localStorage</strong> and <strong>sessionStorage</strong>.</p>"
        "<p>This policy explains what we use on <strong>ouviescrevi.pt</strong> and how you can manage "
        "your preferences.</p>"
    ),
    "maintenance_message": (
        "<p>🛑 O serviço está temporariamente em manutenção. Novas transcrições estão indisponíveis.</p>"
    ),
    "home_testimonials": "",
    "meta_home_title": "Ouviescrevi — Transcrição de Áudio e Vídeo com IA Grátis",
    "meta_home_description": (
        "Transcreve áudio e vídeo online com inteligência artificial, grátis e sem registo. "
        "Resumos, tradução, legendas SRT e conversão de ficheiros. Feito em Portugal."
    ),
    "meta_conversor_title": "Conversor de Ficheiros Online Grátis — Word, PDF, Imagem | Ouviescrevi",
    "meta_conversor_description": "Converte Word para PDF, PDF para texto e imagens para PDF no browser. Gratuito e sem instalação.",
    "meta_resumo_title": "Resumo Automático com IA — PDF, Word e Texto | Ouviescrevi",
    "meta_resumo_description": "Gera resumos inteligentes com IA a partir de PDF, Word ou texto.",
    "meta_ajuda_title": "Ajuda e FAQ — Como Usar o Ouviescrevi",
    "meta_ajuda_description": "Respostas às perguntas frequentes sobre transcrição com IA e funcionalidades do Ouviescrevi.",
}

_LOCALE_SEO_DEFAULTS: dict[str, dict[str, str]] = {
    "en": {
        "meta_home_title_en": "Ouviescrevi — Free AI Audio & Video Transcription",
        "meta_home_description_en": (
            "Transcribe audio and video online with AI for free. Summaries, translation, "
            "SRT subtitles and file conversion. No sign-up required."
        ),
        "meta_ajuda_title_en": "Help & FAQ — How to Use Ouviescrevi",
        "meta_ajuda_description_en": (
            "Frequently asked questions about AI transcription, supported formats, "
            "privacy and Ouviescrevi features."
        ),
        "meta_conversor_title_en": "Free Online File Converter — Word, PDF, Image | Ouviescrevi",
        "meta_conversor_description_en": (
            "Convert Word to PDF, PDF to text and images to PDF in your browser. Free, fast and no installation."
        ),
        "meta_resumo_title_en": "AI Summary Generator — PDF, Word & Text | Ouviescrevi",
        "meta_resumo_description_en": (
            "Generate smart AI summaries from PDF, Word or plain text. Formal, simple, bullet points or meeting minutes."
        ),
    },
    "es": {
        "meta_home_title_es": "Ouviescrevi — Transcripción de audio y vídeo con IA gratis",
        "meta_home_description_es": (
            "Transcribe audio y vídeo online con inteligencia artificial, gratis y sin registro. "
            "Resúmenes, traducción, subtítulos SRT y conversión de archivos."
        ),
        "meta_ajuda_title_es": "Ayuda y FAQ — Cómo usar Ouviescrevi",
        "meta_ajuda_description_es": (
            "Preguntas frecuentes sobre transcripción con IA, formatos compatibles y privacidad."
        ),
        "meta_conversor_title_es": "Conversor de archivos online gratis — Word, PDF, imagen | Ouviescrevi",
        "meta_conversor_description_es": (
            "Convierte Word a PDF, PDF a texto e imágenes a PDF en el navegador. Gratis y sin instalación."
        ),
        "meta_resumo_title_es": "Resumen automático con IA — PDF, Word y texto | Ouviescrevi",
        "meta_resumo_description_es": (
            "Genera resúmenes inteligentes con IA a partir de PDF, Word o texto."
        ),
    },
    "fr": {
        "meta_home_title_fr": "Ouviescrevi — Transcription audio et vidéo IA gratuite",
        "meta_home_description_fr": (
            "Transcrivez audio et vidéo en ligne avec l'IA, gratuitement et sans inscription. "
            "Résumés, traduction, sous-titres SRT et conversion de fichiers."
        ),
        "meta_ajuda_title_fr": "Aide et FAQ — Utiliser Ouviescrevi",
        "meta_ajuda_description_fr": (
            "Questions fréquentes sur la transcription IA, les formats pris en charge et la confidentialité."
        ),
        "meta_conversor_title_fr": "Convertisseur de fichiers en ligne gratuit — Word, PDF | Ouviescrevi",
        "meta_conversor_description_fr": (
            "Convertissez Word en PDF, PDF en texte et images en PDF dans le navigateur."
        ),
        "meta_resumo_title_fr": "Résumé automatique avec IA — PDF, Word et texte | Ouviescrevi",
        "meta_resumo_description_fr": (
            "Générez des résumés intelligents avec l'IA à partir de PDF, Word ou texte."
        ),
    },
    "de": {
        "meta_home_title_de": "Ouviescrevi — Kostenlose KI-Audio- und Video-Transkription",
        "meta_home_description_de": (
            "Transkribiere Audio und Video online mit KI, kostenlos und ohne Anmeldung. "
            "Zusammenfassungen, Übersetzung, SRT-Untertitel und Dateikonvertierung."
        ),
        "meta_ajuda_title_de": "Hilfe & FAQ — Ouviescrevi nutzen",
        "meta_ajuda_description_de": (
            "Häufige Fragen zu KI-Transkription, unterstützten Formaten und Datenschutz."
        ),
        "meta_conversor_title_de": "Kostenloser Online-Dateikonverter — Word, PDF | Ouviescrevi",
        "meta_conversor_description_de": (
            "Word zu PDF, PDF zu Text und Bilder zu PDF im Browser konvertieren."
        ),
        "meta_resumo_title_de": "KI-Zusammenfassung — PDF, Word & Text | Ouviescrevi",
        "meta_resumo_description_de": (
            "Erstelle intelligente Zusammenfassungen mit KI aus PDF, Word oder Text."
        ),
    },
}

for _lng, _defaults in _LOCALE_SEO_DEFAULTS.items():
    for _k, _v in _defaults.items():
        DEFAULT_SITE_CONTENT.setdefault(_k, _v)

for _lc_lang in ("es", "fr", "de"):
    for _key, _val in list(DEFAULT_SITE_CONTENT.items()):
        if _key.startswith("en_"):
            DEFAULT_SITE_CONTENT.setdefault(_lc_lang + _key[2:], _val)
    DEFAULT_SITE_CONTENT.setdefault(
        f"{_lc_lang}_home_intro_html",
        DEFAULT_SITE_CONTENT.get("home_intro_html", ""),
    )
    for _suffix in (
        "resumo_title",
        "resumo_lead",
        "url_resumo_title",
        "url_resumo_lead",
        "perguntas_title",
        "perguntas_lead",
        "capitulos_title",
        "capitulos_lead",
    ):
        DEFAULT_SITE_CONTENT.setdefault(
            f"{_lc_lang}_{_suffix}",
            DEFAULT_SITE_CONTENT.get(f"en_{_suffix}", ""),
        )

for _nav_lang in ("pt", "en", "es", "fr", "de"):
    _nav_base = "pt" if _nav_lang == "pt" else "en"
    DEFAULT_SITE_CONTENT.setdefault(
        nav_config_key(_nav_lang),
        json.dumps(default_nav_config(_nav_base), ensure_ascii=False),
    )

CONTENT_KEYS = frozenset(DEFAULT_SITE_CONTENT.keys())
_PAGE_KEYS: dict[str, list[str]] = {
    page["id"]: [f["key"] for f in page["fields"]] for page in PAGE_SCHEMA
}


def get_page_schema() -> list[dict[str, Any]]:
    return PAGE_SCHEMA


def keys_for_page(page_id: str) -> list[str]:
    return list(_PAGE_KEYS.get(page_id, []))


def get_seo_overrides() -> dict[str, dict[str, str]]:
    content = get_all_content()
    mapping: list[tuple[str, str, str]] = [
        ("/index.html", "meta_home_title", "meta_home_description"),
        ("/conversor.html", "meta_conversor_title", "meta_conversor_description"),
        ("/resumo.html", "meta_resumo_title", "meta_resumo_description"),
        ("/ajuda.html", "meta_ajuda_title", "meta_ajuda_description"),
    ]
    for lang in ("en", "es", "fr", "de"):
        mapping.extend(
            [
                (f"/{lang}/index.html", f"meta_home_title_{lang}", f"meta_home_description_{lang}"),
                (f"/{lang}/conversor.html", f"meta_conversor_title_{lang}", f"meta_conversor_description_{lang}"),
                (f"/{lang}/resumo.html", f"meta_resumo_title_{lang}", f"meta_resumo_description_{lang}"),
                (f"/{lang}/ajuda.html", f"meta_ajuda_title_{lang}", f"meta_ajuda_description_{lang}"),
            ]
        )
    out: dict[str, dict[str, str]] = {}
    for path, tk, dk in mapping:
        title = content.get(tk, "")
        description = content.get(dk, "")
        if title or description:
            out[path] = {"title": title, "description": description}
    return out


def _db_conn():
    return get_connection()


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
    if field_type == "json" or key.startswith("nav_config_"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Valor inválido para {key}")
        return json.dumps(parsed, ensure_ascii=False)
    return text


def get_all_content() -> dict[str, str]:
    out = dict(DEFAULT_SITE_CONTENT)
    conn = _db_conn()
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
    conn = _db_conn()
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
    conn = _db_conn()
    try:
        cur = conn.cursor()
        for key in to_drop:
            if key in CONTENT_KEYS:
                cur.execute("DELETE FROM site_content WHERE key = ?", (key,))
        conn.commit()
    finally:
        conn.close()
    return get_all_content()
