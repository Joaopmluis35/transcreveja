"""Conteúdo editável do site (CMS simples)."""
from __future__ import annotations

import sqlite3
from datetime import datetime

DEFAULT_SITE_CONTENT: dict[str, str] = {
    "home_intro_html": (
        "<strong>🧠 Ouviescrevi</strong> é o teu assistente com IA para<br>"
        "<strong>transcrever</strong> 🎙️, <strong>traduzir</strong> 🌍, <strong>resumir</strong> 📌 "
        "e <strong>converter ficheiros</strong> 📄<br>— simples, rápido e gratuito."
    ),
    "seo_title": "🧠 O que é o Ouviescrevi?",
    "seo_p1": (
        "Ouviescrevi é uma ferramenta online que usa Inteligência Artificial (IA) para "
        "<strong>transcrever, resumir, traduzir, corrigir</strong> e <strong>classificar conteúdos</strong>, "
        "além de <strong>gerar emails</strong>, <strong>criar perguntas de estudo</strong> e até "
        "<strong>produzir vídeos com legendas automáticas</strong>."
    ),
    "seo_p2": (
        "Ideal para professores, jornalistas, estudantes, empresas e qualquer pessoa que precise "
        "transformar áudio, vídeo ou texto em conhecimento útil. 🚀"
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
}

CONTENT_KEYS = frozenset(DEFAULT_SITE_CONTENT.keys())


def _db_path() -> str:
    return "ouviescrevi.db"


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
                (key, str(value), now),
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
