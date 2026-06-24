"""Esquema SQLite e migrações incrementais (ficheiro local ou Turso Cloud)."""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DATABASE_PATH", "ouviescrevi.db")
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "").strip()
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "").strip()


def use_turso() -> bool:
    return bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)


def database_backend() -> str:
    return "turso" if use_turso() else "local"


def db_path() -> str:
    if use_turso():
        return TURSO_DATABASE_URL
    return DB_PATH


def _ensure_db_dir() -> None:
    if use_turso():
        return
    parent = os.path.dirname(os.path.abspath(DB_PATH))
    if parent:
        os.makedirs(parent, exist_ok=True)


def get_connection() -> Any:
    if use_turso():
        import libsql

        conn = libsql.connect(
            database=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        conn.row_factory = sqlite3.Row
        return conn
    _ensure_db_dir()
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _migrate_transcricoes(cur: sqlite3.Cursor) -> None:
    for col, typedef in (
        ("language", "TEXT"),
        ("size_bytes", "INTEGER"),
        ("duration_sec", "REAL"),
        ("processing_sec", "REAL"),
        ("status", "TEXT"),
        ("error_message", "TEXT"),
    ):
        if not _column_exists(cur, "transcricoes", col):
            cur.execute(f"ALTER TABLE transcricoes ADD COLUMN {col} {typedef}")


def _migrate_status(cur: sqlite3.Cursor) -> None:
    for col, typedef in (
        ("maintenance_message", "TEXT"),
        ("block_transcribe_only", "INTEGER DEFAULT 1"),
    ):
        if not _column_exists(cur, "status", col):
            cur.execute(f"ALTER TABLE status ADD COLUMN {col} {typedef}")


def _migrate_visitas(cur: sqlite3.Cursor) -> None:
    for col, typedef in (
        ("referrer", "TEXT"),
        ("user_agent", "TEXT"),
        ("device_type", "TEXT"),
    ):
        if not _column_exists(cur, "visitas", col):
            cur.execute(f"ALTER TABLE visitas ADD COLUMN {col} {typedef}")


def criar_base() -> None:
    if os.getenv("APP_ENV") == "production" and not use_turso():
        logger.warning(
            "APP_ENV=production sem TURSO_DATABASE_URL/TURSO_AUTH_TOKEN: "
            "a base SQLite no Render Free é efémera e perde-se em cada redeploy."
        )

    backend = database_backend()
    if use_turso():
        logger.info("Base de dados: Turso (%s)", TURSO_DATABASE_URL.split("/")[2] if "/" in TURSO_DATABASE_URL else "remoto")
    else:
        logger.warning(
            "Base de dados: SQLite local (%s). TURSO_URL=%s TURSO_TOKEN=%s",
            DB_PATH,
            "sim" if TURSO_DATABASE_URL else "não",
            "sim" if TURSO_AUTH_TOKEN else "não",
        )

    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcricoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ficheiro TEXT,
                data TEXT
            )
        """)
        _migrate_transcricoes(cur)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS status (
                id INTEGER PRIMARY KEY,
                manutencao BOOLEAN
            )
        """)
        _migrate_status(cur)
        cur.execute("INSERT OR IGNORE INTO status (id, manutencao) VALUES (1, 0)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS visitas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                day TEXT NOT NULL,
                visitor_hash TEXT,
                created_at TEXT NOT NULL
            )
        """)
        _migrate_visitas(cur)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_visitas_day ON visitas(day)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS site_content (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sugestoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT,
                mensagem TEXT NOT NULL,
                lang TEXT DEFAULT 'pt',
                created_at TEXT NOT NULL,
                lida INTEGER DEFAULT 0
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS api_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                status_code INTEGER,
                message TEXT,
                client_ip TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS site_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS site_banners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texto TEXT NOT NULL,
                link TEXT,
                ativo INTEGER DEFAULT 0,
                starts_at TEXT,
                ends_at TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'editor',
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)

        conn.commit()
    finally:
        conn.close()


criar_base()
