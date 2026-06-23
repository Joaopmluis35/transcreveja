"""Esquema SQLite e migrações incrementais."""
from __future__ import annotations

import os
import sqlite3

DB_PATH = os.getenv("DATABASE_PATH", "ouviescrevi.db")


def db_path() -> str:
    return DB_PATH


def get_connection() -> sqlite3.Connection:
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
