"""Esquema SQLite e migrações incrementais (ficheiro local ou Turso Cloud)."""
from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator, KeysView
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DATABASE_PATH", "ouviescrevi.db")


def _turso_url() -> str:
    return os.getenv("TURSO_DATABASE_URL", "").strip()


def _turso_token() -> str:
    return os.getenv("TURSO_AUTH_TOKEN", "").strip()


def use_turso() -> bool:
    return bool(_turso_url() and _turso_token())


def database_backend() -> str:
    return "turso" if use_turso() else "local"


def db_path() -> str:
    if use_turso():
        return _turso_url()
    return DB_PATH


class _DictRow:
    """Linha estilo sqlite3.Row para resultados libsql (nome ou índice)."""

    __slots__ = ("_columns", "_values", "_map")

    def __init__(self, columns: list[str], values: tuple[Any, ...]):
        self._columns = columns
        self._values = values
        self._map = dict(zip(columns, values))

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._map[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def keys(self) -> KeysView[str]:
        return self._map.keys()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._map)


class _TursoCursor:
    def __init__(self, inner_cursor: Any):
        self._cur = inner_cursor
        self.description = getattr(inner_cursor, "description", None)

    @property
    def lastrowid(self) -> Any:
        return getattr(self._cur, "lastrowid", None)

    def execute(self, sql: str, params: tuple | list = ()) -> _TursoCursor:
        if params:
            self._cur.execute(sql, params)
        else:
            self._cur.execute(sql)
        self.description = getattr(self._cur, "description", None)
        return self

    def fetchone(self) -> Any:
        return _as_sqlite_row(self, self._cur.fetchone())

    def fetchall(self) -> list[Any]:
        raw = self._cur.fetchall()
        if raw is None:
            return []
        return [_as_sqlite_row(self, row) for row in raw]

    def __iter__(self) -> Iterator[Any]:
        row = self.fetchone()
        while row is not None:
            yield row
            row = self.fetchone()


class _TursoConnection:
    """Compatibiliza a API sqlite3 (row_factory, execute, cursor) com libsql."""

    def __init__(self, inner_conn: Any):
        self._conn = inner_conn

    def cursor(self) -> _TursoCursor:
        return _TursoCursor(self._conn.cursor())

    def execute(self, sql: str, params: tuple | list = ()) -> _TursoCursor:
        return self.cursor().execute(sql, params)

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _as_sqlite_row(cur: _TursoCursor, row: Any) -> Any:
    if row is None:
        return None
    if hasattr(row, "keys") and not isinstance(row, (tuple, list)):
        return row
    if not cur.description:
        return row
    columns = [col[0] for col in cur.description]
    if isinstance(row, dict):
        return _DictRow(columns, tuple(row.get(c) for c in columns))
    return _DictRow(columns, tuple(row))


def _ensure_db_dir() -> None:
    if use_turso():
        return
    parent = os.path.dirname(os.path.abspath(DB_PATH))
    if parent:
        os.makedirs(parent, exist_ok=True)


def row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, _DictRow):
        return row.as_dict()
    if isinstance(row, dict):
        return row
    return dict(row)


def scalar_int(row: Any, key: str, *, index: int = 0, default: int = 0) -> int:
    if row is None:
        return default
    raw: Any = default
    if isinstance(row, (_DictRow, dict)):
        data = row.as_dict() if isinstance(row, _DictRow) else row
        raw = data.get(key, data.get(key.upper(), data.get(key.lower())))
        if raw is None and index >= 0:
            try:
                raw = row[index]
            except (KeyError, IndexError, TypeError):
                raw = default
    else:
        try:
            raw = row[index]
        except (IndexError, TypeError):
            raw = default
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return default


def scalar_float(row: Any, key: str, *, index: int = 0, default: float = 0.0) -> float:
    if row is None:
        return default
    raw: Any = default
    if isinstance(row, (_DictRow, dict)):
        data = row.as_dict() if isinstance(row, _DictRow) else row
        raw = data.get(key, data.get(key.upper(), data.get(key.lower())))
        if raw is None and index >= 0:
            try:
                raw = row[index]
            except (KeyError, IndexError, TypeError):
                raw = default
    else:
        try:
            raw = row[index]
        except (IndexError, TypeError):
            raw = default
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def get_connection() -> Any:
    if use_turso():
        import libsql

        raw = libsql.connect(
            database=_turso_url(),
            auth_token=_turso_token(),
        )
        return _TursoConnection(raw)
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
        ("ui_locale", "TEXT"),
        ("page_path", "TEXT"),
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
        ("visitor_uid", "TEXT"),
        ("ip_label", "TEXT"),
        ("utm_source", "TEXT"),
        ("utm_medium", "TEXT"),
        ("utm_campaign", "TEXT"),
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
        url = _turso_url()
        logger.info("Base de dados: Turso (%s)", url.split("/")[2] if "/" in url else "remoto")
    else:
        logger.warning(
            "Base de dados: SQLite local (%s). TURSO_URL=%s TURSO_TOKEN=%s",
            DB_PATH,
            "sim" if _turso_url() else "não",
            "sim" if _turso_token() else "não",
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

        cur.execute("""
            CREATE TABLE IF NOT EXISTS site_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS email_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                recipient TEXT NOT NULL,
                subject TEXT NOT NULL,
                status TEXT NOT NULL,
                detail TEXT,
                actor TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                filename TEXT,
                language TEXT,
                size_bytes INTEGER,
                duration_sec REAL,
                transcription TEXT,
                formatted TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_transcriptions_email "
            "ON user_transcriptions(user_email, created_at DESC)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                original_text TEXT,
                corrected_text TEXT,
                mode TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_corrections_email "
            "ON user_corrections(user_email, created_at DESC)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_usage (
                usage_key TEXT NOT NULL,
                usage_day TEXT NOT NULL,
                transcribe_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (usage_key, usage_day)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                user_email TEXT PRIMARY KEY,
                plan TEXT NOT NULL DEFAULT 'free',
                status TEXT,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                current_period_end TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_password_reset_email "
            "ON password_reset_tokens(email)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS shared_transcripts (
                id TEXT PRIMARY KEY,
                title TEXT,
                text TEXT NOT NULL,
                locale TEXT DEFAULT 'pt',
                created_at TEXT NOT NULL,
                expires_at TEXT,
                view_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                priority TEXT DEFAULT 'media',
                category TEXT DEFAULT 'produto',
                evidence TEXT,
                cursor_prompt TEXT,
                status TEXT NOT NULL DEFAULT 'new',
                source_days INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_insights_status "
            "ON ai_insights(status, id DESC)"
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_estudo_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                source_days INTEGER,
                horizon_days INTEGER,
                model TEXT,
                summary TEXT,
                trend_label TEXT,
                risk_level TEXT,
                metrics_json TEXT,
                series_json TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_estudo_runs_created "
            "ON ai_estudo_runs(id DESC)"
        )
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_estudo_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                title TEXT NOT NULL,
                detail TEXT NOT NULL,
                priority TEXT DEFAULT 'media',
                category TEXT DEFAULT 'crescimento',
                evidence TEXT,
                cursor_prompt TEXT,
                status TEXT NOT NULL DEFAULT 'new',
                source_days INTEGER,
                horizon_days INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_estudo_suggestions_status "
            "ON ai_estudo_suggestions(status, id DESC)"
        )

        if not _column_exists(cur, "site_users", "marketing_opt_in"):
            cur.execute("ALTER TABLE site_users ADD COLUMN marketing_opt_in INTEGER DEFAULT 0")

        conn.commit()
    finally:
        conn.close()


criar_base()
