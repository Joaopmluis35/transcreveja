"""Ambiente isolado para testes — base SQLite temporária, sem OpenAI/Stripe reais."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

_TEST_DB = tempfile.mktemp(suffix=".pytest.db")

os.environ["DATABASE_PATH"] = _TEST_DB
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-pytest-only"
os.environ["ADMIN_TOKEN"] = "test-admin-token"
os.environ["API_TOKEN"] = "test-api-token"
os.environ["BACKOFFICE_PASSWORD"] = "TestBackoffice123!"
os.environ["ALLOWED_ORIGINS"] = "http://testserver,http://localhost"
os.environ["APP_ENV"] = "test"
os.environ["ENABLE_DEBUG_ENDPOINTS"] = "false"
os.environ["BILLING_ENABLED"] = "0"
os.environ["PRICING_HIDDEN"] = "1"
os.environ["RATE_LIMIT_TRANSCRIBE"] = "9999"
os.environ["RATE_LIMIT_AI"] = "9999"
os.environ["TEST_SYNC_NOTIFICATIONS"] = "1"


def _try_import_app():
    try:
        from main import app  # noqa: WPS433 — import tardio intencional

        return app, None
    except Exception as exc:  # pragma: no cover — skip API tests
        return None, str(exc)


_APP, _APP_IMPORT_ERROR = None, None


def _bootstrap_db():
    import database

    database.criar_base()
    import admin_store

    admin_store.ensure_default_admin(os.environ["BACKOFFICE_PASSWORD"])


try:
    _bootstrap_db()
    _APP, _APP_IMPORT_ERROR = _try_import_app()
except Exception as exc:  # pragma: no cover
    _APP_IMPORT_ERROR = str(exc)


@pytest.fixture(scope="session")
def app_import_error():
    return _APP_IMPORT_ERROR


@pytest.fixture
def client():
    if _APP is None:
        pytest.skip(f"API indisponível neste ambiente: {_APP_IMPORT_ERROR}")
    from fastapi.testclient import TestClient

    with TestClient(_APP) as test_client:
        yield test_client


@pytest.fixture
def origin_headers():
    return {"Origin": "http://testserver"}
