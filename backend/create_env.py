"""Gera backend/.env para desenvolvimento local (não commitar)."""
import secrets
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"

if ENV_PATH.exists():
    print(f"Já existe: {ENV_PATH}")
    print("Apaga manualmente se quiseres regenerar.")
    raise SystemExit(1)

admin = secrets.token_urlsafe(32)
api = secrets.token_urlsafe(32)
password = secrets.token_urlsafe(18)

content = f"""# Gerado automaticamente para desenvolvimento local — NÃO commitar
OPENAI_API_KEY=sk-COLE_AQUI_A_TUA_CHAVE_OPENAI

ADMIN_TOKEN={admin}
API_TOKEN={api}
BACKOFFICE_PASSWORD={password}

ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000
PUBLIC_API_BASE=http://127.0.0.1:8000
APP_ENV=development
ENABLE_DEBUG_ENDPOINTS=true

RATE_LIMIT_TRANSCRIBE=20
RATE_LIMIT_TRANSCRIBE_WINDOW=3600
RATE_LIMIT_VIDEO_SUBS=10
RATE_LIMIT_VIDEO_SUBS_WINDOW=3600
RATE_LIMIT_AI=60
RATE_LIMIT_AI_WINDOW=3600
"""

ENV_PATH.write_text(content, encoding="utf-8")
print(f"Criado: {ENV_PATH}")
print(f"BACKOFFICE_PASSWORD (guarda isto): {password}")
print(f"API_TOKEN (para testes manuais): {api}")
