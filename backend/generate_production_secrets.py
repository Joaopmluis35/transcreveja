#!/usr/bin/env python3
"""Gera tokens aleatórios para colar no Render → Environment (uma vez)."""
import secrets

print("Copia estes valores para o Render (Environment):")
print()
print(f"ADMIN_TOKEN={secrets.token_urlsafe(32)}")
print(f"API_TOKEN={secrets.token_urlsafe(32)}")
print(f"BACKOFFICE_PASSWORD={secrets.token_urlsafe(24)}")
print()
print("OPENAI_API_KEY=sk-...  (cola a tua chave da OpenAI manualmente)")
