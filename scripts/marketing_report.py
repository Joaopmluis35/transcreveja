"""Lê o relatório de marketing da API com a chave só de leitura do Claude.

A chave (criada no backoffice → Marketing → «Criar chave para o Claude») fica fora do repositório,
em %USERPROFILE%\\.ouviescrevi\\claude.key. Uso: py scripts/marketing_report.py [dias]
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.ouviescrevi.pt/api/admin/marketing?days="
KEY_FILE = Path.home() / ".ouviescrevi" / "claude.key"


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    if not KEY_FILE.exists():
        print(f"Falta a chave em {KEY_FILE}", file=sys.stderr)
        return 2
    req = urllib.request.Request(API + str(days), headers={
        "Authorization": "Bearer " + KEY_FILE.read_text(encoding="utf-8").strip(),
        "User-Agent": "ouviescrevi-marketing-report",
    })
    try:
        # A API no Render adormece: o 1.º pedido pode levar ~1 minuto.
        with urllib.request.urlopen(req, timeout=120) as res:
            data = json.load(res)
    except urllib.error.HTTPError as exc:
        print(f"Erro {exc.code}: chave inválida, revogada ou expirada?" if exc.code == 403 else f"Erro {exc.code}",
              file=sys.stderr)
        return 1
    print(json.dumps(data, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
