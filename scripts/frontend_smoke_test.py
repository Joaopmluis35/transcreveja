#!/usr/bin/env python3
"""Smoke tests do frontend estático — ficheiros referenciados existem."""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

REQUIRED_AUDIENCE = ("professores", "podcasts", "aulas", "jornalistas", "reunioes", "testemunhos")
REQUIRED_LOCALES = ("pt", "en", "es", "fr", "de")
REQUIRED_PRICING = ("pt", "en", "es", "fr", "de")

SCRIPT_SRC = re.compile(r'<script[^>]+src=["\']([^"\']+)["\']', re.I)
LINK_HREF = re.compile(r'<link[^>]+href=["\']([^"\']+)["\']', re.I)


def path_from_url_path(url_path: str) -> Path:
    p = url_path.strip("/") or "index.html"
    return FRONTEND.joinpath(*p.split("/"))


def resolve_asset(html_path: Path, ref: str) -> Path | None:
    ref = ref.split("?")[0].split("#")[0]
    if not ref or ref.startswith(("http://", "https://", "//", "data:", "mailto:")):
        return None
    if ref.startswith("/"):
        return FRONTEND / ref.lstrip("/")
    return (html_path.parent / ref).resolve()


def should_skip_html(path: Path) -> bool:
    parts = path.parts
    if "archive" in parts:
        return True
    if path.name in ("admin.html", "backoffice.html", "header.html", "footer.html"):
        return True
    return False


def sitemap_paths() -> list[str]:
    tree = ET.parse(FRONTEND / "sitemap.xml")
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    paths: list[str] = []
    for url in tree.findall(".//sm:url", ns):
        loc = url.find("sm:loc", ns)
        if loc is None or not loc.text:
            continue
        text = loc.text.strip()
        if "ouviescrevi.pt/" in text:
            paths.append(text.split("ouviescrevi.pt", 1)[1])
        else:
            paths.append(text)
    return paths


def main() -> int:
    errors: list[str] = []

    for loc in REQUIRED_LOCALES:
        if loc == "pt":
            precos = FRONTEND / "precos.html"
        else:
            precos = FRONTEND / loc / "precos.html"
        if not precos.is_file():
            errors.append(f"Missing pricing page: {precos.relative_to(ROOT)}")

    for loc in REQUIRED_LOCALES:
        prefix = FRONTEND if loc == "pt" else FRONTEND / loc
        for slug in REQUIRED_AUDIENCE:
            page = prefix / f"{slug}.html"
            if loc in ("es", "fr", "de") or loc in ("pt", "en"):
                if not page.is_file():
                    errors.append(f"Missing audience landing: {page.relative_to(ROOT)}")

    for url_path in sitemap_paths():
        file_path = path_from_url_path(url_path)
        if not file_path.is_file():
            errors.append(f"Sitemap loc missing file: {url_path} → {file_path.relative_to(ROOT)}")

    checked_html = 0
    for html_file in sorted(FRONTEND.rglob("*.html")):
        if should_skip_html(html_file):
            continue
        text = html_file.read_text(encoding="utf-8", errors="replace")
        checked_html += 1
        if "<html" not in text.lower() or "<title" not in text.lower():
            errors.append(f"Invalid HTML skeleton: {html_file.relative_to(ROOT)}")
        for pattern in (SCRIPT_SRC, LINK_HREF):
            for ref in pattern.findall(text):
                if ref.startswith("#"):
                    continue
                resolved = resolve_asset(html_file, ref)
                if resolved is None:
                    continue
                try:
                    resolved.relative_to(FRONTEND)
                except ValueError:
                    continue
                if not resolved.is_file():
                    errors.append(f"Broken ref in {html_file.relative_to(ROOT)}: {ref}")

    print(f"Checked {checked_html} HTML files under frontend/")
    if errors:
        print(f"FAILED — {len(errors)} issue(s):", file=sys.stderr)
        for err in errors[:50]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more", file=sys.stderr)
        return 1

    print("OK — frontend smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
