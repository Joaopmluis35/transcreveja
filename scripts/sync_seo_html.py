"""Sync static <title> and meta description from ouviescrevi-seo.js PAGES."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "frontend"
SEO_FILE = ROOT / "js" / "ouviescrevi-seo.js"
SKIP = {
    "backoffice.html", "admin.html", "index2.html",
    "exemplo-interface-profissional.html", "exemplo-logo-profissional.html",
    "header.html", "footer.html",
}

src = SEO_FILE.read_text(encoding="utf-8")
block = re.search(r"var PAGES = \{([\s\S]*?)\n  \};", src)
if not block:
    raise SystemExit("PAGES not found")

pages = {}
for m in re.finditer(r'"(/[^"]+)":\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}', block.group(1)):
    path, body = m.group(1), m.group(2)
    title_m = re.search(r'title:\s*"([^"]+)"', body)
    desc_m = re.search(r'description:\s*(?:\n\s*)?"([^"]+)"', body)
    if title_m:
        pages[path] = {
            "title": title_m.group(1),
            "description": desc_m.group(1) if desc_m else "",
        }


def og_image_for(path: str) -> str:
    site = "https://www.ouviescrevi.pt"
    default = f"{site}/og/index.png"
    mapping = {
        "/aulas.html": f"{site}/og/aulas.png",
        "/en/aulas.html": f"{site}/og/aulas.png",
        "/es/aulas.html": f"{site}/og/aulas.png",
        "/fr/aulas.html": f"{site}/og/aulas.png",
        "/de/aulas.html": f"{site}/og/aulas.png",
        "/professores.html": f"{site}/og/professores.png",
        "/en/professores.html": f"{site}/og/professores.png",
        "/es/professores.html": f"{site}/og/professores.png",
        "/fr/professores.html": f"{site}/og/professores.png",
        "/de/professores.html": f"{site}/og/professores.png",
        "/podcasts.html": f"{site}/og/podcasts.png",
        "/en/podcasts.html": f"{site}/og/podcasts.png",
        "/es/podcasts.html": f"{site}/og/podcasts.png",
        "/fr/podcasts.html": f"{site}/og/podcasts.png",
        "/de/podcasts.html": f"{site}/og/podcasts.png",
        "/reunioes.html": f"{site}/og/reunioes.png",
        "/en/reunioes.html": f"{site}/og/reunioes.png",
        "/es/reunioes.html": f"{site}/og/reunioes.png",
        "/fr/reunioes.html": f"{site}/og/reunioes.png",
        "/de/reunioes.html": f"{site}/og/reunioes.png",
        "/jornalistas.html": f"{site}/og/jornalistas.png",
        "/en/jornalistas.html": f"{site}/og/jornalistas.png",
        "/es/jornalistas.html": f"{site}/og/jornalistas.png",
        "/fr/jornalistas.html": f"{site}/og/jornalistas.png",
        "/de/jornalistas.html": f"{site}/og/jornalistas.png",
        "/precos.html": f"{site}/og/precos.png",
        "/en/precos.html": f"{site}/og/precos.png",
        "/es/precos.html": f"{site}/og/precos.png",
        "/fr/precos.html": f"{site}/og/precos.png",
        "/de/precos.html": f"{site}/og/precos.png",
        "/partilha.html": f"{site}/og/partilha.png",
        "/blog/index.html": f"{site}/og/blog.png",
        "/index.html": default,
        "/en/index.html": default,
        "/es/index.html": default,
        "/fr/index.html": default,
        "/de/index.html": default,
    }
    if path.startswith("/blog/"):
        return f"{site}/og/blog.png"
    return mapping.get(path, default)


def upsert_head(html: str, title: str, description: str, path: str = "") -> str:
    out = html
    site = "https://www.ouviescrevi.pt"
    canonical = f"{site}{path}" if path else ""
    og_image = og_image_for(path) if path else f"{site}/og/index.png"
    if title:
        if re.search(r"<title>[^<]*</title>", out, re.I):
            out = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", out, count=1, flags=re.I)
        else:
            out = out.replace("<head>", f"<head>\n  <title>{title}</title>", 1)
    if description:
        esc = description.replace('"', "&quot;")
        meta = f'<meta name="description" content="{esc}">'
        if re.search(r'<meta\s+name="description"', out, re.I):
            first = True
            def repl(_m):
                nonlocal first
                if first:
                    first = False
                    return meta
                return ""
            out = re.sub(r'<meta\s+name="description"\s+content="[^"]*"\s*/?>', repl, out, flags=re.I)
        else:
            insert_after = re.search(r'<meta[^>]+viewport[^>]*>', out, re.I)
            if insert_after:
                pos = insert_after.end()
                out = out[:pos] + f"\n  {meta}" + out[pos:]
            else:
                out = out.replace("<head>", f"<head>\n  {meta}", 1)

    def upsert_meta(attr: str, key: str, value: str) -> None:
        nonlocal out
        if not value:
            return
        esc = value.replace('"', "&quot;")
        pattern = rf'<meta\s+{attr}="{re.escape(key)}"\s+content="[^"]*"\s*/?>'
        tag = f'<meta {attr}="{key}" content="{esc}">'
        if re.search(pattern, out, re.I):
            out = re.sub(pattern, tag, out, count=1, flags=re.I)
        else:
            insert_after = re.search(r'<meta\s+name="description"[^>]*>', out, re.I) or re.search(
                r"<title>[^<]*</title>", out, re.I
            )
            if insert_after:
                pos = insert_after.end()
                out = out[:pos] + f"\n  {tag}" + out[pos:]

    def upsert_link_rel(rel: str, href: str) -> None:
        nonlocal out
        if not href:
            return
        pattern = rf'<link\s+rel="{re.escape(rel)}"\s+href="[^"]*"\s*/?>'
        tag = f'<link rel="{rel}" href="{href}">'
        if re.search(pattern, out, re.I):
            out = re.sub(pattern, tag, out, count=1, flags=re.I)
        else:
            insert_after = re.search(r"<title>[^<]*</title>", out, re.I)
            if insert_after:
                pos = insert_after.end()
                out = out[:pos] + f"\n  {tag}" + out[pos:]

    if canonical:
        upsert_link_rel("canonical", canonical)
        upsert_meta("property", "og:url", canonical)
    if title:
        upsert_meta("property", "og:title", title)
        upsert_meta("name", "twitter:title", title)
    if description:
        upsert_meta("property", "og:description", description)
        upsert_meta("name", "twitter:description", description)
    upsert_meta("property", "og:image", og_image)
    upsert_meta("property", "og:type", "website")
    upsert_meta("name", "twitter:card", "summary_large_image")
    upsert_link_rel("manifest", "/manifest.webmanifest")
    return out


updated = 0
for html_path in ROOT.rglob("*.html"):
    if html_path.name in SKIP or "archive" in html_path.parts:
        continue
    rel = "/" + html_path.relative_to(ROOT).as_posix()
    cfg = pages.get(rel)
    if not cfg:
        # Still inject OG for blog pages with existing title/description
        html = html_path.read_text(encoding="utf-8", errors="replace")
        title_m = re.search(r"<title>([^<]*)</title>", html, re.I)
        desc_m = re.search(r'<meta\s+name="description"\s+content="([^"]*)"', html, re.I)
        if not title_m:
            continue
        new_html = upsert_head(
            html,
            title_m.group(1),
            desc_m.group(1) if desc_m else "",
            rel,
        )
    else:
        html = html_path.read_text(encoding="utf-8", errors="replace")
        new_html = upsert_head(html, cfg["title"], cfg["description"], rel)
    if new_html != html:
        html_path.write_text(new_html, encoding="utf-8")
        print("updated", rel)
        updated += 1

print(f"done — {updated} files updated, {len(pages)} pages in config")
