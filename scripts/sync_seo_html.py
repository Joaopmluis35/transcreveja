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


def upsert_head(html: str, title: str, description: str) -> str:
    out = html
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
    return out


updated = 0
for html_path in ROOT.rglob("*.html"):
    if html_path.name in SKIP or "archive" in html_path.parts:
        continue
    rel = "/" + html_path.relative_to(ROOT).as_posix()
    cfg = pages.get(rel)
    if not cfg:
        continue
    html = html_path.read_text(encoding="utf-8", errors="replace")
    new_html = upsert_head(html, cfg["title"], cfg["description"])
    if new_html != html:
        html_path.write_text(new_html, encoding="utf-8")
        print("updated", rel)
        updated += 1

print(f"done — {updated} files updated, {len(pages)} pages in config")
