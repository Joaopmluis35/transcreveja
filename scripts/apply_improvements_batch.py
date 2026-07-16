#!/usr/bin/env python3
"""Batch: cache-bust core assets + inject shared job/docx scripts + fix conversor race."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
ASSET_V = "25"

CORE_JS = (
    "ouviescrevi-ui.js",
    "ouviescrevi-api.js",
    "transcribe-jobs-ui.js",
    "docx-export.js",
    "suggestion-fab.js",
    "news-ticker.js",
    "pricing-visibility.js",
    "auth-ui.js",
    "upsell-ui.js",
    "history-ui.js",
    "media-trim-ui.js",
)
CORE_CSS = (
    "ouviescrevi.css",
    "index-home.css",
)


def bump_ref(html: str, filename: str, version: str) -> str:
    # href/src with optional existing ?v=
    pat = re.compile(
        rf"""((?:src|href)=["'])([^"']*?{re.escape(filename)})(?:\?v=[^"']*)?(["'])"""
    )

    def repl(m: re.Match) -> str:
        return f'{m.group(1)}{m.group(2)}?v={version}{m.group(3)}'

    return pat.sub(repl, html)


def ensure_script_before_ui(html: str, script_src: str) -> str:
    """Insert shared script tag before ouviescrevi-ui.js if missing."""
    if script_src.split("?")[0].split("/")[-1] in html and script_src.split("/")[-1].split("?")[0] in html:
        # already referenced
        base = Path(script_src).name.split("?")[0]
        if base in html:
            return html
    marker = re.search(
        r"""<script[^>]+src=["']([^"']*ouviescrevi-ui\.js[^"']*)["'][^>]*>\s*</script>""",
        html,
        re.I,
    )
    if not marker:
        return html
    prefix = marker.group(1)
    # derive relative path depth from how ui.js is referenced
    if prefix.startswith("../"):
        src = "../js/" + Path(script_src).name
    elif prefix.startswith("/js/") or prefix.startswith("/"):
        src = "/js/" + Path(script_src).name
    else:
        src = "js/" + Path(script_src).name
    if src.split("?")[0] in html or Path(script_src).name in html:
        # might already exist without version
        if Path(script_src).name in html:
            return html
    tag = f'<script src="{src}?v={ASSET_V}" defer></script>\n  '
    # if ui has defer, keep defer; else plain
    if "defer" not in marker.group(0):
        tag = f'<script src="{src}?v={ASSET_V}"></script>\n'
        return html[: marker.start()] + tag + marker.group(0) + html[marker.end() :]
    return html[: marker.start()] + tag + html[marker.start() :]


def fix_pdf_to_text(html: str) -> str:
    """Replace raced convertPDFtoText with sequential awaits (all locales)."""
    # Match the broken parallel pattern across PT/EN/ES/FR/DE variants
    pattern = re.compile(
        r"pdfjsLib\.getDocument\(\{\s*data:\s*typedarray\s*\}\)\.promise\.then\(function\s*\(pdf\)\s*\{\s*"
        r"let\s+textContent\s*=\s*\"\";\s*"
        r"let\s+totalPages\s*=\s*pdf\.numPages;\s*"
        r"let\s+processed\s*=\s*0;\s*"
        r"for\s*\(let\s+i\s*=\s*1;\s*i\s*<=\s*totalPages;\s*i\+\+\)\s*\{\s*"
        r"pdf\.getPage\(i\)\.then\(function\s*\(page\)\s*\{\s*"
        r"page\.getTextContent\(\)\.then\(function\s*\(content\)\s*\{\s*"
        r"const\s+strings\s*=\s*content\.items\.map\(function\s*\(item\)\s*\{\s*return\s+item\.str;\s*\}\)\.join\(\" \"\);\s*"
        r"textContent\s*\+=\s*strings\s*\+\s*\"\\n\\n\";\s*"
        r"processed\+\+;\s*"
        r"if\s*\(processed\s*===\s*totalPages\)\s*\{[\s\S]*?\}\s*"
        r"\}\);\s*"
        r"\}\);\s*"
        r"\}\s*"
        r"\}\);",
        re.M,
    )
    replacement = (
        'pdfjsLib.getDocument({ data: typedarray }).promise.then(async function (pdf) {\n'
        '        let textContent = "";\n'
        '        for (let i = 1; i <= pdf.numPages; i++) {\n'
        '          const page = await pdf.getPage(i);\n'
        '          const content = await page.getTextContent();\n'
        '          const strings = content.items.map(function (item) { return item.str; }).join(" ");\n'
        '          textContent += strings + "\\n\\n";\n'
        '        }\n'
        '        const blob = new Blob([textContent], { type: "text/plain;charset=utf-8" });\n'
        '        const url = URL.createObjectURL(blob);\n'
        '        const a = document.createElement("a");\n'
        '        a.href = url;\n'
        '        a.download = "ficheiro_extraido.txt";\n'
        '        a.click();\n'
        '        URL.revokeObjectURL(url);\n'
        '        setStatus("✅ Texto extraído!", "ok");\n'
        '      });'
    )
    new_html, n = pattern.subn(replacement, html, count=1)
    return new_html if n else html


def fix_pdf_to_word_docx(html: str) -> str:
    """Replace fake .doc blob with real DOCX via OuviescreviDocx."""
    old = re.compile(
        r'const\s+blob\s*=\s*new\s+Blob\(\[textContent\],\s*\{\s*type:\s*[\'"]application/msword;charset=utf-8[\'"]\s*\}\);\s*'
        r'const\s+url\s*=\s*URL\.createObjectURL\(blob\);\s*'
        r'const\s+a\s*=\s*document\.createElement\([\'"]a[\'"]\);\s*'
        r'a\.href\s*=\s*url;\s*'
        r'a\.download\s*=\s*[\'"]ficheiro_convertido\.doc[\'"];\s*'
        r'a\.click\(\);\s*'
        r'URL\.revokeObjectURL\(url\);\s*'
        r'setStatus\([^;]+;',
        re.M,
    )
    repl = (
        'if (window.OuviescreviDocx) {\n'
        '          OuviescreviDocx.exportLocalDocx(textContent, "ficheiro_convertido.docx", "Ouviescrevi");\n'
        '        } else {\n'
        '          const blob = new Blob([textContent], { type: "application/msword;charset=utf-8" });\n'
        '          const url = URL.createObjectURL(blob);\n'
        '          const a = document.createElement("a");\n'
        '          a.href = url;\n'
        '          a.download = "ficheiro_convertido.doc";\n'
        '          a.click();\n'
        '          URL.revokeObjectURL(url);\n'
        '        }\n'
        '        setStatus("✅ Conversão DOCX concluída!", "ok");'
    )
    new_html, n = old.subn(repl, html, count=1)
    return new_html if n else html


def inject_conversor_docx_script(html: str) -> str:
    if "docx-export.js" in html:
        return html
    # before first inline <script> after body tools, or after jspdf/pdfjs
    m = re.search(r'<script(?![^>]+src=)[^>]*>', html)
    if not m:
        return html
    # prefer relative js/
    prefix = "js/" if 'src="js/' in html or "src='js/" in html else "../js/"
    if "/en/" in str(html[:200]) or 'lang="en"' in html[:80]:
        pass
    tag = f'<script src="{prefix}docx-export.js?v={ASSET_V}"></script>\n'
    # detect path: if page has ../css then ../js
    if 'href="../css/' in html or 'src="../js/' in html:
        tag = f'<script src="../js/docx-export.js?v={ASSET_V}"></script>\n'
    elif 'href="css/' in html or 'src="js/' in html:
        tag = f'<script src="js/docx-export.js?v={ASSET_V}"></script>\n'
    return html[: m.start()] + tag + html[m.start() :]


def patch_locale_index_jobs(html: str) -> str:
    """After successful /transcribe response, poll job_id when present."""
    if "OuviescreviJobs" in html and "awaitTranscribeResult" in html:
        return html

    # Pattern used by ES/FR/DE (and similar)
    needle = "sucesso = transcricaoRecebidaComSucesso(data);"
    if needle in html and "data.job_id" not in html.split(needle)[0][-400:]:
        replacement = (
            "if (data.job_id && window.OuviescreviJobs) {\n"
            "          clearInterval(interval);\n"
            "          pararFrasesAnimadas(statusEl);\n"
            "          statusEl.textContent = \"✅ Ficheiro enviado — a transcrever no servidor…\";\n"
            "          progressBar.style.width = \"45%\";\n"
            "          try {\n"
            "            data = await OuviescreviJobs.awaitTranscribeResult(data, { statusEl: statusEl, progressBar: progressBar });\n"
            "            sucesso = transcricaoRecebidaComSucesso(data);\n"
            "          } catch (pollErr) {\n"
            "            data = { error: pollErr.message };\n"
            "            sucesso = false;\n"
            "          }\n"
            "          break;\n"
            "        }\n"
            "        sucesso = transcricaoRecebidaComSucesso(data);"
        )
        html = html.replace(needle, replacement, 1)

    # EN variant
    needle_en = "success=!!(data.formatted||data.transcription||data.text);"
    if needle_en in html and "data.job_id" not in html.split(needle_en)[0][-500:]:
        replacement_en = (
            "if (data.job_id && window.OuviescreviJobs) {\n"
            "          clearInterval(interval);\n"
            "          stopStatusAnimation(statusEl);\n"
            "          statusEl.textContent = \"✅ Uploaded — transcribing on server…\";\n"
            "          progressBar.style.width = \"45%\";\n"
            "          try {\n"
            "            data = await OuviescreviJobs.awaitTranscribeResult(data, { statusEl: statusEl, progressBar: progressBar });\n"
            "            success=!!(data.formatted||data.transcription||data.text);\n"
            "          } catch (pollErr) {\n"
            "            data = { error: pollErr.message };\n"
            "            success = false;\n"
            "          }\n"
            "          break;\n"
            "        }\n"
            "        success=!!(data.formatted||data.transcription||data.text);"
        )
        html = html.replace(needle_en, replacement_en, 1)

    # video-subs: after JSON parse + ok check, poll job
    # ES/FR/DE: if (!res.ok){ throw ... } then const srtUrl
    vs_pat = re.compile(
        r"(if\s*\(!res\.ok\)\s*\{?\s*throw new Error\(data\.detail \|\| data\.error \|\| `Erro \$\{res\.status\}`\);\s*\}?\s*)"
        r"(const srtUrl = toAbsUrl\(data\.srt_url\);)",
        re.M,
    )
    vs_repl = (
        r"\1"
        "if (data.job_id && window.OuviescreviJobs) {\n"
        "        data = await OuviescreviJobs.awaitVideoSubsResult(data, { statusEl: statusEl, progressBar: progressBar });\n"
        "      }\n"
        r"      \2"
    )
    html, _ = vs_pat.subn(vs_repl, html, count=1)

    # EN video-subs
    vs_en = re.compile(
        r"(if\s*\(!res\.ok\)\s*throw new Error\(data\.detail\|\|data\.error\|\|`Error \$\{res\.status\}`\);\s*)"
        r"(const srtUrl = toAbsUrl\(data\.srt_url\);)",
        re.M,
    )
    vs_en_repl = (
        r"\1"
        "if (data.job_id && window.OuviescreviJobs) {\n"
        "        data = await OuviescreviJobs.awaitVideoSubsResult(data, { statusEl: statusEl, progressBar: progressBar });\n"
        "      }\n"
        r"      \2"
    )
    html, _ = vs_en.subn(vs_en_repl, html, count=1)

    # append analytics fields after formData.append file
    if "appendAnalyticsFields" not in html:
        html = html.replace(
            'formData.append("file", file);',
            'formData.append("file", file);\n'
            '    if (window.OuviescreviJobs) OuviescreviJobs.appendAnalyticsFields(formData);',
        )
        html = html.replace(
            "formData.append(\"file\", file);",
            "formData.append(\"file\", file);\n"
            "    if (window.OuviescreviJobs) OuviescreviJobs.appendAnalyticsFields(formData);",
        )

    # EN style for video-subs
    if 'formData.append("token", API_TOKEN);' in html and 'formData.append("style"' not in html:
        html = html.replace(
            'formData.append("token", API_TOKEN);',
            'formData.append("token", API_TOKEN);\n'
            '    if (typeof getStyle === "function") formData.append("style", JSON.stringify(getStyle()));',
            1,
        )

    return html


def patch_speakers(html: str, lang: str) -> str:
    """Replace sync aplicarLocutores/applySpeakers with async diarize when possible."""
    # Make aplicarLocutores async wrapper - callers use `textoFinal = aplicarLocutores(...)`
    # Change callers to await if we convert to async.

    if "diarizeSpeakers" in html:
        return html

    # Replace function body for PT-style aplicarLocutores
    old_fn = re.compile(
        r"function aplicarLocutores\(transcricao\)\{[\s\S]*?return resultado\.trim\(\);\s*\}",
        re.M,
    )
    names_lit = {
        "pt": '["João","Maria"]',
        "es": '["Juan","María"]',
        "fr": '["Jean","Marie"]',
        "de": '["Hans","Maria"]',
        "en": '["John","Mary"]',
    }.get(lang, '["João","Maria"]')

    new_fn = (
        "async function aplicarLocutores(transcricao){\n"
        "    if (window.OuviescreviJobs) {\n"
        f"      return OuviescreviJobs.diarizeSpeakers(transcricao, {{ names: {names_lit}, lang: \"{lang}\" }});\n"
        "    }\n"
        f"    return (window.OuviescreviJobs ? OuviescreviJobs.applyAlternatingSpeakers(transcricao, {names_lit}) : transcricao);\n"
        "  }"
    )
    html, n = old_fn.subn(new_fn, html, count=1)

    old_en = re.compile(
        r"function applySpeakers\([^{]+\{[\s\S]*?return resultado\.trim\(\);\s*\}|"
        r"function applySpeakers\([^{]+\{[\s\S]*?return finalText\.trim\(\);\s*\}",
        re.M,
    )
    # EN applySpeakers often similar
    old_en2 = re.compile(
        r"function applySpeakers\(transcricao\)\{[\s\S]*?return resultado\.trim\(\);\s*\}",
        re.M,
    )
    new_en = (
        "async function applySpeakers(transcricao){\n"
        "    if (window.OuviescreviJobs) {\n"
        '      return OuviescreviJobs.diarizeSpeakers(transcricao, { names: ["John","Mary"], lang: "en" });\n'
        "    }\n"
        '    return OuviescreviJobs.applyAlternatingSpeakers(transcricao, ["John","Mary"]);\n'
        "  }"
    )
    html, n2 = old_en2.subn(new_en, html, count=1)

    # await aplicarLocutores / applySpeakers calls
    html = re.sub(
        r"(?<!await )(textoFinal\s*=\s*aplicarLocutores\(textoFinal\))",
        r"textoFinal = await aplicarLocutores(textoFinal)",
        html,
    )
    html = re.sub(
        r"(?<!await )(finalText\s*=\s*applySpeakers\(finalText\))",
        r"finalText = await applySpeakers(finalText)",
        html,
    )
    return html


def process_html(path: Path) -> bool:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")
    html = raw
    for name in CORE_JS + CORE_CSS:
        html = bump_ref(html, name, ASSET_V)

    rel = path.relative_to(FRONTEND).as_posix()
    is_index = path.name == "index.html"
    is_conversor = path.name == "conversor.html"
    is_gerar = path.name == "gerar-video.html"

    if is_index or is_gerar:
        html = ensure_script_before_ui(html, "transcribe-jobs-ui.js")
        html = ensure_script_before_ui(html, "docx-export.js")
        # bump again after inject
        for name in ("transcribe-jobs-ui.js", "docx-export.js"):
            html = bump_ref(html, name, ASSET_V)

    if is_index:
        lang = "pt"
        if "/en/" in rel or rel.startswith("en/"):
            lang = "en"
        elif "/es/" in rel or rel.startswith("es/"):
            lang = "es"
        elif "/fr/" in rel or rel.startswith("fr/"):
            lang = "fr"
        elif "/de/" in rel or rel.startswith("de/"):
            lang = "de"
        if lang != "pt":
            html = patch_locale_index_jobs(html)
        html = patch_speakers(html, lang)

    if is_conversor:
        html = inject_conversor_docx_script(html)
        html = fix_pdf_to_text(html)
        html = fix_pdf_to_word_docx(html)
        html = bump_ref(html, "docx-export.js", ASSET_V)

    if html != raw:
        path.write_text(html, encoding="utf-8", newline="\n")
        return True
    return False


def update_generators() -> None:
    for script in (
        ROOT / "scripts" / "gen_locale_pages.py",
        ROOT / "scripts" / "generate_locales.py",
    ):
        if not script.exists():
            continue
        text = script.read_text(encoding="utf-8")
        orig = text
        text = re.sub(
            r"ouviescrevi-ui\.js(?:\?v=\d+)?",
            f"ouviescrevi-ui.js?v={ASSET_V}",
            text,
        )
        text = re.sub(
            r"ouviescrevi\.css\?v=\d+",
            f"ouviescrevi.css?v={ASSET_V}",
            text,
        )
        text = re.sub(
            r"index-home\.css\?v=\d+",
            f"index-home.css?v={ASSET_V}",
            text,
        )
        if text != orig:
            script.write_text(text, encoding="utf-8", newline="\n")
            print(f"updated generator {script.name}")


def main() -> None:
    changed = 0
    for path in sorted(FRONTEND.rglob("*.html")):
        if "archive" in path.parts:
            continue
        if process_html(path):
            changed += 1
            print(f"updated {path.relative_to(ROOT)}")
    update_generators()
    print(f"done: {changed} html files, ASSET_V={ASSET_V}")


if __name__ == "__main__":
    main()
