#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCALE_NEEDLE = """  function exportText(type){
    const text = transcriptionText.value || "";
    if (!text) return;

    if (type === 'pdf'){"""

LOCALE_REPL = """  async function exportText(type){
    const text = transcriptionText.value || "";
    if (!text) return;

    if (type === 'docx' || type === 'doc'){
      if (window.OuviescreviDocx) {
        await OuviescreviDocx.exportDocxPro(text, { title: 'Ouviescrevi', filename: 'FILENAME', allowLocalFallback: true });
        return;
      }
    }

    if (type === 'pdf'){"""

EN_NEEDLE = """  function exportText(type){
    const text=transcriptionText.value||"";
    if (!text) return;
    if (type==='pdf'){"""

EN_REPL = """  async function exportText(type){
    const text=transcriptionText.value||"";
    if (!text) return;
    if (type==='docx' || type==='doc'){
      if (window.OuviescreviDocx) {
        await OuviescreviDocx.exportDocxPro(text, { title: 'Ouviescrevi', filename: 'transcription.docx', allowLocalFallback: true });
        return;
      }
    }
    if (type==='pdf'){"""


def main() -> None:
    for lang, fname in (
        ("es", "transcripcion.docx"),
        ("fr", "transcription.docx"),
        ("de", "transkript.docx"),
    ):
        path = ROOT / "frontend" / lang / "index.html"
        text = path.read_text(encoding="utf-8")
        repl = LOCALE_REPL.replace("FILENAME", fname)
        if LOCALE_NEEDLE not in text:
            print("miss", lang)
            continue
        path.write_text(text.replace(LOCALE_NEEDLE, repl, 1), encoding="utf-8", newline="\n")
        print("ok", lang)

    en = ROOT / "frontend" / "en" / "index.html"
    text = en.read_text(encoding="utf-8")
    if EN_NEEDLE not in text:
        print("miss en")
        return
    en.write_text(text.replace(EN_NEEDLE, EN_REPL, 1), encoding="utf-8", newline="\n")
    print("ok en")


if __name__ == "__main__":
    main()
