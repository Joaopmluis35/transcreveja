"""Compara modelos de transcrição da OpenAI com o mesmo áudio (custa alguns cêntimos por ficheiro).

Uso:  py scripts/compare_transcribe_models.py caminho\audio.mp3 [pt]
Lê OPENAI_API_KEY de backend/.env. Grava cada resultado em scripts/compare_out/<modelo>.txt.

Nota: o site usa os tempos (segmentos) do whisper-1 para legendas, capítulos e filtros de qualidade.
Os gpt-4o-transcribe* não devolvem segmentos, por isso trocar de modelo no site não é só mudar o nome.
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
# Preço por minuto (USD), tabela OpenAI — confirmar em platform.openai.com/pricing antes de decidir.
PRICE_PER_MIN = {"whisper-1": 0.006, "gpt-4o-transcribe": 0.006, "gpt-4o-mini-transcribe": 0.003}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    audio = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "pt"
    load_dotenv(ROOT / "backend" / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Falta OPENAI_API_KEY em backend/.env")
    client = OpenAI()
    out_dir = ROOT / "scripts" / "compare_out"
    out_dir.mkdir(exist_ok=True)
    for model in MODELS:
        t0 = time.monotonic()
        try:
            with open(audio, "rb") as f:
                r = client.audio.transcriptions.create(model=model, file=f, language=lang, response_format="text")
        except Exception as e:  # um modelo falhar não impede os outros
            print(f"{model:24} ERRO: {str(e)[:200]}")
            continue
        text = r if isinstance(r, str) else getattr(r, "text", str(r))
        (out_dir / f"{model}.txt").write_text(text, encoding="utf-8")
        print(f"{model:24} {time.monotonic() - t0:6.1f}s  {len(text.split()):6} palavras  ~{PRICE_PER_MIN[model]}$/min")
    print(f"\nTextos em {out_dir} — compara-os lado a lado.")


if __name__ == "__main__":
    main()
