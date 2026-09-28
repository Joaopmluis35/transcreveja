"""Testes do filtro de alucinações Whisper."""
from __future__ import annotations

import main as app_main


def _seg(text: str, *, no_speech: float = 0.1, logprob: float = -0.2, compression: float = 1.2):
    return {
        "text": text,
        "no_speech_prob": no_speech,
        "avg_logprob": logprob,
        "compression_ratio": compression,
        "start": 0,
        "end": 1,
    }


def test_filter_drops_cjk_bracket_spam():
    segs = [
        _seg("「」「」「」「」「」「」「」「」「」「」"),
        _seg("Pronto, e agora para já faltava-nos uma página."),
    ]
    kept = app_main.filter_whisper_segments(segs, language=None)
    assert len(kept) == 1
    assert "página" in kept[0]["text"]


def test_filter_drops_ok_loops():
    segs = [
        _seg("ok, ok, ok, ok, ok, ok, ok, ok, ok, ok, ok, ok"),
        _seg("vamos fazendo os ajustes, não há problema"),
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    assert len(kept) == 1
    assert "ajustes" in kept[0]["text"]


def test_filter_does_not_restore_all_when_mostly_noise():
    segs = [_seg("「」" * 20) for _ in range(10)]
    segs.append(_seg("Olá, esta é fala real em português."))
    kept = app_main.filter_whisper_segments(segs, language=None)
    assert len(kept) == 1
    assert "português" in kept[0]["text"].lower() or "Olá" in kept[0]["text"]


def test_clean_collapses_ok_loop_in_block():
    text = (
        "[11:08] Não, não, esse não, só o Bufabi, sim, uhum, ok, ok, ok, ok, ok, ok, ok, ok\n\n"
        "[11:36] ok, ok, ok, ok, ok, ok, ok, ok, ok, ok, ok, ok\n\n"
        "[11:03] porque assim que for necessário fazemos."
    )
    cleaned = app_main.clean_transcription_text(text, language=None)
    assert cleaned.count("ok") <= 3
    assert "necessário" in cleaned or "Bufabi" in cleaned


def test_clean_removes_cyrillic_stub():
    text = "[20:00] Да.\n\n[10:00] Pronto, e agora para já."
    cleaned = app_main.clean_transcription_text(text, language="pt")
    assert "Да" not in cleaned
    assert "Pronto" in cleaned


def test_is_hallucination_bracket_spam():
    assert app_main._is_hallucination_text("「」「」「」「」「」「」「」")
    assert not app_main._is_hallucination_text(
        "Pronto, e agora para já, segundo as últimas atualizações."
    )
