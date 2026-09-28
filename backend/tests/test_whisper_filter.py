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


def test_looks_like_real_speech_keeps_quiet_pt():
    assert app_main._looks_like_real_speech(
        "Nós entretanto passámos o projeto todo para o staging.",
        "pt",
    )
    assert not app_main._looks_like_real_speech("「」「」「」「」「」", "pt")


def test_resolve_auto_defaults_to_pt_on_pt_ui():
    assert app_main.resolve_whisper_language(None, "pt") == "pt"
    assert app_main.resolve_whisper_language("auto", None) == "pt"
    assert app_main.resolve_whisper_language("auto", "en") == "en"
    assert app_main.resolve_whisper_language("es", "pt") == "es"


def test_filter_keeps_real_pt_despite_high_no_speech():
    segs = [
        {
            "text": "Criámos aqui um subdomínio que é o staging.lin.org.pt",
            "no_speech_prob": 0.72,
            "avg_logprob": -1.2,
            "compression_ratio": 1.3,
            "start": 0,
            "end": 2,
        },
        {
            "text": "ok, ok, ok, ok, ok, ok, ok, ok",
            "no_speech_prob": 0.2,
            "avg_logprob": -0.2,
            "compression_ratio": 1.1,
            "start": 2,
            "end": 3,
        },
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    assert len(kept) == 1
    assert "staging" in kept[0]["text"]
