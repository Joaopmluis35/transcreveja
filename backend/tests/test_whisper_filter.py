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


def test_is_music_only_transcript():
    assert app_main.is_music_only_transcript("[00:00] Música")
    assert app_main.is_music_only_transcript("Música")
    assert app_main.is_music_only_transcript("[Music]")
    assert not app_main.is_music_only_transcript("[00:00] Música\n\n[00:12] Olá a todos")
    assert not app_main.is_music_only_transcript("")


def test_filter_drops_pt_silence_slogans():
    segs = [
        _seg("A CIDADE NO BRASIL"),
        _seg("A CIDADE NO BRASILEIRO"),
        _seg("Obrigado por assistir"),
        _seg("Bom dia, vamos começar o treino de hoje na sala."),
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    assert len(kept) == 1
    assert "treino" in kept[0]["text"].lower()


def test_clean_removes_cidade_hallucination_blocks():
    text = (
        "[00:00] A CIDADE NO BRASIL\n\n"
        "[10:00] A CIDADE NO BRASILEIRO"
    )
    cleaned = app_main.clean_transcription_text(text, language="pt")
    assert not cleaned.strip()


def test_whisper_prompt_mentions_lyrics():
    pt = app_main.whisper_prompt_for_language("pt")
    assert pt and "letra cantada" in pt.lower()
    assert "não substituir" in pt.lower()
    assert "reunião de trabalho" not in pt.lower()


def test_music_tag_does_not_hide_lyrics():
    segs = [
        _seg("Música", no_speech=0.05),
        _seg("Canta", no_speech=0.95, logprob=-1.4, compression=1.2),
        _seg("A CIDADE NO BRASIL", no_speech=0.2),
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    texts = " ".join(s["text"] for s in kept).lower()
    assert "canta" in texts
    assert "cidade" not in texts
    assert "música" not in texts


def test_raw_segments_cidade_are_unusable():
    segs = [
        _seg("A CIDADE NO BRASIL"),
        _seg("A CIDADE NO BRASILEIRO"),
        _seg("Música"),
    ]
    assert app_main.raw_segments_are_unusable(segs, language="pt") is True
    segs.append(_seg("Bom dia, vamos começar o treino."))
    assert app_main.raw_segments_are_unusable(segs, language="pt") is False


def test_salvage_keeps_lyrics_when_metrics_wipe_all():
    """Música com letra: no_speech e compression extremos não podem zerar o texto."""
    segs = [
        _seg("Vamos todos", no_speech=0.97, logprob=-2.1, compression=3.4),
        _seg("Vamos com tudo agora", no_speech=0.96, logprob=-1.9, compression=3.2),
        _seg("A CIDADE NO BRASIL", no_speech=0.99, compression=4.0),
        _seg("「」" * 12, no_speech=0.2),
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    texts = " ".join(s["text"] for s in kept).lower()
    assert "vamos todos" in texts
    assert "vamos com tudo" in texts
    assert "cidade" not in texts


def test_relaxed_keeps_few_narration_segs_when_strict_empty():
    """Música+voz: poucos segs, no_speech alto e compression que falha no strict."""
    segs = [
        _seg(
            "Vamos começar o aquecimento na sala de fitness.",
            no_speech=0.78,
            compression=2.55,
        ),
        _seg("A CIDADE NO BRASIL", no_speech=0.95),
    ]
    kept = app_main.filter_whisper_segments(segs, language="pt")
    assert len(kept) == 1
    assert "aquecimento" in kept[0]["text"].lower()
