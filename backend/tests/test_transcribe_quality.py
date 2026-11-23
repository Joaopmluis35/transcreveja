"""Aviso de áudio difícil, tempos de um trecho e quota reservada (revisão de 23/11/2026)."""
from __future__ import annotations

import main as app_main


def _seg(text: str, *, no_speech: float = 0.1, logprob: float = -0.2):
    return {"text": text, "no_speech_prob": no_speech, "avg_logprob": logprob, "compression_ratio": 1.2}


def test_clear_speech_is_not_low_confidence():
    segs = [_seg("Bom dia a todos, vamos começar."), _seg("Hoje falamos do orçamento.", logprob=-0.4)]
    assert app_main.audio_is_low_confidence(segs) is False


def test_noisy_shouting_is_low_confidence():
    # Caso do vídeo com gritos (23/11): frases com ar normal mas confiança muito baixa.
    segs = [_seg("É O LOCAL DO LOCAL!", logprob=-1.3), _seg("É CARACA!", logprob=-1.1, no_speech=0.4)]
    assert app_main.audio_is_low_confidence(segs) is True


def test_mostly_silence_is_low_confidence():
    assert app_main.audio_is_low_confidence([_seg("Obrigado.", no_speech=0.8, logprob=-0.5)]) is True


def test_low_confidence_ignores_empty_segments():
    assert app_main.audio_is_low_confidence([]) is False
    assert app_main.audio_is_low_confidence([_seg("   ", logprob=-3)]) is False


def test_start_offset_uses_trim_or_browser_cut():
    assert app_main._start_offset_seconds(90, None) == 90
    assert app_main._start_offset_seconds(None, "125.7") == 125
    assert app_main._start_offset_seconds(None, None) == 0
    assert app_main._start_offset_seconds(None, "abc") == 0
    assert app_main._start_offset_seconds(None, "-5") == 0


def test_count_active_jobs_only_counts_processing_of_same_key():
    with app_main._transcribe_jobs_lock:
        app_main._transcribe_jobs["t1"] = {"status": "processing", "usage_key": "ip:abc"}
        app_main._transcribe_jobs["t2"] = {"status": "completed", "usage_key": "ip:abc"}
        app_main._transcribe_jobs["t3"] = {"status": "processing", "usage_key": "ip:outro"}
    try:
        assert app_main._count_active_transcribe_jobs("ip:abc") == 1
        assert app_main._count_active_transcribe_jobs(None) == 0
    finally:
        with app_main._transcribe_jobs_lock:
            for k in ("t1", "t2", "t3"):
                app_main._transcribe_jobs.pop(k, None)


def test_long_recordings_keep_timestamps_over_100_minutes():
    # [105:00] tem 3 dígitos: os filtros só aceitavam [99:59].
    assert app_main._format_time(6300) == "[105:00]"
    assert app_main._TS_BLOCK_RE.match("[105:00] Texto")
    assert app_main._TIMESTAMP_PREFIX.match("[105:00] Texto")


def test_ffmpeg_timeout_grows_with_duration():
    assert app_main.ffmpeg_timeout_for(None) == app_main.FFMPEG_TIMEOUT
    assert app_main.ffmpeg_timeout_for(3600) > app_main.FFMPEG_TIMEOUT
    assert app_main.ffmpeg_timeout_for(10 ** 6) == 900
