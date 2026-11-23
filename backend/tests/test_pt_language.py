"""Site PT: fala em português detetada como inglês é repetida em pt; áudio noutra língua fica na língua original."""
from __future__ import annotations

import os
import tempfile
import uuid
from types import SimpleNamespace

import pytest

import main as app_main


def _seg(text: str, start: float, logprob: float):
    return {
        "text": text,
        "start": start,
        "end": start + 4,
        "no_speech_prob": 0.05,
        "avg_logprob": logprob,
        "compression_ratio": 1.3,
    }


EN_SEGS_LOW = [_seg(" hello João, how are you?", 0, -0.7), _seg(" good morning, can you hear me well?", 5, -0.75)]
PT_SEGS_HIGH = [_seg(" olá João, como estás?", 0, -0.25), _seg(" bom dia, consegues ouvir-me bem?", 5, -0.3)]
EN_SEGS_HIGH = [_seg(" hello João, how are you?", 0, -0.2), _seg(" good morning, can you hear me well?", 5, -0.25)]
PT_SEGS_LOW = [_seg(" olá João, como estás?", 0, -0.9), _seg(" bom dia, consegues ouvir-me bem?", 5, -0.95)]


def _run_job(monkeypatch, first, pt_retry, *, whisper_lang=None, ui_locale="pt", translate=None):
    calls = []

    def fake_whisper(path, retries=3, sleep_base=1.0, timeout=0, language=None, temperature=None, skip_language=False):
        calls.append(language)
        if language == "pt" and whisper_lang != "pt":
            return SimpleNamespace(segments=pt_retry, language="portuguese")
        return SimpleNamespace(segments=first[0], language=first[1])

    translated = []

    def fake_translate(formatted):
        translated.append(formatted)
        if translate is None:
            raise AssertionError("não devia traduzir")
        return translate(formatted)

    monkeypatch.setattr(app_main, "transcrever_parte_c_com_retries", fake_whisper)
    monkeypatch.setattr(app_main, "translate_formatted_transcript", fake_translate)
    monkeypatch.setattr(app_main, "safe_run_ffmpeg", lambda *a, **k: None)
    monkeypatch.setattr(app_main, "split_audio", lambda src, out_dir: [src])
    monkeypatch.setattr(app_main, "trim_leading_silence_wav", lambda p: (p, 0.0))
    monkeypatch.setattr(app_main, "probe_media_duration_sec", lambda p: 10.0)

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.write(fd, b"RIFF0000WAVE")
    os.close(fd)
    job_id = uuid.uuid4().hex
    app_main._transcribe_job_set(job_id, status="processing", message="teste")
    app_main._execute_transcribe_job(
        job_id, "test", tmp_path, "reuniao.mp4", 12, whisper_lang,
        {"type": "anonymous"}, f"test-{job_id}", False, "teste", ui_locale=ui_locale,
    )
    return app_main._transcribe_job_get(job_id), calls, translated


def test_portuguese_speech_detected_as_english_is_redone_in_portuguese(monkeypatch):
    job, calls, translated = _run_job(monkeypatch, (EN_SEGS_LOW, "english"), PT_SEGS_HIGH)
    assert job["status"] == "completed"
    assert calls == [None, "pt"]
    assert "olá João" in job["formatted"]
    assert "hello" not in job["formatted"]
    assert job.get("original_formatted") is None
    assert translated == []


def test_real_english_audio_stays_in_english(monkeypatch):
    """Pedido do João (23/11/2026): a transcrição fica na língua original; traduzir é com o botão «Traduzir»."""
    job, calls, translated = _run_job(monkeypatch, (EN_SEGS_HIGH, "english"), PT_SEGS_LOW)
    assert job["status"] == "completed"
    assert calls == [None, "pt"]
    assert "hello João" in job["formatted"] and "olá" not in job["formatted"]
    assert translated == []
    assert job.get("original_formatted") is None
    assert not job.get("warning")


SONG_EN = [_seg(" I remember, don't worry", 0, -0.6), _seg(" how could I ever forget", 5, -0.65)]
SONG_FORCED_PT = [_seg(" E eu estou esperando por isso o tempo todo", 0, -0.58), _seg(" por toda a minha vida", 5, -0.62)]


def test_english_song_is_not_rewritten_in_portuguese(monkeypatch):
    """Música em inglês: forçar português dá frases inventadas com confiança parecida — fica o inglês."""
    job, calls, _ = _run_job(monkeypatch, (SONG_EN, "english"), SONG_FORCED_PT)
    assert calls == [None, "pt"]
    assert "I remember" in job["formatted"]
    assert "esperando" not in job["formatted"]


def test_explicit_english_choice_is_respected(monkeypatch):
    job, calls, translated = _run_job(monkeypatch, (EN_SEGS_HIGH, "english"), PT_SEGS_LOW, whisper_lang="en")
    assert calls == ["en"]
    assert "hello João" in job["formatted"]
    assert translated == []


def test_english_site_keeps_audio_language(monkeypatch):
    job, calls, translated = _run_job(monkeypatch, (EN_SEGS_HIGH, "english"), PT_SEGS_LOW, ui_locale="en")
    assert calls == [None]
    assert "hello João" in job["formatted"]
    assert translated == []


def test_translate_keeps_timestamps(monkeypatch):
    def fake_create(**kwargs):
        text = kwargs["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text.replace("hello", "olá")))])

    monkeypatch.setattr(app_main.client.chat.completions, "create", fake_create)
    out = app_main.translate_formatted_transcript("[00:01] hello\n\n[00:05] hello again")
    assert out == "[00:01] olá\n\n[00:05] olá again"


def test_translate_rejects_lost_timestamps(monkeypatch):
    def fake_create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="olá, olá outra vez"))])

    monkeypatch.setattr(app_main.client.chat.completions, "create", fake_create)
    with pytest.raises(ValueError):
        app_main.translate_formatted_transcript("[00:01] hello\n\n[00:05] hello again")
