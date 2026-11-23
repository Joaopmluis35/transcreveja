"""Textos longos nas ferramentas de IA: partir sem perder nada, limite claro e erro em linguagem simples."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import main as app_main

API_TOKEN = "test-api-token"


def _mock_completion(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_split_short_text_is_one_batch():
    assert app_main.split_text_batches("olá", 100) == ["olá"]


def test_split_keeps_all_words_and_respects_size():
    paras = [f"[{i:02d}:00] " + "palavra " * 60 for i in range(40)]
    text = "\n\n".join(p.strip() for p in paras)
    batches = app_main.split_text_batches(text, 2000)
    assert len(batches) > 1
    assert all(len(b) <= 2000 for b in batches)
    assert "\n\n".join(batches).split() == text.split()


def test_split_cuts_one_huge_line_by_sentences():
    text = "Uma frase curta. " * 1000
    batches = app_main.split_text_batches(text, 3000)
    assert all(len(b) <= 3000 for b in batches)
    assert "\n\n".join(batches).split() == text.split()


def test_correct_processes_every_part_of_a_long_text(client):
    text = "\n\n".join(f"[{i:02d}:00] Frase numero {i} com erros." for i in range(900))
    assert len(text) > 6000
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"][1]["content"])
        return _mock_completion(kwargs["messages"][1]["content"].upper())

    with patch.object(app_main.ai_chat, "create", side_effect=fake_create):
        res = client.post("/correct", json={"text": text, "token": API_TOKEN, "lang": "pt"})
    assert res.status_code == 200
    assert len(calls) > 1
    corrected = res.json()["corrected"]
    assert "FRASE NUMERO 899" in corrected  # o fim não se perde


def test_ai_text_too_long_gets_clear_message(client, monkeypatch):
    monkeypatch.setattr(app_main, "AI_MAX_INPUT_CHARS", 1000)
    res = client.post("/correct", json={"text": "a" * 1500, "token": API_TOKEN, "lang": "pt"})
    assert res.status_code == 413
    assert "demasiado longo" in res.json()["detail"]


def test_ai_failure_shows_simple_message(client):
    with patch.object(app_main.ai_chat, "create", side_effect=RuntimeError("Error code: 429 insufficient_quota sk-xyz")):
        res = client.post("/summarize", json={"text": "Um texto qualquer para resumir.", "token": API_TOKEN, "lang": "pt"})
    assert res.status_code == 500
    detail = res.json()["detail"]
    assert detail == app_main.AI_FRIENDLY_ERROR
    assert "insufficient_quota" not in detail
