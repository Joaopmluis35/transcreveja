"""Testes dos endpoints de IA — validação e respostas mockadas (sem OpenAI real)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

API_TOKEN = "test-api-token"
LONG_TEXT = (
    "Esta é uma transcrição de exemplo com conteúdo suficiente para validar os endpoints de IA. "
    "Inclui várias frases sobre história, ciência e métodos de estudo para gerar flashcards e resumos. "
    "O objectivo é testar o contrato da API sem chamar a OpenAI de verdade."
)


def _mock_completion(content: str):
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_summarize_rejects_invalid_token(client):
    res = client.post(
        "/summarize",
        json={"text": "Texto curto.", "token": "wrong-token", "lang": "pt"},
    )
    assert res.status_code == 403


@patch("main.client.chat.completions.create")
def test_summarize_returns_summary(mock_create, client):
    mock_create.return_value = _mock_completion("Resumo de teste.")
    res = client.post(
        "/summarize",
        json={"text": LONG_TEXT, "token": API_TOKEN, "lang": "pt", "mode": "normal"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["summary"] == "Resumo de teste."
    mock_create.assert_called_once()


def test_generate_flashcards_rejects_short_text(client):
    res = client.post(
        "/generate-flashcards",
        json={"text": "Curto.", "token": API_TOKEN, "lang": "pt"},
    )
    assert res.status_code == 400
    assert "curto" in res.json()["detail"].lower()


def test_generate_flashcards_rejects_invalid_token(client):
    res = client.post(
        "/generate-flashcards",
        json={"text": LONG_TEXT, "token": "bad", "lang": "pt"},
    )
    assert res.status_code == 403


@patch("main.client.chat.completions.create")
def test_generate_flashcards_returns_cards(mock_create, client):
    payload = {
        "title": "Conjunto de teste",
        "cards": [
            {"front": "Pergunta 1?", "back": "Resposta 1."},
            {"front": "Pergunta 2?", "back": "Resposta 2."},
        ],
    }
    mock_create.return_value = _mock_completion(json.dumps(payload))
    res = client.post(
        "/generate-flashcards",
        json={"text": LONG_TEXT, "token": API_TOKEN, "lang": "pt", "num_cards": 10},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["title"] == "Conjunto de teste"
    assert len(data["cards"]) == 2
    assert data["cards"][0]["front"] == "Pergunta 1?"
    assert data["num_cards"] == 2


@patch("main.client.chat.completions.create")
def test_classify_returns_label(mock_create, client):
    mock_create.return_value = _mock_completion("Aula")
    res = client.post(
        "/classify",
        json={"text": LONG_TEXT, "token": API_TOKEN},
    )
    assert res.status_code == 200, res.text
    assert res.json()["type"] == "Aula"


@patch("main.client.chat.completions.create")
def test_generate_questions_returns_text(mock_create, client):
    mock_create.return_value = _mock_completion("1. Pergunta de teste?\n2. Outra pergunta?")
    res = client.post(
        "/generate-questions",
        json={"text": LONG_TEXT, "token": API_TOKEN, "lang": "pt", "num_questions": 5},
    )
    assert res.status_code == 200, res.text
    assert "questions" in res.json()
    assert "Pergunta" in res.json()["questions"]
