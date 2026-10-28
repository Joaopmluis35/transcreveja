"""Partilha do jogo de quiz: guardar perguntas e abrir pelo link."""
from __future__ import annotations

QUESTIONS = [
    {
        "prompt": "Qual foi o tema da reunião?",
        "options": [{"letter": "A", "text": "Design"}, {"letter": "B", "text": "Férias"}],
        "answer": "A",
        "explanation": "Falou-se do site.",
    },
    {"prompt": "Sem opções", "options": [], "answer": "A"},
    {
        "prompt": "Resposta que não existe",
        "options": [{"letter": "A", "text": "x"}, {"letter": "B", "text": "y"}],
        "answer": "D",
    },
]


def test_share_quiz_roundtrip(client, origin_headers):
    res = client.post(
        "/api/share/quiz",
        json={"title": "Reunião", "locale": "pt", "questions": QUESTIONS},
        headers=origin_headers,
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["url"].endswith(f"/jogo.html?id={body['id']}")

    got = client.get(f"/api/share/quiz/{body['id']}", headers=origin_headers)
    assert got.status_code == 200, got.text
    quiz = got.json()
    assert quiz["title"] == "Reunião"
    # As perguntas inválidas são descartadas.
    assert len(quiz["questions"]) == 1
    assert quiz["questions"][0]["answer"] == "A"


def test_share_quiz_rejects_empty(client, origin_headers):
    res = client.post("/api/share/quiz", json={"questions": [QUESTIONS[1]]}, headers=origin_headers)
    assert res.status_code == 400


def test_share_quiz_requires_allowed_origin(client):
    res = client.post("/api/share/quiz", json={"questions": QUESTIONS}, headers={"Origin": "https://evil.example"})
    assert res.status_code == 403


def test_share_quiz_unknown_id(client, origin_headers):
    assert client.get("/api/share/quiz/naoexiste", headers=origin_headers).status_code == 404
