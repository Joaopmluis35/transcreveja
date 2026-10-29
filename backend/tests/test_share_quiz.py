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


def _create(client, origin_headers):
    res = client.post("/api/share/quiz", json={"title": "Ranking", "questions": QUESTIONS}, headers=origin_headers)
    return res.json()["id"]


def test_quiz_ranking_keeps_best_score_per_name(client, origin_headers):
    qid = _create(client, origin_headers)  # 1 pergunta válida
    post = lambda name, score, correct: client.post(
        f"/api/share/quiz/{qid}/scores",
        json={"name": name, "score": score, "correct": correct, "total": 1},
        headers=origin_headers,
    )
    assert post("Ana", 150, 1).json()["position"] == 1
    assert post("Rui", 320, 1).json()["position"] == 1
    third = post("ana", 200, 1).json()  # mesmo nome, melhor pontuação
    assert third["position"] == 2
    assert third["players"] == 2
    ranking = client.get(f"/api/share/quiz/{qid}/scores", headers=origin_headers).json()["ranking"]
    assert [(r["name"], r["score"]) for r in ranking] == [("Rui", 320), ("ana", 200)]


def test_quiz_score_rejects_impossible_results(client, origin_headers):
    qid = _create(client, origin_headers)
    url = f"/api/share/quiz/{qid}/scores"
    assert client.post(url, json={"name": "X", "score": 999999, "correct": 1, "total": 1}, headers=origin_headers).status_code == 400
    assert client.post(url, json={"name": "X", "score": 100, "correct": 2, "total": 1}, headers=origin_headers).status_code == 400
    assert client.post(url, json={"name": "X", "score": 100, "correct": 1, "total": 7}, headers=origin_headers).status_code == 400
    assert client.post(url, json={"name": "   ", "score": 100, "correct": 1, "total": 1}, headers=origin_headers).status_code == 400


def test_quiz_scores_unknown_quiz(client, origin_headers):
    assert client.get("/api/share/quiz/naoexiste/scores", headers=origin_headers).status_code == 404
    res = client.post("/api/share/quiz/naoexiste/scores", json={"name": "A", "score": 0, "correct": 0, "total": 1}, headers=origin_headers)
    assert res.status_code == 404
