"""Opiniões dos utilizadores: só com autorização, 4-5 estrelas e aprovação no backoffice."""
from __future__ import annotations

ADMIN = {"Authorization": "Bearer test-admin-token"}


def _send(client, origin_headers, **extra):
    body = {"mensagem": "Poupou-me horas na tese.", "source": "csat", "rating": 5, "lang": "pt"}
    body.update(extra)
    return client.post("/api/suggestions", headers=origin_headers, json=body)


def test_testimonial_needs_consent_rating_and_approval(client, origin_headers):
    assert _send(client, origin_headers).status_code == 200  # sem autorização
    assert _send(client, origin_headers, publicar=True, rating=3).status_code == 200  # nota baixa
    assert _send(client, origin_headers, publicar=True, mensagem="Sem comentário.").status_code == 200
    assert _send(client, origin_headers, publicar=True, nome_publico="  Ana,   estudante ").status_code == 200

    items = client.get("/api/admin/testemunhos", headers=ADMIN).json()["items"]
    assert len(items) == 1
    t = items[0]
    assert t["nome"] == "Ana, estudante" and t["rating"] == 5 and t["aprovado"] == 0

    # Antes de aprovar não aparece no site
    assert client.get("/api/testimonials").json()["items"] == []
    assert client.post(f"/api/admin/testemunhos/{t['id']}", headers=ADMIN, json={"aprovado": True}).status_code == 200
    public = client.get("/api/testimonials?lang=pt").json()["items"]
    assert public == [{"nome": "Ana, estudante", "texto": "Poupou-me horas na tese.", "rating": 5}]

    assert client.delete(f"/api/admin/testemunhos/{t['id']}", headers=ADMIN).status_code == 200
    assert client.get("/api/testimonials").json()["items"] == []


def test_testimonial_admin_requires_auth(client):
    assert client.get("/api/admin/testemunhos").status_code in (401, 403)
    assert client.post("/api/admin/testemunhos/1", json={"aprovado": True}).status_code in (401, 403)
