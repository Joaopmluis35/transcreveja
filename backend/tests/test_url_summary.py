"""Resumo por URL: um só resumo coerente, língua da página, título/site e erros claros (sem rede nem OpenAI)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import main

API_TOKEN = "test-api-token"
PARAGRAFO = "A padaria da vila abriu em 2019 e hoje faz encomendas para casamentos e batizados. " * 12
HTML = (
    "<html><head><title>Título fallback</title>"
    '<meta property="og:title" content="Padaria da vila faz 5 anos">'
    '<meta property="og:site_name" content="Jornal da Terra"></head>'
    f"<body><article><p>{PARAGRAFO}</p><p>{PARAGRAFO}</p></article></body></html>"
).encode()


def _resposta(conteudo: str):
    choice = MagicMock()
    choice.message.content = conteudo
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _pedir(client, **extra):
    corpo = {"url": "https://www.exemplo.pt/artigo", "token": API_TOKEN, "lang": "pt", "mode": "normal"}
    corpo.update(extra)
    return client.post("/summarize-url", json=corpo)


@pytest.fixture
def pagina():
    with patch("main.validate_public_http_url"), patch("main.safe_http_get", return_value=SimpleNamespace(content=HTML)) as g:
        yield g


@patch("main.client.chat.completions.create")
def test_um_so_resumo_com_titulo_e_site(mock_ia, client, pagina):
    mock_ia.return_value = _resposta("A padaria abriu em 2019.")
    res = _pedir(client)
    assert res.status_code == 200, res.text
    dados = res.json()
    assert dados["summary"] == "A padaria abriu em 2019."
    assert "Parte" not in dados["summary"]
    assert dados["title"] == "Padaria da vila faz 5 anos"
    assert dados["site"] == "Jornal da Terra"
    assert dados["words"] > 100 and dados["reading_minutes"] >= 1
    mock_ia.assert_called_once()  # antes era um pedido por cada bocado de 3000 letras


@pytest.mark.parametrize("lang,esperado", [("pt", "português de Portugal"), ("es", "espanhol"), ("de", "alemão")])
@patch("main.client.chat.completions.create")
def test_escreve_na_lingua_da_pagina(mock_ia, client, pagina, lang, esperado):
    mock_ia.return_value = _resposta("ok")
    assert _pedir(client, lang=lang).status_code == 200
    sistema = mock_ia.call_args.kwargs["messages"][0]["content"]
    assert esperado in sistema


@pytest.mark.parametrize("modo,marca", [("pontos", "pontos-chave"), ("simples", "linguagem muito simples"), ("detalhado", "subtítulos")])
@patch("main.client.chat.completions.create")
def test_estilos_novos(mock_ia, client, pagina, modo, marca):
    mock_ia.return_value = _resposta("ok")
    assert _pedir(client, mode=modo).status_code == 200
    assert marca in mock_ia.call_args.kwargs["messages"][0]["content"]


@patch("main.client.chat.completions.create")
def test_estilo_desconhecido_usa_o_classico(mock_ia, client, pagina):
    mock_ia.return_value = _resposta("ok")
    assert _pedir(client, mode="inventado").status_code == 200
    assert main.SUMMARY_STYLES["normal"] in mock_ia.call_args.kwargs["messages"][0]["content"]


def test_pagina_bloqueada_da_mensagem_clara(client):
    pequena = SimpleNamespace(content=b"<html><body>Inicia sessao</body></html>")
    with patch("main.validate_public_http_url"), patch("main.safe_http_get", return_value=pequena):
        res = _pedir(client)
    assert res.status_code == 422
    assert "Resumo Inteligente" in res.json()["detail"]


def test_erro_a_abrir_nao_mostra_texto_tecnico(client):
    with patch("main.validate_public_http_url"), patch("main.safe_http_get", side_effect=RuntimeError("Traceback interno")):
        res = _pedir(client, lang="en")
    assert res.status_code == 502
    assert "Traceback" not in res.json()["detail"]
    assert "couldn't open" in res.json()["detail"]


@patch("main.client.chat.completions.create", side_effect=RuntimeError("quota"))
def test_falha_da_ia_da_mensagem_clara(mock_ia, client, pagina):
    res = _pedir(client)
    assert res.status_code == 502
    assert "ocupado" in res.json()["detail"]


def test_extract_url_meta_sem_meta_usa_titulo_e_dominio():
    info = main.extract_url_meta(b"<html><head><title> Ola  mundo </title></head></html>", "https://www.site.pt/x")
    assert info == {"title": "Ola mundo", "site": "site.pt"}
