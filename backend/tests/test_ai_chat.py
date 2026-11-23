"""Funções de texto: Claude primeiro; se falhar (ex.: sem crédito) passa para a OpenAI sem a pessoa notar."""
from __future__ import annotations

from types import SimpleNamespace

import ai_chat


class FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return ai_chat._openai_like("resposta openai")


class FakeClaude:
    def __init__(self, *, error=None, text="resposta claude"):
        self.error, self.text, self.calls = error, text, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=self.text)])


def _chat(claude):
    openai = FakeOpenAI()
    chat = ai_chat.ChatWithFallback(openai)
    chat._claude = claude
    return chat, openai


MSGS = [{"role": "system", "content": "Resume."}, {"role": "assistant", "content": "olá"}, {"role": "user", "content": "texto"}]


def test_uses_claude_when_it_works():
    claude = FakeClaude()
    chat, openai = _chat(claude)
    resp = chat.create(model="gpt-4o-mini", messages=MSGS, temperature=0.3, max_tokens=20)
    assert resp.choices[0].message.content == "resposta claude"
    assert openai.calls == []
    sent = claude.calls[0]
    assert sent["system"] == "Resume."
    assert sent["messages"] == [{"role": "user", "content": "texto"}]  # sem system e a começar pelo utilizador
    assert "temperature" not in sent
    assert sent["max_tokens"] == 20 + ai_chat.THINKING_HEADROOM  # respeita o limite de quem chama


def test_falls_back_to_openai_when_claude_fails():
    chat, openai = _chat(FakeClaude(error=RuntimeError("credit balance is too low")))
    resp = chat.create(model="gpt-4o-mini", messages=MSGS, temperature=0.3, max_tokens=20)
    assert resp.choices[0].message.content == "resposta openai"
    assert openai.calls[0]["model"] == "gpt-4o-mini" and openai.calls[0]["max_tokens"] == 20
    assert "credit balance" in ai_chat.last_status["claude_error"]


def test_without_key_goes_straight_to_openai():
    chat, openai = _chat(None)
    chat.create(model="gpt-4o-mini", messages=MSGS)
    assert len(openai.calls) == 1 and not chat.claude_enabled


def test_json_mode_strips_text_around_the_object():
    chat, _ = _chat(FakeClaude(text='Aqui está:\n```json\n{"a": 1}\n```'))
    resp = chat.create(model="m", messages=MSGS, response_format={"type": "json_object"})
    assert resp.choices[0].message.content == '{"a": 1}'
