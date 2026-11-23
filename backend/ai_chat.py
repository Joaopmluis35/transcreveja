"""Funções de texto da IA (resumo, tradução, chat…): Claude primeiro, ChatGPT (OpenAI) se o Claude falhar.

Sem ANTHROPIC_API_KEY o Claude fica desligado e tudo vai direto para a OpenAI, como antes.
A transcrição do áudio continua sempre no Whisper da OpenAI (o Claude não transcreve áudio).
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import anthropic

logger = logging.getLogger("ouviescrevi.ai_chat")

CLAUDE_MODEL = (os.getenv("CLAUDE_MODEL") or "claude-opus-5-5").strip()
# Tarefas curtas (resumos, traduções): esforço baixo chega e sai mais barato e rápido.
CLAUDE_EFFORT = (os.getenv("CLAUDE_EFFORT") or "low").strip()
CLAUDE_TIMEOUT = float(os.getenv("CLAUDE_TIMEOUT", "90"))
THINKING_HEADROOM = 2048

# Estado do último pedido, para o backoffice mostrar se o Claude está a funcionar.
last_status: dict = {"provider": None, "claude_error": None}


def _openai_like(text: str):
    """Mesmo formato que a resposta da OpenAI (resp.choices[0].message.content), para não mexer nos chamadores."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _json_only(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class ChatWithFallback:
    def __init__(self, openai_client):
        self._openai = openai_client
        key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        # base_url fixo: no PC do João há um ANTHROPIC_BASE_URL de outras ferramentas que não serve aqui.
        self._claude = (
            anthropic.Anthropic(api_key=key, base_url="https://api.anthropic.com", timeout=CLAUDE_TIMEOUT, max_retries=1)
            if key
            else None
        )

    @property
    def claude_enabled(self) -> bool:
        return self._claude is not None

    def create(self, *, model, messages, temperature=None, max_tokens=None, response_format=None, **kwargs):
        if self._claude is not None:
            try:
                text = self._ask_claude(messages, json_mode=bool(response_format), max_tokens=max_tokens)
                last_status.update(provider="claude", claude_error=None)
                return _openai_like(text)
            except Exception as e:  # sem crédito, chave inválida, serviço em baixo… → ChatGPT
                last_status.update(claude_error=f"{type(e).__name__}: {str(e)[:200]}")
                logger.warning("Claude falhou, a usar OpenAI: %s", last_status["claude_error"])
        openai_kwargs = dict(model=model, messages=messages, **kwargs)
        if temperature is not None:
            openai_kwargs["temperature"] = temperature
        if max_tokens is not None:
            openai_kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            openai_kwargs["response_format"] = response_format
        resp = self._openai.chat.completions.create(**openai_kwargs)
        last_status["provider"] = "openai"
        return resp

    def _ask_claude(self, messages, *, json_mode: bool, max_tokens=None) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        if json_mode:
            system_parts.append("Responde só com um objeto JSON válido, sem texto antes nem depois.")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] in ("user", "assistant")]
        while convo and convo[0]["role"] == "assistant":  # o Claude exige que a conversa comece pelo utilizador
            convo.pop(0)
        resp = self._claude.messages.create(
            model=CLAUDE_MODEL,
            # O limite de quem chama + folga para o pensamento do modelo (que também conta); sem limite → 16000.
            max_tokens=min(16000, max_tokens + THINKING_HEADROOM) if max_tokens else 16000,
            system="\n\n".join(system_parts) or anthropic.omit,
            messages=convo,
            output_config={"effort": CLAUDE_EFFORT},
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("Claude recusou o pedido")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise RuntimeError(f"Claude sem texto (stop_reason={resp.stop_reason})")
        return _json_only(text) if json_mode else text
