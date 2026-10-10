"""LLM providers behind one streaming interface.

  anthropic - Claude via the Anthropic API
  openai    - GPT via the OpenAI API
  ollama    - Llama / Mistral / Qwen etc. via any OpenAI-compatible server
              (Ollama, vLLM, Groq, Together) using LLM_BASE_URL
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol
from collections.abc import Callable, Iterator, Sequence

from .config import settings

log = logging.getLogger(__name__)

Message = dict  # {"role": "user" | "assistant", "content": str}


class LLM(Protocol):
    name: str

    def stream(self, system: str, messages: Sequence[Message],
               max_tokens: Optional[int] = None) -> Iterator[str]: ...


def complete(llm: LLM, system: str, messages: Sequence[Message],
             max_tokens: Optional[int] = None) -> str:
    return "".join(llm.stream(system, messages, max_tokens)).strip()


class AnthropicLLM:
    def __init__(self, model: Optional[str] = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        self.model = model or settings.default_llm_model()
        self.name = f"anthropic:{self.model}"

    def stream(self, system, messages, max_tokens=None):
        with self.client.messages.stream(
            model=self.model, system=system, messages=list(messages),
            max_tokens=max_tokens or settings.llm_max_tokens,
        ) as s:
            yield from s.text_stream


class OpenAICompatLLM:
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None, label: str = "openai"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or None,
                             api_key=api_key or settings.openai_api_key or "not-needed")
        self.model = model or settings.default_llm_model()
        self.name = f"{label}:{self.model}"

    def stream(self, system, messages, max_tokens=None):
        resp = self.client.chat.completions.create(
            model=self.model, stream=True, max_tokens=max_tokens or settings.llm_max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        for event in resp:
            if event.choices and (delta := event.choices[0].delta.content):
                yield delta


class FakeLLM:
    """Deterministic LLM for tests and offline demos.

    `responder(system, messages) -> str` decides the reply; the reply is streamed
    word by word so streaming code paths are exercised.
    """

    name = "fake"

    def __init__(self, responder: Optional[Callable[[str, Sequence[Message]], str]] = None):
        self.responder = responder or self._default
        self.calls: list[tuple[str, list[Message]]] = []

    @staticmethod
    def _default(system: str, messages: Sequence[Message]) -> str:
        last = messages[-1]["content"]
        if "Rewrite the follow-up" in system:
            return last.split("Follow-up:")[-1].strip()
        has_ctx = "[1]" in last
        return ("Based on the sources, here is the answer [1]. More detail is in the "
                "second source [2]." if has_ctx else "I could not find that in your files.")

    def stream(self, system, messages, max_tokens=None):
        self.calls.append((system, list(messages)))
        for i, word in enumerate(self.responder(system, messages).split(" ")):
            yield word if i == 0 else " " + word


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> LLM:
    provider = (provider or settings.llm_provider).lower()
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "openai":
        return OpenAICompatLLM(model, base_url=settings.llm_base_url or None, label="openai")
    if provider == "ollama":
        return OpenAICompatLLM(model, base_url=settings.llm_base_url or "http://localhost:11434/v1",
                               api_key="ollama", label="ollama")
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (anthropic | openai | ollama | fake)")
