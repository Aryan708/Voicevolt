"""LLM providers behind one streaming interface.

  anthropic - Claude via the Anthropic API
  openai    - GPT via the OpenAI API
  ollama    - Llama / Mistral / Qwen etc. via any OpenAI-compatible server
              (Ollama, vLLM, Groq, Together) using LLM_BASE_URL
"""
from __future__ import annotations

import logging
import re
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


class ExtractiveLLM(FakeLLM):
    """Offline, key-free "LLM" for demos: answers by quoting the source sentences that best
    match the question, with citations. It cannot reason or translate, but it never invents."""

    name = "extractive (offline demo)"
    _SRC = re.compile(r"^\[(\d+)\] [^\n]*\n(.*?)(?=^\[\d+\] |\Z)", re.S | re.M)
    _TAG = re.compile(r"\[(\d+:\d{2}(?::\d{2})?) ([^\]]+)\]\s*")

    def __init__(self):
        super().__init__(self._answer)

    def _answer(self, system: str, messages: Sequence[Message]) -> str:
        last = messages[-1]["content"]
        if "Rewrite the follow-up" in system:
            return last.split("Follow-up:")[-1].strip()
        if "Question:" not in last:
            return "I could not find that in your files."
        ctx, question = last.rsplit("Question:", 1)
        q = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}
        scored = []
        for n, body in self._SRC.findall(ctx):
            # re-join PDF line wraps: a line that doesn't start with a [time SPEAKER] tag
            # continues the previous one
            lines: list[str] = []
            for raw in body.splitlines():
                if lines and raw.strip() and not self._TAG.match(raw):
                    lines[-1] += " " + raw.strip()
                elif raw.strip():
                    lines.append(raw.strip())
            for line in lines:
                tag = self._TAG.match(line)
                text = self._TAG.sub("", line)
                text = re.sub(r"^\[[^\]]+ p\.\d+\]\s*", "", text).strip()
                for sent in re.split(r"(?<=[.!?।])\s+", text):
                    words = set(re.findall(r"\w+", sent.lower()))
                    score = len(q & words) / (1 + 0.02 * len(words))
                    if score > 0 and len(sent) > 15:
                        who = tag.group(2) if tag else None
                        prefix = (f"{who.title()} said at {tag.group(1)}: " if who and who != "SLIDE"
                                  else f"The slide at {tag.group(1)} shows: " if tag else "")
                        scored.append((score, int(n), prefix + sent.rstrip(".") + "."))
        if not scored:
            return "I could not find that in your files."
        scored.sort(key=lambda x: -x[0])
        picked, used = [], set()
        best = scored[0][0]
        for score, n, s in scored:
            if s not in used and len(picked) < 2 and score >= 0.6 * best:
                picked.append(f"{s[:-1]} [{n}].")
                used.add(s)
        return " ".join(picked)


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
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' "
                     "(anthropic | openai | ollama | extractive | fake)")
