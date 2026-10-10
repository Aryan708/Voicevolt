"""Retrieval-augmented answering with citations, follow-ups and Hindi/English support."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from .config import settings
from .llm import LLM, Message, complete
from .retrieval import Hit, HybridRetriever

log = logging.getLogger(__name__)

SYSTEM = """You are VoiceVault, an assistant that answers questions about the user's own \
meetings, lectures, documents and images.

Rules:
- Answer ONLY from the numbered sources. If they do not contain the answer, say you could \
not find it in their files. Never invent facts, names, numbers or timestamps.
- Cite sources inline with their number in square brackets, like [2]. Every factual \
sentence needs at least one citation.
- When a source is a recording, mention who said it and roughly when if that helps \
(e.g. "Speaker 1 said at 12:30 ...").
- Reply in the same language and script the user used (English, Hindi, or Hinglish).
- Be concise and direct."""

VOICE_ADDENDUM = """
- Your reply will be spoken aloud: 2 to 4 short sentences, no markdown, no lists, no URLs. \
Say times naturally ("at twelve thirty" or "around minute twelve")."""

CONDENSE = """Rewrite the follow-up question as a standalone search query, using the \
conversation for missing context (names, topics, "it", "that", "us"). Keep the user's \
language. Output only the rewritten query."""

_CITE_RE = re.compile(r"\[(\d{1,2})\]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def parse_citations(text: str) -> list[int]:
    return sorted({int(n) for n in _CITE_RE.findall(text)})


def strip_citations(text: str) -> str:
    """Remove [n] markers before speaking."""
    return re.sub(r"\s*\[\d{1,2}\]", "", text)


def is_hindi(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


def format_context(hits: Sequence[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        c = h.chunk
        parts.append(f"[{i}] {c.citation} ({c.modality})\n{c.text.strip()}")
    return "\n\n".join(parts)


@dataclass
class Answer:
    question: str
    query: str
    text: str
    hits: list[Hit]
    cited: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def sources(self, only_cited: bool = True) -> list[dict]:
        return [source_dict(i, h) for i, h in enumerate(self.hits, 1)
                if not only_cited or i in self.cited]


def source_dict(n: int, h: Hit) -> dict:
    c = h.chunk
    return {
        "n": n, "chunk_id": c.id, "source": c.source, "modality": c.modality,
        "citation": c.citation, "start": c.start, "end": c.end, "page": c.page,
        "image": h.image, "speakers": c.speakers, "matched_by": h.matched_by,
        "score": h.score, "text": c.text[:600],
    }


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM, k: Optional[int] = None):
        self.retriever, self.llm = retriever, llm
        self.k = k or settings.retrieval_k

    def condense(self, question: str, history: Sequence[Message]) -> str:
        """Turn 'and what about the south region?' into a self-contained query."""
        if not history:
            return question
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
        try:
            q = complete(self.llm, CONDENSE,
                         [{"role": "user", "content": f"{convo}\n\nFollow-up: {question}"}],
                         max_tokens=100)
            return q.strip().strip('"') or question
        except Exception as e:
            log.warning("condense failed (%s); using raw question", e)
            return question

    def stream(
        self,
        question: str,
        history: Sequence[Message] = (),
        voice: bool = False,
        **filters,
    ) -> Iterator[dict]:
        """Yields events: {"type": "sources", ...}, {"type": "token", "text"}, {"type": "done", ...}"""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        query = self.condense(question, history)
        timings["condense"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        hits = self.retriever.search(query, k=self.k, **filters)
        timings["retrieval"] = round(time.perf_counter() - t1, 3)
        yield {"type": "sources", "query": query,
               "sources": [source_dict(i, h) for i, h in enumerate(hits, 1)]}

        system = SYSTEM + (VOICE_ADDENDUM if voice else "")
        context = format_context(hits) if hits else "(no sources found)"
        messages = [*list(history)[-6:], {"role": "user", "content":
                    f"Sources:\n{context}\n\nQuestion: {question}"}]

        t2 = time.perf_counter()
        parts: list[str] = []
        for tok in self.llm.stream(system, messages):
            if not parts:
                timings["first_token"] = round(time.perf_counter() - t2, 3)
            parts.append(tok)
            yield {"type": "token", "text": tok}
        text = "".join(parts).strip()
        timings["generation"] = round(time.perf_counter() - t2, 3)
        timings["total"] = round(time.perf_counter() - t0, 3)

        cited = [n for n in parse_citations(text) if 1 <= n <= len(hits)]
        yield {"type": "done", "answer": Answer(question, query, text, hits, cited, timings)}

    def ask(self, question: str, history: Sequence[Message] = (), voice: bool = False,
            **filters) -> Answer:
        for ev in self.stream(question, history, voice, **filters):
            if ev["type"] == "done":
                return ev["answer"]
        raise RuntimeError("RAG stream ended without an answer")
