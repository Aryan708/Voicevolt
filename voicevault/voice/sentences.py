"""Split a token stream into speakable sentences as soon as each one is complete.

Speaking sentence-by-sentence (instead of waiting for the whole answer) is the
single biggest latency win in a voice agent: TTS of sentence 1 overlaps with the
LLM still writing sentence 2.
"""
from __future__ import annotations

import re
from typing import Iterator

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "no", "rs",
           "approx", "inc", "ltd", "fig", "st"}
# sentence end: . ! ? or Hindi danda, followed by whitespace
_END = re.compile(r"([.!?।॥])(\s+)")


class SentenceStreamer:
    def __init__(self, min_chars: int = 25, max_chars: int = 240):
        self.min_chars, self.max_chars = min_chars, max_chars
        self.buf = ""

    def feed(self, text: str) -> list[str]:
        self.buf += text
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            sentence, self.buf = self.buf[:cut].strip(), self.buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> list[str]:
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []

    def _find_cut(self):
        for m in _END.finditer(self.buf):
            end = m.end(1)
            if end < self.min_chars:
                continue
            if m.group(1) == ".":
                before = self.buf[:m.start(1)]
                word = re.split(r"\s", before)[-1].lower().rstrip(".")
                if word in _ABBREV or (len(word) == 1 and word.isalpha()):
                    continue  # "Dr. Rao", "A. Sharma"
            return m.end(2)
        # very long run with no punctuation: cut at a comma / space
        if len(self.buf) > self.max_chars:
            for sep in (", ", "; ", " "):
                i = self.buf.rfind(sep, self.min_chars, self.max_chars)
                if i > 0:
                    return i + len(sep)
        return None


def split_stream(tokens: Iterator[str], **kw) -> Iterator[str]:
    s = SentenceStreamer(**kw)
    for t in tokens:
        yield from s.feed(t)
    yield from s.flush()
