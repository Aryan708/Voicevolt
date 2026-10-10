"""ASR metrics for Hindi / English / Hinglish: normalisation, WER, CER, error analysis.

Pure Python (no jiwer / torch) so it runs anywhere and is unit-tested.

Why a custom normaliser: Whisper's English normaliser mangles Devanagari, and
plain WER over-penalises things that don't change meaning for Hindi
(danda vs full stop, Devanagari vs ASCII digits, nukta variants, ZWJ/ZWNJ).
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍﻿"), None)
# nukta forms -> base letter (क़→क, ज़→ज, फ़→फ, ड़ stays distinct in many orthographies but
# Whisper outputs are inconsistent, so we fold it too when fold_nukta=True)
_NUKTA = "़"
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
# anything that is not a letter, mark or digit (keeps Devanagari vowel signs, which are marks)
_PUNCT = re.compile(r"[^\w\sऀ-ॿ]|_", re.UNICODE)


def normalize(text: str, fold_nukta: bool = True) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = t.translate(_ZERO_WIDTH).translate(_DEV_DIGITS)
    if fold_nukta:
        t = unicodedata.normalize("NFD", t).replace(_NUKTA, "")
        t = unicodedata.normalize("NFC", t)
    t = t.replace("।", " ").replace("॥", " ")    # । ॥
    t = re.sub(r"(?<=\d)[,.](?=\d)", "", t)                 # 4,200 / 4.2 -> keep as one token
    t = _PUNCT.sub(" ", t).lower()
    return re.sub(r"\s+", " ", t).strip()


def script_of(text: str) -> str:
    """'hindi', 'english', 'hinglish' (both scripts) or 'other'."""
    dev, lat = bool(_DEVANAGARI.search(text)), bool(_LATIN.search(text))
    return "hinglish" if dev and lat else "hindi" if dev else "english" if lat else "other"


@dataclass
class Alignment:
    sub: int = 0
    dele: int = 0
    ins: int = 0
    hits: int = 0
    pairs: list[tuple[str, str]] = field(default_factory=list)  # (ref, hyp) substitutions
    deleted: list[str] = field(default_factory=list)
    inserted: list[str] = field(default_factory=list)

    @property
    def ref_len(self) -> int:
        return self.sub + self.dele + self.hits

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Minimum-edit alignment. Among equally cheap alignments it keeps the one with the
    most matched words, so "the cat sat on the mat" vs "the cat sit on mat now" is read as
    sat→sit, -the, +now (useful for error analysis) rather than three substitutions.
    WER is identical either way; only the breakdown differs."""
    n, m = len(ref), len(hyp)
    # cost = (edits, -hits): compared lexicographically
    d = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            e, h = prev[j - 1]
            diag = (e, h - 1) if ri == hyp[j - 1] else (e + 1, h)
            up = (prev[j][0] + 1, prev[j][1])
            left = (row[j - 1][0] + 1, row[j - 1][1])
            row[j] = min(diag, up, left)
    a = Alignment()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            e, h = d[i - 1][j - 1]
            diag = (e, h - 1) if ref[i - 1] == hyp[j - 1] else (e + 1, h)
        if i > 0 and j > 0 and d[i][j] == diag:
            if ref[i - 1] == hyp[j - 1]:
                a.hits += 1
            else:
                a.sub += 1
                a.pairs.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == (d[i - 1][j][0] + 1, d[i - 1][j][1]):
            a.dele += 1
            a.deleted.append(ref[i - 1])
            i -= 1
        else:
            a.ins += 1
            a.inserted.append(hyp[j - 1])
            j -= 1
    return a


def _rate(refs: Iterable[str], hyps: Iterable[str], unit: str, norm: bool) -> tuple[float, Alignment]:
    total = Alignment()
    for r, h in zip(refs, hyps):
        if norm:
            r, h = normalize(r), normalize(h)
        rt, ht = (r.split(), h.split()) if unit == "word" else (list(r.replace(" ", "")),
                                                                 list(h.replace(" ", "")))
        a = align(rt, ht)
        total.sub += a.sub
        total.dele += a.dele
        total.ins += a.ins
        total.hits += a.hits
        total.pairs += a.pairs
        total.deleted += a.deleted
        total.inserted += a.inserted
    return (total.errors / total.ref_len if total.ref_len else 0.0), total


def wer(refs: Sequence[str], hyps: Sequence[str], normalize_text: bool = True) -> float:
    """Corpus-level WER (total errors / total reference words), as papers report it."""
    return _rate(refs, hyps, "word", normalize_text)[0]


def cer(refs: Sequence[str], hyps: Sequence[str], normalize_text: bool = True) -> float:
    return _rate(refs, hyps, "char", normalize_text)[0]


def report(refs: Sequence[str], hyps: Sequence[str], top: int = 15) -> dict:
    """WER/CER overall and per script category, plus the most common errors."""
    w, wa = _rate(refs, hyps, "word", True)
    c, _ = _rate(refs, hyps, "char", True)
    by_cat: dict[str, dict] = {}
    for cat in ("english", "hindi", "hinglish"):
        idx = [i for i, r in enumerate(refs) if script_of(r) == cat]
        if idx:
            by_cat[cat] = {
                "n": len(idx),
                "wer": round(wer([refs[i] for i in idx], [hyps[i] for i in idx]), 4),
                "cer": round(cer([refs[i] for i in idx], [hyps[i] for i in idx]), 4),
            }
    return {
        "n": len(refs), "wer": round(w, 4), "cer": round(c, 4),
        "substitutions": wa.sub, "deletions": wa.dele, "insertions": wa.ins,
        "ref_words": wa.ref_len, "by_script": by_cat,
        "top_substitutions": Counter(wa.pairs).most_common(top),
        "top_deletions": Counter(wa.deleted).most_common(top),
        "top_insertions": Counter(wa.inserted).most_common(top),
    }
