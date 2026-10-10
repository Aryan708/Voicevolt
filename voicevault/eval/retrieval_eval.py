"""Retrieval evaluation + ablation (which retrievers actually help?).

Eval set: JSONL, one question per line
  {"id": "q1", "question": "Why did the Chennai launch slip?", "language": "en",
   "relevant": [{"source": "sync.mp4", "start": 740, "end": 800}],
   "answer": "Vendor delays pushed it to Q4."}           # reference answer, optional
  {"id": "q2", "question": "travel reimbursement limit?",
   "relevant": [{"source": "policy.pdf", "page": 3}]}
  {"id": "q3", "question": "What is the CEO's salary?", "unanswerable": true}

A retrieved chunk counts as relevant if it is from the same source AND matches the
page (documents) or overlaps the time range (recordings). A target with only a
source accepts any chunk from that file.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional
from collections.abc import Iterable

from ..schemas import Chunk

ABLATIONS: dict[str, tuple[str, ...]] = {
    "dense": ("dense",),
    "bm25": ("bm25",),
    "visual": ("visual",),
    "dense+bm25": ("dense", "bm25"),
    "hybrid (all)": ("dense", "bm25", "visual"),
}


def load_eval_set(path: Path) -> list[dict]:
    items = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            it = json.loads(line)
            it.setdefault("id", f"q{i + 1}")
            it.setdefault("relevant", [])
            items.append(it)
    return items


def matches(chunk: Chunk, target: dict, tolerance: float = 5.0) -> bool:
    if chunk.source != target.get("source"):
        return False
    if target.get("page") is not None:
        return chunk.page == target["page"]
    if target.get("start") is not None and chunk.start is not None:
        t0, t1 = target["start"] - tolerance, (target.get("end") or target["start"]) + tolerance
        return chunk.start <= t1 and (chunk.end or chunk.start) >= t0
    return True


def score_ranking(ranked: list[Chunk], targets: list[dict], k: int) -> dict:
    """hit@k, recall@k (share of targets found), MRR, nDCG@k for one query."""
    top = ranked[:k]
    rel = [any(matches(c, t) for t in targets) for c in top]
    found = {i for i, t in enumerate(targets) if any(matches(c, t) for c in top)}
    first = next((i for i, r in enumerate(rel, 1) if r), None)
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel) if r)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(targets), k)))
    return {
        "hit": float(bool(found)),
        "recall": len(found) / len(targets) if targets else 0.0,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / ideal if ideal else 0.0,
    }


def _mean(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]} | {"n": len(rows)}


def evaluate_retrieval(retriever, items: list[dict], k: int = 5,
                       configs: Optional[dict[str, tuple[str, ...]]] = None) -> dict:
    """Run every retriever configuration on every answerable question."""
    configs = configs or ABLATIONS
    answerable = [it for it in items if it["relevant"] and not it.get("unanswerable")]
    out: dict = {"k": k, "questions": len(answerable), "configs": {}, "per_question": []}
    for name, retrievers in configs.items():
        rows, by_lang, by_mod = [], defaultdict(list), defaultdict(list)
        for it in answerable:
            hits = retriever.search(it["question"], k=k, retrievers=retrievers)
            s = score_ranking([h.chunk for h in hits], it["relevant"], k)
            rows.append(s)
            by_lang[it.get("language", "en")].append(s)
            by_mod[it.get("modality") or _target_modality(it)].append(s)
            if name == "hybrid (all)":
                out["per_question"].append({"id": it["id"], "question": it["question"], **s,
                                            "top": [h.chunk.citation for h in hits]})
        out["configs"][name] = {"overall": _mean(rows),
                                "by_language": {lg: _mean(v) for lg, v in by_lang.items()},
                                "by_modality": {m: _mean(v) for m, v in by_mod.items()}}
    return out


def _target_modality(it: dict) -> str:
    t = it["relevant"][0]
    return "document" if t.get("page") is not None else "recording" if t.get("start") is not None else "any"


def retrieval_table(res: dict) -> str:
    k = res["k"]
    rows = [f"| Retrievers | Hit@{k} | Recall@{k} | MRR | nDCG@{k} |", "|---|---|---|---|---|"]
    best = max(res["configs"].values(), key=lambda c: c["overall"].get("mrr", 0))["overall"].get("mrr")
    for name, c in res["configs"].items():
        o = c["overall"]
        if not o:
            continue
        mark = "**" if o["mrr"] == best else ""
        rows.append(f"| {mark}{name}{mark} | {o['hit']:.3f} | {o['recall']:.3f} | "
                    f"{mark}{o['mrr']:.3f}{mark} | {o['ndcg']:.3f} |")
    return "\n".join(rows)
