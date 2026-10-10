"""Answer-quality evaluation (RAGAS-style metrics, implemented directly).

Per question:
  faithfulness       share of the answer's claims supported by the retrieved sources (LLM judge)
  relevance          does the answer address the question? 1-5 -> 0-1 (LLM judge)
  correctness        agreement with the reference answer, when one exists, 1-5 -> 0-1 (LLM judge)
  citation_precision share of cited [n] that point at a truly relevant chunk (no LLM)
  refusal_ok         for unanswerable questions: did it decline instead of inventing? (LLM judge)

Use a stronger model as judge than the one answering (e.g. answer with Haiku, judge with Sonnet).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..llm import LLM, complete
from ..rag import RAGPipeline, format_context
from .retrieval_eval import matches

log = logging.getLogger(__name__)

JUDGE_SYSTEM = "You are a strict evaluator of a retrieval-augmented assistant. Reply with JSON only."

FAITHFULNESS = """Sources:
{context}

Answer to check:
{answer}

Split the answer into atomic factual claims. For each claim decide whether the sources \
explicitly support it. Ignore citation markers like [1]. Reply as JSON:
{{"claims": [{{"claim": "...", "supported": true}}]}}"""

RELEVANCE = """Question: {question}
Answer: {answer}

How well does the answer address the question (regardless of correctness)?
1 = ignores it, 3 = partially, 5 = fully and directly. Reply as JSON: {{"score": <1-5>}}"""

CORRECTNESS = """Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

How well does the candidate agree with the reference on the facts that matter?
1 = contradicts or misses them, 3 = partly, 5 = same facts (wording may differ).
Reply as JSON: {{"score": <1-5>}}"""

REFUSAL = """Question: {question}
Answer: {answer}

The question CANNOT be answered from the user's files. Did the answer say it could not \
find the information (instead of inventing an answer)? Reply as JSON: {{"declined": true}}"""


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences / chatter)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:120]!r}")
    return json.loads(m.group(0))


def _judge(llm: LLM, prompt: str) -> dict:
    try:
        return parse_json(complete(llm, JUDGE_SYSTEM, [{"role": "user", "content": prompt}],
                                   max_tokens=800))
    except Exception as e:
        log.warning("judge failed: %s", e)
        return {}


def _scale(score) -> Optional[float]:
    try:
        return round((float(score) - 1) / 4, 3)
    except (TypeError, ValueError):
        return None


def evaluate_item(rag: RAGPipeline, judge: LLM, item: dict) -> dict:
    ans = rag.ask(item["question"])
    row: dict = {"id": item["id"], "question": item["question"], "answer": ans.text,
                 "cited": ans.cited, "latency_s": ans.timings.get("total")}

    if item.get("unanswerable"):
        v = _judge(judge, REFUSAL.format(question=item["question"], answer=ans.text))
        row["refusal_ok"] = float(bool(v.get("declined"))) if v else None
        return row

    context = format_context(ans.hits) if ans.hits else "(none)"
    f = _judge(judge, FAITHFULNESS.format(context=context, answer=ans.text))
    claims = f.get("claims") or []
    row["faithfulness"] = (round(sum(bool(c.get("supported")) for c in claims) / len(claims), 3)
                           if claims else None)
    row["unsupported_claims"] = [c.get("claim") for c in claims if not c.get("supported")]
    row["relevance"] = _scale(_judge(judge, RELEVANCE.format(
        question=item["question"], answer=ans.text)).get("score"))
    if item.get("answer"):
        row["correctness"] = _scale(_judge(judge, CORRECTNESS.format(
            question=item["question"], reference=item["answer"], answer=ans.text)).get("score"))
    if ans.cited and item.get("relevant"):
        good = [n for n in ans.cited
                if any(matches(ans.hits[n - 1].chunk, t) for t in item["relevant"])]
        row["citation_precision"] = round(len(good) / len(ans.cited), 3)
    row["has_citation"] = float(bool(ans.cited))
    return row


METRICS = ("faithfulness", "relevance", "correctness", "citation_precision",
           "has_citation", "refusal_ok")


def evaluate_rag(rag: RAGPipeline, judge: LLM, items: list[dict]) -> dict:
    rows = [evaluate_item(rag, judge, it) for it in items]
    summary = {}
    for m in METRICS:
        vals = [r[m] for r in rows if r.get(m) is not None]
        if vals:
            summary[m] = {"mean": round(sum(vals) / len(vals), 3), "n": len(vals)}
    lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    if lat:
        summary["latency_p50_s"] = lat[len(lat) // 2]
    return {"summary": summary, "rows": rows,
            "answer_llm": getattr(rag.llm, "name", "?"), "judge_llm": getattr(judge, "name", "?")}


def rag_table(res: dict) -> str:
    names = {"faithfulness": "Faithfulness (claims supported)", "relevance": "Answer relevance",
             "correctness": "Correctness vs reference", "citation_precision": "Citation precision",
             "has_citation": "Answers with ≥1 citation", "refusal_ok": "Declines unanswerable"}
    rows = ["| Metric | Score | n |", "|---|---|---|"]
    for m in METRICS:
        if m in res["summary"]:
            s = res["summary"][m]
            rows.append(f"| {names[m]} | {s['mean']:.3f} | {s['n']} |")
    return "\n".join(rows)
