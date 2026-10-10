"""Run the evaluation suite and write reports/eval/report.md.

    python scripts/evaluate.py retrieval --eval-set eval/eval_set.jsonl
    python scripts/evaluate.py rag       --eval-set eval/eval_set.jsonl --judge-model claude-sonnet-5-5
    python scripts/evaluate.py latency                       # from live voice sessions
    python scripts/evaluate.py feedback                      # 👍/👎 -> SFT + DPO datasets
    python scripts/evaluate.py all       --eval-set eval/eval_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.config import settings  # noqa: E402
from voicevault.eval.feedback import (build_preference_data, latency_report,  # noqa: E402
                                      latency_table, read_jsonl)
from voicevault.eval.rag_eval import evaluate_rag, rag_table  # noqa: E402
from voicevault.eval.retrieval_eval import evaluate_retrieval, load_eval_set, retrieval_table  # noqa: E402

OUT = Path(os.getenv("REPORTS_DIR") or settings.root / "reports") / "eval"


def _save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_retrieval(a) -> str:
    from voicevault.retrieval import HybridRetriever, open_store

    items = load_eval_set(a.eval_set)
    store, temb, iemb = open_store()
    try:
        res = evaluate_retrieval(HybridRetriever(store, temb, iemb), items, k=a.k)
    finally:
        store.close()
    _save("retrieval", res)
    misses = [q for q in res["per_question"] if not q["hit"]]
    md = [f"## Retrieval ({res['questions']} questions, k={a.k})\n", retrieval_table(res)]
    hy = res["configs"]["hybrid (all)"]
    if len(hy["by_language"]) > 1:
        md += ["\nHybrid by question language:\n", "| Language | MRR | Recall | n |", "|---|---|---|---|"]
        md += [f"| {lg} | {v['mrr']:.3f} | {v['recall']:.3f} | {v['n']} |"
               for lg, v in sorted(hy["by_language"].items())]
    if misses:
        md.append(f"\nMissed by hybrid ({len(misses)}):")
        md += [f"- {q['id']}: {q['question']}" for q in misses[:10]]
    return "\n".join(md)


def run_rag(a) -> str:
    from voicevault.llm import get_llm
    from voicevault.rag import RAGPipeline
    from voicevault.retrieval import HybridRetriever, open_store

    items = load_eval_set(a.eval_set)[: a.limit]
    store, temb, iemb = open_store()
    try:
        rag = RAGPipeline(HybridRetriever(store, temb, iemb), get_llm(), k=a.k)
        judge = get_llm(a.judge_provider, a.judge_model)
        res = evaluate_rag(rag, judge, items)
    finally:
        store.close()
    _save("rag", res)
    worst = sorted((r for r in res["rows"] if r.get("faithfulness") is not None),
                   key=lambda r: r["faithfulness"])[:5]
    md = [f"## Answer quality ({len(items)} questions)\n",
          f"Answering: `{res['answer_llm']}` · judge: `{res['judge_llm']}`\n", rag_table(res)]
    if worst and worst[0]["faithfulness"] < 1:
        md.append("\nLeast faithful answers:")
        md += [f"- {r['id']} ({r['faithfulness']:.2f}): unsupported → "
               + "; ".join(map(str, r["unsupported_claims"][:2])) for r in worst if r["faithfulness"] < 1]
    return "\n".join(md)


def run_latency(a) -> str:
    rep = latency_report(read_jsonl(settings.metrics_path), mode=a.mode)
    _save("latency", rep)
    if not rep["turns"]:
        return f"## Latency\n\nNo {a.mode} turns logged yet in {settings.metrics_path}."
    return (f"## Latency ({rep['turns']} {a.mode} turns, {rep['interrupted']} interrupted)\n\n"
            + latency_table(rep))


def run_feedback(a) -> str:
    data = build_preference_data(read_jsonl(settings.feedback_path))
    out = Path(os.getenv("FEEDBACK_DATASETS_DIR") or settings.root / "data" / "feedback_datasets")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sft", "dpo"):
        (out / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data[name]), encoding="utf-8")
    _save("feedback", {"stats": data["stats"], "issues": data["issues"]})
    s = data["stats"]
    md = ["## User feedback\n", f"- Ratings: {s['ratings']} on {s['questions']} questions",
          f"- 👍 rate: {s['thumbs_up_rate']}" if s["thumbs_up_rate"] is not None else "- 👍 rate: n/a",
          f"- Exported {s['sft']} SFT examples and {s['dpo_pairs']} DPO preference pairs → `{out}`"]
    if data["issues"]:
        md += ["\nTop 👎 reasons:"] + [f"- {r} ({n})" for r, n in data["issues"]]
    return "\n".join(md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["retrieval", "rag", "latency", "feedback", "all"])
    p.add_argument("--eval-set", type=Path, default=Path("eval/eval_set.jsonl"))
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--limit", type=int, help="max questions for the (paid) LLM-judge run")
    p.add_argument("--judge-provider")
    p.add_argument("--judge-model")
    p.add_argument("--mode", default="voice", choices=["voice", "text"])
    a = p.parse_args()

    runs = {"retrieval": run_retrieval, "rag": run_rag, "latency": run_latency, "feedback": run_feedback}
    todo = list(runs) if a.what == "all" else [a.what]
    sections = []
    for name in todo:
        if name in ("retrieval", "rag") and not a.eval_set.exists():
            sections.append(f"## {name.title()}\n\nSkipped: {a.eval_set} not found "
                            "(create one with scripts/make_eval_set.py).")
            continue
        print(f"• {name}…")
        sections.append(runs[name](a))

    report = f"# VoiceVault evaluation\n\n_{time.strftime('%Y-%m-%d %H:%M')}_\n\n" + "\n\n".join(sections)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "report.md" if a.what == "all" else f"{a.what}.md"
    (OUT / name).write_text(report + "\n", encoding="utf-8")
    print("\n" + report + f"\n\n→ {OUT / name}")


if __name__ == "__main__":
    main()
