"""Turn 👍/👎 feedback and latency logs into datasets and reports.

Human feedback -> training data (the data-collection half of RLHF):
  sft.jsonl  {"prompt", "response"}                        from 👍 answers (supervised fine-tuning)
  dpo.jsonl  {"prompt", "chosen", "rejected"}              same question rated 👍 and 👎 (DPO / reward model)
  issues     most common 👎 reasons, to fix prompts/retrieval first

DPO data can train an open model (e.g. Llama with TRL's DPOTrainer) to prefer
grounded, cited answers. The cheaper first step is reading the 👎 issues.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from collections.abc import Iterable

from .asr_metrics import normalize


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def build_preference_data(feedback: Iterable[dict]) -> dict:
    by_q: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"up": [], "down": []})
    for fb in feedback:
        key = normalize(fb.get("question", ""))
        if key and fb.get("answer"):
            by_q[key]["up" if fb.get("rating", 0) > 0 else "down"].append(fb)

    sft, dpo, reasons = [], [], Counter()
    for group in by_q.values():
        ups = list({u["answer"]: u for u in group["up"]}.values())      # dedupe identical answers
        downs = list({d["answer"]: d for d in group["down"]}.values())
        sft += [{"prompt": u["question"], "response": u["answer"]} for u in ups]
        for d in downs:
            if d.get("comment"):
                reasons[d["comment"].strip().lower()[:80]] += 1
            for u in ups:
                if u["answer"] != d["answer"]:
                    dpo.append({"prompt": u["question"], "chosen": u["answer"], "rejected": d["answer"]})
    total = sum(len(g["up"]) + len(g["down"]) for g in by_q.values())
    ups_n = sum(len(g["up"]) for g in by_q.values())
    return {"sft": sft, "dpo": dpo, "issues": reasons.most_common(10),
            "stats": {"ratings": total, "thumbs_up_rate": round(ups_n / total, 3) if total else None,
                      "questions": len(by_q), "sft": len(sft), "dpo_pairs": len(dpo)}}


def percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    i = (len(s) - 1) * p / 100
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (i - lo), 1)


LATENCY_KEYS = {"stt_ms": "Speech-to-text", "retrieval_done_ms": "Retrieval done",
                "first_token_ms": "First LLM token", "first_audio_ms": "First audio (voice-to-voice)",
                "total_ms": "Full answer"}


def latency_report(metrics: Iterable[dict], mode: str | None = "voice") -> dict:
    metrics = list(metrics)  # iterated twice below
    rows = [m for m in metrics if (mode is None or m.get("mode") == mode) and "total_ms" in m]
    out = {"turns": len(rows), "interrupted": 0, "stages": {}}
    out["interrupted"] = sum(1 for m in metrics if "interrupted_ms" in m)
    for k, label in LATENCY_KEYS.items():
        vals = [m[k] for m in rows if k in m]
        if vals:
            out["stages"][k] = {"label": label, "p50": percentile(vals, 50),
                                "p90": percentile(vals, 90), "p95": percentile(vals, 95), "n": len(vals)}
    return out


def latency_table(rep: dict) -> str:
    rows = ["| Stage (ms after you stop talking) | p50 | p90 | p95 | n |", "|---|---|---|---|---|"]
    for s in rep["stages"].values():
        rows.append(f"| {s['label']} | {s['p50']:.0f} | {s['p90']:.0f} | {s['p95']:.0f} | {s['n']} |")
    return "\n".join(rows)
