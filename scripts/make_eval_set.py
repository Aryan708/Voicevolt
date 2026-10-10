"""Draft an evaluation set from your indexed files with an LLM, then review it by hand.

    python scripts/make_eval_set.py -n 40 --out eval/eval_set.jsonl

Each question is generated from one chunk (so we know which chunk is relevant),
in a mix of English, Hindi and Hinglish, paraphrased so it does not just copy
the passage's words (which would make BM25 look unrealistically good).
A few unanswerable questions are added to test that the assistant declines.

Always skim the output: delete bad questions and fix references before trusting scores.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.eval.rag_eval import parse_json  # noqa: E402
from voicevault.llm import complete, get_llm  # noqa: E402
from voicevault.retrieval import open_store  # noqa: E402

PROMPT = """Here is a passage from a user's {kind}:
---
{text}
---
Write ONE question the user might realistically ask an assistant that is answered by this \
passage, and a short reference answer (1-2 sentences) taken only from the passage.
Language: {language}. Paraphrase: do not copy distinctive phrases from the passage.
Reply as JSON: {{"question": "...", "answer": "..."}}"""

LANGS = {"en": "English", "hi": "Hindi in Devanagari script",
         "hinglish": "Hinglish (Hindi written in Latin script, mixed with English words)"}

UNANSWERABLE = [
    ("What is the CEO's home address?", "en"),
    ("How many vacation days did I take in 2019?", "en"),
    ("पिछले साल की दिवाली पार्टी में क्या खाना था?", "hi"),
    ("Office ka wifi password kya hai?", "hinglish"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=30, help="number of answerable questions")
    p.add_argument("--langs", default="en:0.5,hi:0.25,hinglish:0.25")
    p.add_argument("--out", type=Path, default=Path("eval/eval_set.jsonl"))
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    rng = random.Random(a.seed)
    weights = {k: float(v) for k, v in (x.split(":") for x in a.langs.split(","))}
    store, *_ = open_store()
    chunks = [c for c in store.iter_chunks() if len(c.text) > 80]
    store.close()
    if not chunks:
        raise SystemExit("index is empty or has no text chunks - run scripts/index.py first")

    # stratify across modalities so documents don't drown out recordings
    by_mod = defaultdict(list)
    for c in chunks:
        by_mod[c.modality].append(c)
    for v in by_mod.values():
        rng.shuffle(v)
    picked, mods = [], sorted(by_mod)
    while len(picked) < min(a.n, len(chunks)):
        for m in mods:
            if by_mod[m] and len(picked) < a.n:
                picked.append(by_mod[m].pop())

    llm = get_llm()
    items = []
    for i, c in enumerate(picked, 1):
        lang = rng.choices(list(weights), weights=list(weights.values()))[0]
        kind = {"video": "meeting/lecture video", "audio": "audio recording"}.get(c.modality, "document")
        try:
            qa = parse_json(complete(llm, "You write evaluation data. Reply with JSON only.",
                                     [{"role": "user", "content": PROMPT.format(
                                         kind=kind, text=c.text[:2500], language=LANGS[lang])}],
                                     max_tokens=300))
        except Exception as e:
            print(f"  skip chunk {c.id}: {e}")
            continue
        target = {"source": c.source}
        if c.page is not None:
            target["page"] = c.page
        if c.start is not None:
            target.update(start=c.start, end=c.end)
        items.append({"id": f"q{i}", "question": qa["question"], "answer": qa.get("answer"),
                      "language": lang, "modality": c.modality, "relevant": [target],
                      "from_chunk": c.id})
        print(f"  [{lang:>8}] {qa['question'][:90]}")

    for j, (q, lang) in enumerate(UNANSWERABLE, 1):
        items.append({"id": f"u{j}", "question": q, "language": lang, "unanswerable": True,
                      "relevant": []})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
                     encoding="utf-8")
    print(f"\n✓ {len(items)} questions -> {a.out}  (review them before trusting scores!)")


if __name__ == "__main__":
    main()
