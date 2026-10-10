"""Dataset manifests: filtering, de-duplication and speaker-disjoint splits.

A manifest is JSONL, one utterance per line:
    {"audio": "clips/0001.wav", "text": "kal ki meeting cancel ho gayi", "duration": 3.2,
     "speaker": "spk_07", "split": "train"}

Pure Python so it is unit-tested; the heavy lifting (audio I/O) lives in prepare_data.py.
"""
from __future__ import annotations

import hashlib
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional
from collections.abc import Iterable

from voicevault.eval.asr_metrics import normalize, script_of


def clean_records(
    records: Iterable[dict],
    min_dur: float = 0.5,
    max_dur: float = 30.0,
    max_chars_per_sec: float = 25.0,
) -> tuple[list[dict], dict[str, int]]:
    """Drop clips Whisper cannot learn from. Returns (kept, drop-reason counts)."""
    kept, dropped, seen = [], defaultdict(int), set()
    for r in records:
        text = (r.get("text") or "").strip()
        dur = float(r.get("duration") or 0)
        if not text:
            dropped["empty_text"] += 1
            continue
        if not (min_dur <= dur <= max_dur):   # Whisper's window is 30 s
            dropped["duration"] += 1
            continue
        if len(normalize(text)) / max(dur, 1e-6) > max_chars_per_sec:
            dropped["text_too_long_for_audio"] += 1   # usually a misaligned transcript
            continue
        key = (normalize(text), round(dur, 1), r.get("speaker"))
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({**r, "text": text, "duration": round(dur, 3), "script": script_of(text)})
    return kept, dict(dropped)


def split_by_speaker(
    records: list[dict],
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 13,
    min_speakers: int = 5,
) -> list[dict]:
    """Assign splits so no speaker appears in more than one split.

    Mixing a speaker across train and test inflates results: the model learns
    the voice, not the language. Records without a speaker are split by a hash
    of their audio path instead (stable across runs).
    """
    rng = random.Random(seed)
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        spk = r.get("speaker") or "anon_" + hashlib.md5(str(r["audio"]).encode()).hexdigest()[:8]
        by_spk[spk].append(r)

    if len(by_spk) < min_speakers:
        # e.g. you recorded everything yourself: speaker-disjoint is impossible, so split by
        # utterance and say so - results then measure "same speaker" accuracy.
        warnings.warn(f"only {len(by_spk)} speaker(s): splitting by utterance, so test "
                      "speakers are also in train", stacklevel=2)
        out, recs = [], sorted(records, key=lambda r: str(r["audio"]))
        rng.shuffle(recs)
        n = len(recs)
        n_test, n_val = round(n * test), round(n * val)
        for i, r in enumerate(recs):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out.append({**r, "speaker": r.get("speaker") or "unknown", "split": split})
        return out

    speakers = sorted(by_spk)
    rng.shuffle(speakers)

    total = sum(r["duration"] for r in records) or 1.0
    targets = {"test": test * total, "val": val * total}
    got = {"test": 0.0, "val": 0.0}
    out = []
    for spk in speakers:
        dur = sum(r["duration"] for r in by_spk[spk])
        split = next((s for s in ("test", "val") if got[s] + dur / 2 < targets[s]), "train")
        if split != "train":
            got[split] += dur
        out += [{**r, "speaker": spk, "split": split} for r in by_spk[spk]]
    return out


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: Path, split: Optional[str] = None) -> list[dict]:
    base = Path(path).parent
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if split and r.get("split") != split:
            continue
        p = Path(r["audio"])
        r["audio"] = str(p if p.is_absolute() else base / p)
        out.append(r)
    return out


def summary(records: list[dict]) -> dict:
    s: dict = defaultdict(lambda: {"clips": 0, "hours": 0.0, "speakers": set()})
    for r in records:
        x = s[r.get("split", "all")]
        x["clips"] += 1
        x["hours"] += r["duration"] / 3600
        x["speakers"].add(r.get("speaker"))
    return {k: {"clips": v["clips"], "hours": round(v["hours"], 2), "speakers": len(v["speakers"])}
            for k, v in sorted(s.items())}
