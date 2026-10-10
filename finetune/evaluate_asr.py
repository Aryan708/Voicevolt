"""Compare Whisper models on the held-out test split: WER/CER overall and by script.

    python finetune/evaluate_asr.py --manifest data/asr/manifest.jsonl \
        --model small --model models/whisper-small-hinglish/ct2 --language hi

Writes reports/asr/results.json, reports/asr/results.md (the before/after table for
your README) and a per-utterance CSV for error analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.manifest import read_manifest  # noqa: E402
from voicevault.eval.asr_metrics import normalize, report, script_of  # noqa: E402


def comparison_table(results: list[dict]) -> str:
    """Markdown table; the first model is the baseline for relative change."""
    if not results:
        return ""
    base = results[0]["wer"]
    cats = [c for c in ("english", "hindi", "hinglish") if any(c in r["by_script"] for r in results)]
    head = "| Model | WER | CER | Δ WER vs base | " + " | ".join(f"WER {c}" for c in cats) + " | RTF |"
    sep = "|" + "---|" * (5 + len(cats))
    rows = [head, sep]
    for r in results:
        delta = "—" if r is results[0] else f"{(r['wer'] - base) / base * 100:+.1f}%" if base else "—"
        cat_cells = [f"{r['by_script'][c]['wer'] * 100:.1f}%" if c in r["by_script"] else "—"
                     for c in cats]
        rows.append(f"| `{r['label']}` | {r['wer'] * 100:.1f}% | {r['cer'] * 100:.1f}% | {delta} | "
                    + " | ".join(cat_cells) + f" | {r.get('rtf', 0):.2f} |")
    return "\n".join(rows)


def transcribe_all(model_name: str, clips: list[dict], language: str | None, beam: int,
                   device: str, compute_type: str) -> tuple[list[str], float]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    hyps, t0 = [], time.perf_counter()
    for i, c in enumerate(clips, 1):
        segs, _ = model.transcribe(c["audio"], language=language, beam_size=beam,
                                   vad_filter=False, condition_on_previous_text=False)
        hyps.append(" ".join(s.text.strip() for s in segs))
        if i % 50 == 0:
            print(f"    {i}/{len(clips)}")
    return hyps, time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", action="append", required=True,
                   help="faster-whisper model name or CT2 path; first one is the baseline")
    p.add_argument("--label", action="append", help="display names, same order as --model")
    p.add_argument("--split", default="test")
    p.add_argument("--language", default="hi")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--compute-type", default="int8")
    p.add_argument("--out", type=Path, default=Path("reports/asr"))
    a = p.parse_args()

    clips = read_manifest(a.manifest, a.split)[: a.limit]
    if not clips:
        raise SystemExit(f"no '{a.split}' clips in {a.manifest}")
    audio_s = sum(c["duration"] for c in clips)
    refs = [c["text"] for c in clips]
    labels = a.label or [Path(m).name if "/" in m else m for m in a.model]
    if len(labels) != len(a.model):
        raise SystemExit("give one --label per --model (or none)")
    a.out.mkdir(parents=True, exist_ok=True)

    results, all_hyps = [], {}
    for name, label in zip(a.model, labels, strict=True):
        print(f"• {label}: {len(clips)} clips, {audio_s / 60:.1f} min of audio")
        hyps, secs = transcribe_all(name, clips, a.language, a.beam, a.device, a.compute_type)
        r = report(refs, hyps)
        r.update(label=label, model=name, rtf=round(secs / audio_s, 3))
        results.append(r)
        all_hyps[label] = hyps
        print(f"    WER {r['wer']:.3f}  CER {r['cer']:.3f}  RTF {r['rtf']}")

    (a.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    table = comparison_table(results)
    errs = results[-1]["top_substitutions"][:10]
    md = [f"# ASR evaluation ({a.split} split, {len(clips)} clips, {audio_s / 60:.1f} min)\n", table,
          "\nRTF = processing time / audio duration (lower is faster).",
          f"\n## Most common substitutions ({results[-1]['label']})\n",
          "| reference | hypothesis | count |", "|---|---|---|",
          *[f"| {ref} | {hyp} | {n} |" for (ref, hyp), n in errs]]
    (a.out / "results.md").write_text("\n".join(md), encoding="utf-8")

    with open(a.out / "utterances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio", "script", "reference", *labels])
        for i, c in enumerate(clips):
            w.writerow([c["audio"], script_of(c["text"]), normalize(c["text"]),
                        *[normalize(all_hyps[lb][i]) for lb in labels]])
    print("\n" + table + f"\n\n→ {a.out}/results.md")


if __name__ == "__main__":
    main()
