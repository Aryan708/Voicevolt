"""Build a clean, speaker-disjoint manifest for Whisper fine-tuning.

Two sources (combine them - public Hindi + your own Hinglish works best):

1. A local folder (best for Hinglish - public Hinglish speech data is scarce):
       my_clips/
         metadata.csv        file,text[,speaker]
         clip_0001.wav ...
   Record 30-60 minutes of yourself and friends reading real Hinglish sentences
   ("kal ki meeting 3 baje shift ho gayi") and this alone gives a visible WER drop.

2. A Hugging Face dataset, e.g.
       --hf google/fleurs --hf-config hi_in --text-col transcription --speaker-col speaker_id
       --hf mozilla-foundation/common_voice_17_0 --hf-config hi --text-col sentence --speaker-col client_id

    python finetune/prepare_data.py --local my_clips --hf google/fleurs --hf-config hi_in \
        --text-col transcription --out data/asr
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.manifest import clean_records, split_by_speaker, summary, write_manifest  # noqa: E402

SR = 16000


def _write_wav(path: Path, audio, sr: int) -> float:
    import numpy as np
    import soundfile as sf

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR, subtype="PCM_16")
    return len(audio) / SR


def from_local(folder: Path, out: Path) -> list[dict]:
    import soundfile as sf

    meta = folder / "metadata.csv"
    if not meta.exists():
        raise SystemExit(f"{meta} not found (columns: file,text[,speaker])")
    recs = []
    with open(meta, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = folder / row["file"]
            if not src.exists():
                print(f"  missing {src.name}, skipped")
                continue
            audio, sr = sf.read(src)
            dst = out / "clips" / f"local_{src.stem}.wav"
            dur = _write_wav(dst, audio, sr)
            recs.append({"audio": str(dst.relative_to(out)), "text": row["text"], "duration": dur,
                         "speaker": row.get("speaker") or f"local_{folder.name}", "source": "local"})
    return recs


def from_hf(name: str, config: str | None, split: str, text_col: str, speaker_col: str | None,
            out: Path, limit: int | None, trust_remote_code: bool) -> list[dict]:
    from datasets import Audio, load_dataset

    ds = load_dataset(name, config, split=split, trust_remote_code=trust_remote_code)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    tag = name.split("/")[-1]
    recs = []
    for i, ex in enumerate(ds):
        dst = out / "clips" / f"{tag}_{split}_{i:06d}.wav"
        dur = _write_wav(dst, ex["audio"]["array"], ex["audio"]["sampling_rate"])
        recs.append({"audio": str(dst.relative_to(out)), "text": ex[text_col], "duration": dur,
                     "speaker": f"{tag}_{ex[speaker_col]}" if speaker_col else None, "source": tag})
        if i and i % 500 == 0:
            print(f"  {i} clips from {name}")
    return recs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=Path, action="append", default=[], help="folder with metadata.csv")
    p.add_argument("--hf", help="Hugging Face dataset name")
    p.add_argument("--hf-config")
    p.add_argument("--hf-splits", default="train,validation,test")
    p.add_argument("--text-col", default="sentence")
    p.add_argument("--speaker-col")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--limit", type=int, help="max clips per HF split (quick experiments)")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--out", type=Path, default=Path("data/asr"))
    a = p.parse_args()

    recs: list[dict] = []
    for folder in a.local:
        print(f"• local {folder}")
        recs += from_local(folder, a.out)
    if a.hf:
        for split in a.hf_splits.split(","):
            print(f"• {a.hf} [{split}]")
            try:
                recs += from_hf(a.hf, a.hf_config, split, a.text_col, a.speaker_col, a.out,
                                a.limit, a.trust_remote_code)
            except ValueError as e:
                print(f"  skipped split {split}: {e}")

    kept, dropped = clean_records(recs)
    final = split_by_speaker(kept, val=a.val, test=a.test)
    write_manifest(final, a.out / "manifest.jsonl")
    print(f"\nKept {len(kept)}/{len(recs)} clips. Dropped: {dropped or 'none'}")
    for split, s in summary(final).items():
        print(f"  {split:<6} {s['clips']:>6} clips  {s['hours']:>6.2f} h  {s['speakers']:>4} speakers")
    print(f"→ {a.out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
