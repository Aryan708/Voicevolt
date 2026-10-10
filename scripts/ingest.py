"""CLI: python scripts/ingest.py data/raw/meeting.mp4 [options]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.ingest.pipeline import ingest  # noqa: E402
from voicevault.schemas import fmt_ts  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest audio/video into VoiceVault chunks.")
    p.add_argument("files", nargs="+", help="Audio or video files (or folders)")
    p.add_argument("--language", help="Force language, e.g. en, hi (default: auto-detect)")
    p.add_argument("--speakers", type=int, help="Known number of speakers")
    p.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization")
    p.add_argument("--no-captions", action="store_true", help="Skip vision captions for frames")
    p.add_argument("--model", help="Whisper model name or path to a fine-tuned model")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    paths: list[Path] = []
    for f in a.files:
        fp = Path(f)
        paths.extend(sorted(x for x in fp.iterdir() if x.is_file()) if fp.is_dir() else [fp])

    for fp in paths:
        print(f"\n▶ {fp.name}")
        try:
            r = ingest(fp, language=a.language, num_speakers=a.speakers,
                       diarization=not a.no_diarize, captions=not a.no_captions,
                       whisper_model=a.model)
        except Exception as e:
            print(f"  ✗ failed: {e}")
            continue
        speakers = sorted({s.speaker for s in r.transcript.segments if s.speaker})
        print(f"  modality={r.modality}  language={r.transcript.language}  "
              f"duration={fmt_ts(r.transcript.duration)}")
        print(f"  segments={len(r.transcript.segments)}  speakers={len(speakers)}  "
              f"keyframes={len(r.keyframes)}  chunks={len(r.chunks)}")
        print(f"  timings: " + ", ".join(f"{k}={v}s" for k, v in r.timings.items()))
        if r.chunks:
            c = r.chunks[0]
            preview = c.text[:300].replace("\n", "\n    ")
            print(f"  first chunk [{c.timestamp_label}]:\n    {preview}")


if __name__ == "__main__":
    main()
