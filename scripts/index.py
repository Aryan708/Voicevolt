"""Add files to the VoiceVault knowledge base.

  python scripts/index.py data/raw/                      # ingest + index a folder (any mix)
  python scripts/index.py meeting.mp4 deck.pptx scan.jpg
  python scripts/index.py --processed                    # (re)index everything already ingested
  python scripts/index.py --stats
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicevault.config import settings  # noqa: E402
from voicevault.ingest.documents import DOC_EXTS, ingest_document  # noqa: E402
from voicevault.ingest.media import AUDIO_EXTS, VIDEO_EXTS  # noqa: E402
from voicevault.retrieval import Indexer, open_store  # noqa: E402
from voicevault.schemas import IngestResult  # noqa: E402

MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS


def load_or_ingest(fp: Path, args) -> IngestResult:
    ext = fp.suffix.lower()
    if ext == ".json":
        return IngestResult.model_validate_json(fp.read_text(encoding="utf-8"))
    if ext in DOC_EXTS:
        return ingest_document(fp, vision=not args.no_vision)
    if ext in MEDIA_EXTS:
        from voicevault.ingest.pipeline import ingest

        return ingest(fp, language=args.language, diarization=not args.no_diarize,
                      captions=not args.no_vision, whisper_model=args.model)
    raise ValueError(f"unsupported file type {ext}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest and index files into VoiceVault.")
    p.add_argument("files", nargs="*", help="Files or folders (media, documents, images, ingest.json)")
    p.add_argument("--processed", action="store_true",
                   help="Index every data/processed/*/ingest.json (no re-ingesting)")
    p.add_argument("--rebuild", action="store_true", help="Wipe the index first")
    p.add_argument("--stats", action="store_true", help="Show what is in the index")
    p.add_argument("--no-vision", action="store_true", help="No vision LLM (frame captions / OCR)")
    p.add_argument("--no-diarize", action="store_true")
    p.add_argument("--language")
    p.add_argument("--model", help="Whisper model name or fine-tuned path")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    store, temb, iemb = open_store(rebuild=a.rebuild)
    indexer = Indexer(store, temb, iemb)

    paths: list[Path] = []
    if a.processed:
        paths += sorted(settings.processed_dir.glob("*/ingest.json"))
    for f in a.files:
        fp = Path(f)
        if fp.is_dir():
            paths += sorted(x for x in fp.rglob("*") if x.is_file()
                            and x.suffix.lower() in MEDIA_EXTS | DOC_EXTS)
        else:
            paths.append(fp)

    for fp in paths:
        t0 = time.perf_counter()
        try:
            res = indexer.index(load_or_ingest(fp, a))
            print(f"✓ {res['source']:<40} chunks={res['chunks']:<4} images={res['images']:<4} "
                  f"({time.perf_counter() - t0:.1f}s)")
        except Exception as e:
            print(f"✗ {fp.name}: {e}")

    if a.stats or not paths:
        s = store.stats()
        print(f"\nIndex: {s['chunks']} chunks, {s['images']} images, {len(s['sources'])} sources")
        for src, n in sorted(s["sources"].items()):
            print(f"  {n:>5}  {src}")
    store.close()


if __name__ == "__main__":
    main()
