"""End-to-end ingestion: media file -> transcript, keyframes, chunks (JSON)."""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import IngestResult
from .chunker import build_chunks
from .diarize import assign_speakers, diarize
from .media import detect_modality, extract_audio
from .transcribe import transcribe
from .video import extract_keyframes
from .vision import caption_frames

log = logging.getLogger(__name__)


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")[:80] or "media"


@contextmanager
def _timer(timings: dict, key: str):
    t0 = time.perf_counter()
    yield
    timings[key] = round(time.perf_counter() - t0, 2)
    log.info("  %-12s %6.2fs", key, timings[key])


def ingest(
    path: str | Path,
    language: Optional[str] = None,
    num_speakers: Optional[int] = None,
    diarization: bool = True,
    captions: bool = True,
    whisper_model: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> IngestResult:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    modality = detect_modality(path)
    work = (out_dir or settings.processed_dir) / _slug(path.stem)
    work.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    log.info("Ingesting %s as %s -> %s", path.name, modality, work)

    with _timer(timings, "audio"):
        wav = extract_audio(path, work / "audio_16k.wav")

    with _timer(timings, "transcribe"):
        transcript = transcribe(wav, source=path.name, language=language,
                                model_name=whisper_model)

    if diarization:
        with _timer(timings, "diarize"):
            turns = diarize(wav, num_speakers=num_speakers)
    else:
        turns = []
    transcript = assign_speakers(transcript, turns)

    keyframes = []
    if modality == "video":
        with _timer(timings, "keyframes"):
            keyframes = extract_keyframes(path, work / "frames")
        if captions:
            with _timer(timings, "captions"):
                keyframes = caption_frames(keyframes)

    with _timer(timings, "chunk"):
        chunks = build_chunks(transcript, keyframes, modality=modality)

    result = IngestResult(
        source=path.name, modality=modality, transcript=transcript,  # type: ignore[arg-type]
        keyframes=keyframes, chunks=chunks, timings=timings,
    )
    (work / "ingest.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    log.info("Done: %d segments, %d keyframes, %d chunks -> %s",
             len(transcript.segments), len(keyframes), len(chunks), work / "ingest.json")
    return result
