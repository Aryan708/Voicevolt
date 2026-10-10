"""Speech-to-text with faster-whisper (word-level timestamps + VAD)."""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import Transcript, TranscriptSegment, Word

log = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _load_model(name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"  # float16 is not supported on CPU
    log.info("Loading Whisper '%s' on %s (%s)", name, device, compute_type)
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe(
    audio_path: Path,
    source: str,
    language: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Transcript:
    """Transcribe a 16 kHz WAV.

    `model_name` can also be a local path to a fine-tuned CTranslate2 model,
    which is how the Week 4 Hinglish fine-tune plugs in.
    """
    model = _load_model(
        model_name or settings.whisper_model,
        settings.whisper_device,
        settings.whisper_compute_type,
    )
    seg_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    segments: list[TranscriptSegment] = []
    for s in seg_iter:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip(), prob=w.probability)
            for w in (s.words or [])
        ]
        segments.append(
            TranscriptSegment(start=s.start, end=s.end, text=s.text.strip(), words=words)
        )

    log.info("Transcribed %d segments, language=%s (p=%.2f)",
             len(segments), info.language, info.language_probability)
    return Transcript(
        source=source,
        language=info.language,
        duration=info.duration,
        segments=segments,
    )
