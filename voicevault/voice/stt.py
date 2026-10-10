"""Speech-to-text for live utterances (numpy audio in, text out)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class STTResult:
    text: str
    language: str
    confidence: float = 1.0


class STT(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> STTResult: ...


class WhisperSTT:
    """faster-whisper on an in-memory utterance. Pass a fine-tuned model path via STT_MODEL."""

    def __init__(self, model: Optional[str] = None):
        from ..ingest.transcribe import _load_model

        self.model_name = model or settings.stt_model or settings.whisper_model
        self.model = _load_model(self.model_name, settings.whisper_device,
                                 settings.whisper_compute_type)
        self.name = f"whisper:{self.model_name}"

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> STTResult:
        x = audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio
        segs, info = self.model.transcribe(
            x, language=language, beam_size=1, vad_filter=False,
            condition_on_previous_text=False, without_timestamps=True,
            # bias towards Hinglish spellings people actually use
            initial_prompt=None if language else "Namaste, aaj ki meeting ke baare mein batao.",
        )
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        conf = float(np.exp(np.mean([s.avg_logprob for s in segs]))) if segs else 0.0
        return STTResult(text=text, language=info.language, confidence=round(conf, 3))


class FakeSTT:
    name = "fake"

    def __init__(self, text: str = "what did we decide about the launch", language: str = "en"):
        self.text, self.language = text, language
        self.calls = 0

    def transcribe(self, audio, language=None) -> STTResult:
        self.calls += 1
        return STTResult(self.text, language or self.language)


def get_stt(provider: Optional[str] = None) -> STT:
    provider = (provider or settings.stt_provider).lower()
    if provider == "fake":
        return FakeSTT()
    if provider == "whisper":
        return WhisperSTT()
    raise ValueError(f"Unknown STT_PROVIDER '{provider}' (whisper | fake)")
