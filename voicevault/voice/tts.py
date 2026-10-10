"""Text-to-speech backends. Each returns complete audio bytes for one sentence.

  edge   - Microsoft Edge neural voices (free, no key, excellent en-IN and hi-IN voices)
  openai - OpenAI TTS (gpt-4o-mini-tts)
  piper  - fully local and offline (download a .onnx voice, e.g. en_US-lessac / hi_IN-*)
  none   - text only
"""
from __future__ import annotations

import asyncio
import io
import logging
import wave
from dataclasses import dataclass
from typing import Optional, Protocol

from ..config import settings
from ..rag import is_hindi

log = logging.getLogger(__name__)


@dataclass
class Speech:
    audio: bytes
    mime: str        # audio/mpeg | audio/wav


class TTS(Protocol):
    name: str

    def synthesize(self, text: str) -> Optional[Speech]: ...


class EdgeTTS:
    name = "edge"

    def __init__(self, voice_en: Optional[str] = None, voice_hi: Optional[str] = None):
        import edge_tts  # noqa: F401  (fail early if missing)

        self.voice_en = voice_en or settings.tts_voice_en
        self.voice_hi = voice_hi or settings.tts_voice_hi

    async def _run(self, text: str, voice: str) -> bytes:
        import edge_tts

        out = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                out.extend(chunk["data"])
        return bytes(out)

    def synthesize(self, text: str) -> Optional[Speech]:
        voice = self.voice_hi if is_hindi(text) else self.voice_en
        audio = asyncio.run(self._run(text, voice))  # called from a worker thread
        return Speech(audio, "audio/mpeg") if audio else None


class OpenAITTS:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini-tts", voice: str = "alloy"):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.openai_api_key or None)
        self.model, self.voice = model, voice

    def synthesize(self, text: str) -> Optional[Speech]:
        r = self.client.audio.speech.create(model=self.model, voice=self.voice, input=text,
                                            response_format="mp3")
        return Speech(r.read(), "audio/mpeg")


class PiperTTS:
    name = "piper"

    def __init__(self, model_en: Optional[str] = None, model_hi: Optional[str] = None):
        from piper import PiperVoice

        en = model_en or settings.piper_model_en
        if not en:
            raise RuntimeError("Set PIPER_MODEL_EN to a downloaded Piper .onnx voice")
        self.voice_en = PiperVoice.load(en)
        hi = model_hi or settings.piper_model_hi
        self.voice_hi = PiperVoice.load(hi) if hi else self.voice_en

    def synthesize(self, text: str) -> Optional[Speech]:
        voice = self.voice_hi if is_hindi(text) else self.voice_en
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            if hasattr(voice, "synthesize_wav"):   # piper-tts >= 1.3
                voice.synthesize_wav(text, w)
            else:                                  # piper-tts 1.2
                voice.synthesize(text, w)
        return Speech(buf.getvalue(), "audio/wav")


class SilentTTS:
    name = "none"

    def synthesize(self, text: str) -> Optional[Speech]:
        return None


class FakeTTS:
    """Returns a tiny valid WAV per sentence and records what it was asked to say."""

    name = "fake"

    def __init__(self, delay: float = 0.0):
        self.spoken: list[str] = []
        self.delay = delay

    def synthesize(self, text: str) -> Optional[Speech]:
        import time

        if self.delay:
            time.sleep(self.delay)
        self.spoken.append(text)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 160)
        return Speech(buf.getvalue(), "audio/wav")


def get_tts(provider: Optional[str] = None) -> TTS:
    provider = (provider or settings.tts_provider).lower()
    try:
        if provider == "edge":
            return EdgeTTS()
        if provider == "openai":
            return OpenAITTS()
        if provider == "piper":
            return PiperTTS()
        if provider == "fake":
            return FakeTTS()
    except Exception as e:
        log.warning("TTS '%s' unavailable (%s) - falling back to text only", provider, e)
    return SilentTTS()
