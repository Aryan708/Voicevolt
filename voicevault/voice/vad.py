"""Voice activity detection + utterance endpointing for a live 16 kHz PCM16 stream.

The segmenter turns a continuous microphone stream into utterances:
  * emits "start" as soon as speech begins (used for barge-in)
  * emits "end" with the full utterance audio after `silence_ms` of silence
  * keeps a short pre-roll so the first syllable is not clipped

Uses webrtcvad when installed (robust to noise), otherwise an adaptive energy
detector with no dependencies.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000      # 480
FRAME_BYTES = FRAME_SAMPLES * 2


class EnergyVAD:
    """Speech if frame energy is well above an adaptive noise floor."""

    def __init__(self, margin_db: float = 12.0, min_db: float = -50.0):
        self.margin_db, self.min_db = margin_db, min_db
        self.noise_db = -60.0

    def is_speech(self, frame: np.ndarray) -> bool:
        x = frame.astype(np.float32) / 32768.0
        db = 10 * np.log10(np.mean(x * x) + 1e-10)
        speech = db > max(self.noise_db + self.margin_db, self.min_db)
        if not speech:  # track the noise floor only on non-speech frames
            self.noise_db = 0.95 * self.noise_db + 0.05 * db
        return speech


class WebRtcVAD:
    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        return self.vad.is_speech(frame.astype(np.int16).tobytes(), SAMPLE_RATE)


def default_vad():
    try:
        return WebRtcVAD()
    except Exception:
        return EnergyVAD()


@dataclass
class VADEvent:
    type: str                       # "start" | "end"
    audio: Optional[np.ndarray] = None   # int16, only on "end"

    @property
    def duration(self) -> float:
        return 0.0 if self.audio is None else len(self.audio) / SAMPLE_RATE


class UtteranceSegmenter:
    def __init__(self, vad=None, silence_ms: int = 700, start_ms: int = 90,
                 preroll_ms: int = 300, min_utterance_ms: int = 300,
                 max_utterance_s: float = 30.0):
        self.vad = vad or default_vad()
        self.end_frames = max(1, silence_ms // FRAME_MS)
        self.start_frames = max(1, start_ms // FRAME_MS)
        self.min_frames = max(1, min_utterance_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 // FRAME_MS)
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._buf = b""
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self.frames: list[np.ndarray] = []
        self.voiced_run = 0
        self.voiced_total = 0   # speech frames in the current utterance (excludes pre-roll)
        self.silent_run = 0
        self.preroll.clear()

    def feed(self, pcm: bytes) -> list[VADEvent]:
        """Feed any number of PCM16 bytes; returns zero or more events."""
        self._buf += pcm
        events: list[VADEvent] = []
        while len(self._buf) >= FRAME_BYTES:
            raw, self._buf = self._buf[:FRAME_BYTES], self._buf[FRAME_BYTES:]
            ev = self._frame(np.frombuffer(raw, dtype=np.int16))
            if ev:
                events.append(ev)
        return events

    def _frame(self, frame: np.ndarray) -> Optional[VADEvent]:
        speech = self.vad.is_speech(frame)
        if not self.in_speech:
            self.preroll.append(frame)
            self.voiced_run = self.voiced_run + 1 if speech else 0
            if self.voiced_run >= self.start_frames:
                self.in_speech = True
                self.frames = list(self.preroll)
                self.voiced_total = self.voiced_run
                self.silent_run = 0
                return VADEvent("start")
            return None

        self.frames.append(frame)
        self.silent_run = 0 if speech else self.silent_run + 1
        self.voiced_total += int(speech)
        if self.silent_run >= self.end_frames or len(self.frames) >= self.max_frames:
            voiced = self.voiced_total
            audio = np.concatenate(self.frames[:len(self.frames) - self.silent_run + 3])
            self.reset()
            if voiced < self.min_frames:
                return VADEvent("end", None)   # a click or cough - ignore
            return VADEvent("end", audio)
        return None
