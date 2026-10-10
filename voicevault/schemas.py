"""Data models shared across the pipeline."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float = 1.0


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    source: str
    language: str
    duration: float
    segments: list[TranscriptSegment]


class SpeakerTurn(BaseModel):
    start: float
    end: float
    speaker: str


class Keyframe(BaseModel):
    timestamp: float
    path: str
    caption: Optional[str] = None


class Chunk(BaseModel):
    """The unit that goes into the vector store in Week 2."""

    id: str
    source: str
    modality: Literal["audio", "video"]
    start: float
    end: float
    text: str
    speakers: list[str] = Field(default_factory=list)
    frames: list[Keyframe] = Field(default_factory=list)

    @property
    def timestamp_label(self) -> str:
        return f"{fmt_ts(self.start)}-{fmt_ts(self.end)}"


class IngestResult(BaseModel):
    source: str
    modality: Literal["audio", "video"]
    transcript: Transcript
    keyframes: list[Keyframe] = Field(default_factory=list)
    chunks: list[Chunk]
    timings: dict[str, float] = Field(default_factory=dict)


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
