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
    modality: Literal["audio", "video", "document", "image"]
    text: str
    # audio / video
    start: Optional[float] = None
    end: Optional[float] = None
    speakers: list[str] = Field(default_factory=list)
    frames: list[Keyframe] = Field(default_factory=list)
    # documents / images
    page: Optional[int] = None
    image_path: Optional[str] = None

    @property
    def timestamp_label(self) -> str:
        if self.start is None:
            return ""
        return f"{fmt_ts(self.start)}-{fmt_ts(self.end or self.start)}"

    @property
    def citation(self) -> str:
        """Human-readable pointer, e.g. 'meeting.mp4 @ 12:30-13:15' or 'deck.pdf p.4'."""
        if self.start is not None:
            return f"{self.source} @ {self.timestamp_label}"
        if self.page is not None:
            return f"{self.source} p.{self.page}"
        return self.source

    def image_paths(self) -> list[str]:
        """Every image that represents this chunk (page image or keyframes)."""
        paths = [self.image_path] if self.image_path else []
        return paths + [f.path for f in self.frames]


class IngestResult(BaseModel):
    source: str
    modality: Literal["audio", "video", "document", "image"]
    transcript: Optional[Transcript] = None
    keyframes: list[Keyframe] = Field(default_factory=list)
    chunks: list[Chunk]
    timings: dict[str, float] = Field(default_factory=dict)


def fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
