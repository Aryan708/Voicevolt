"""Turn a transcript + keyframes into retrieval chunks with timestamps."""
from __future__ import annotations

import hashlib
from typing import Optional

from ..config import settings
from ..schemas import Chunk, Keyframe, Transcript, TranscriptSegment, fmt_ts


def _chunk_id(source: str, start: float, end: float) -> str:
    return hashlib.sha1(f"{source}|{start:.2f}|{end:.2f}".encode()).hexdigest()[:16]


def _window_segments(
    segs: list[TranscriptSegment], window: float, overlap: float
) -> list[list[TranscriptSegment]]:
    """Group segments into ~`window`-second windows that never cut a segment in half,
    with ~`overlap` seconds repeated between neighbours so answers at a boundary
    are not lost."""
    groups: list[list[TranscriptSegment]] = []
    i, n = 0, len(segs)
    while i < n:
        start = segs[i].start
        j = i + 1
        while j < n and segs[j].end - start <= window:
            j += 1
        groups.append(segs[i:j])
        if j >= n:
            break
        end = segs[j - 1].end
        k = j
        while k - 1 > i and segs[k - 1].start >= end - overlap:
            k -= 1
        i = max(k, i + 1)
    return groups


def render_text(segs: list[TranscriptSegment], frames: list[Keyframe]) -> str:
    """Interleave speech and slide captions in time order.

    This exact text is what gets embedded, so the LLM can later say
    "at 12:34, Speaker 2 said ... while the slide showed ...".
    """
    events: list[tuple[float, str]] = []
    for s in segs:
        label = f"[{fmt_ts(s.start)}" + (f" {s.speaker}]" if s.speaker else "]")
        events.append((s.start, f"{label} {s.text}"))
    for f in frames:
        if f.caption:
            events.append((f.timestamp, f"[{fmt_ts(f.timestamp)} SLIDE] {f.caption}"))
    events.sort(key=lambda e: e[0])
    return "\n".join(e[1] for e in events)


def build_chunks(
    transcript: Transcript,
    frames: Optional[list[Keyframe]] = None,
    modality: str = "audio",
    window: Optional[float] = None,
    overlap: Optional[float] = None,
) -> list[Chunk]:
    frames = sorted(frames or [], key=lambda f: f.timestamp)
    window = settings.chunk_seconds if window is None else window
    overlap = settings.chunk_overlap_seconds if overlap is None else overlap
    src = transcript.source

    chunks: list[Chunk] = []
    used: set[int] = set()
    groups = _window_segments(transcript.segments, window, overlap)

    for gi, segs in enumerate(groups):
        start, end = segs[0].start, segs[-1].end
        # Extend the last chunk's range so frames after the last word still attach.
        range_end = transcript.duration if gi == len(groups) - 1 else end
        range_start = 0.0 if gi == 0 else start
        fr = [f for f in frames if range_start <= f.timestamp < range_end + 1e-6]
        used.update(id(f) for f in fr)
        chunks.append(Chunk(
            id=_chunk_id(src, start, end), source=src, modality=modality,  # type: ignore[arg-type]
            start=round(start, 2), end=round(end, 2),
            text=render_text(segs, fr),
            speakers=sorted({s.speaker for s in segs if s.speaker}),
            frames=fr,
        ))

    # Visual-only stretches (silent demo, music + slides, gaps between chunks)
    orphans = [f for f in frames if id(f) not in used and f.caption]
    for f_group in _group_frames(orphans, window):
        start, end = f_group[0].timestamp, f_group[-1].timestamp
        chunks.append(Chunk(
            id=_chunk_id(src, start, end + 0.01), source=src, modality=modality,  # type: ignore[arg-type]
            start=round(start, 2), end=round(end, 2),
            text=render_text([], f_group), frames=f_group,
        ))

    chunks.sort(key=lambda c: c.start)
    return chunks


def _group_frames(frames: list[Keyframe], window: float) -> list[list[Keyframe]]:
    groups: list[list[Keyframe]] = []
    for f in frames:
        if groups and f.timestamp - groups[-1][0].timestamp <= window:
            groups[-1].append(f)
        else:
            groups.append([f])
    return groups
