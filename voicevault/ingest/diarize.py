"""Speaker diarization with pyannote + alignment onto transcript segments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import SpeakerTurn, Transcript, TranscriptSegment

log = logging.getLogger(__name__)


def diarize(audio_path: Path, num_speakers: Optional[int] = None) -> list[SpeakerTurn]:
    """Return speaker turns. Returns [] (and logs why) if diarization is unavailable."""
    if not settings.hf_token:
        log.warning("HF_TOKEN not set - skipping diarization (all speech = one speaker).")
        return []
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        log.warning("pyannote.audio not installed - skipping diarization.")
        return []

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=settings.hf_token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    kwargs = {"num_speakers": num_speakers} if num_speakers else {}
    annotation = pipeline(str(audio_path), **kwargs)
    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, speaker=label)
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    log.info("Diarization found %d speakers in %d turns",
             len({t.speaker for t in turns}), len(turns))
    return turns


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> Optional[str]:
    scores: dict[str, float] = {}
    for t in turns:
        if t.end < start:
            continue
        if t.start > end:
            break
        ov = _overlap(start, end, t.start, t.end)
        if ov > 0:
            scores[t.speaker] = scores.get(t.speaker, 0.0) + ov
    if scores:
        return max(scores, key=scores.get)
    # No overlap (e.g. a word in a tiny gap) -> nearest turn
    if not turns:
        return None
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid))).speaker


def assign_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    """Label each segment with a speaker.

    Works at word level and splits a Whisper segment when the speaker changes
    mid-sentence, which happens often in fast back-and-forth conversation.
    """
    if not turns:
        for seg in transcript.segments:
            seg.speaker = "SPEAKER_00"
        return transcript

    turns = sorted(turns, key=lambda t: t.start)
    new_segments: list[TranscriptSegment] = []

    for seg in transcript.segments:
        if not seg.words:
            seg.speaker = _best_speaker(seg.start, seg.end, turns)
            new_segments.append(seg)
            continue

        current: Optional[TranscriptSegment] = None
        for w in seg.words:
            spk = _best_speaker(w.start, w.end, turns)
            if current is None or spk != current.speaker:
                if current is not None:
                    new_segments.append(current)
                current = TranscriptSegment(
                    start=w.start, end=w.end, text=w.text, speaker=spk, words=[w]
                )
            else:
                current.end = w.end
                current.text = f"{current.text} {w.text}".strip()
                current.words.append(w)
        if current is not None:
            new_segments.append(current)

    transcript.segments = new_segments
    return transcript
