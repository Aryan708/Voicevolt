"""Unit tests that run without GPUs, API keys or model downloads."""
import shutil
import subprocess
from pathlib import Path

import pytest

from voicevault.ingest.chunker import build_chunks
from voicevault.ingest.diarize import assign_speakers
from voicevault.schemas import Keyframe, SpeakerTurn, Transcript, TranscriptSegment, Word, fmt_ts


def make_transcript(n=20, seg_len=5.0):
    segs = []
    for i in range(n):
        s = i * seg_len
        words = [Word(start=s + k, end=s + k + 0.9, text=f"w{i}_{k}") for k in range(5)]
        segs.append(TranscriptSegment(start=s, end=s + seg_len - 0.1,
                                      text=" ".join(w.text for w in words), words=words))
    return Transcript(source="t.mp4", language="en", duration=n * seg_len, segments=segs)


# ---------- chunking ----------

def test_chunks_cover_everything_and_respect_window():
    t = make_transcript()
    chunks = build_chunks(t, window=30, overlap=8)
    assert chunks[0].start == 0
    assert chunks[-1].end == pytest.approx(t.segments[-1].end)
    for c in chunks:
        assert c.end - c.start <= 30 + 1e-6
    # every segment's text appears in some chunk
    all_text = "\n".join(c.text for c in chunks)
    for s in t.segments:
        assert s.text in all_text


def test_chunks_overlap():
    chunks = build_chunks(make_transcript(), window=30, overlap=8)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end, "neighbouring chunks should overlap"
        assert b.start > a.start, "chunking must make progress"


def test_frames_attach_and_render_in_time_order():
    t = make_transcript(n=6)  # 30 s
    frames = [Keyframe(timestamp=12.0, path="f1.jpg", caption="Q3 revenue 4.2 Cr"),
              Keyframe(timestamp=13.0, path="f2.jpg", caption=None)]
    chunks = build_chunks(t, frames, modality="video", window=60, overlap=5)
    assert len(chunks) == 1
    text = chunks[0].text
    assert "[0:12 SLIDE] Q3 revenue 4.2 Cr" in text
    assert text.index("[0:10]") < text.index("SLIDE") < text.index("[0:15]")
    assert len(chunks[0].frames) == 2  # captionless frame kept for display, not in text


def test_visual_only_video_gets_chunks():
    t = Transcript(source="demo.mp4", language="en", duration=60, segments=[])
    frames = [Keyframe(timestamp=x, path=f"{x}.jpg", caption=f"slide {x}") for x in (0, 20, 50)]
    chunks = build_chunks(t, frames, modality="video", window=30)
    assert len(chunks) == 2
    assert "slide 50" in chunks[-1].text


def test_fmt_ts():
    assert fmt_ts(5) == "0:05"
    assert fmt_ts(754) == "12:34"
    assert fmt_ts(3725) == "1:02:05"


# ---------- speaker alignment ----------

def test_speaker_split_mid_segment():
    seg = TranscriptSegment(start=0, end=4, text="a b c d", words=[
        Word(start=0, end=0.9, text="a"), Word(start=1, end=1.9, text="b"),
        Word(start=2, end=2.9, text="c"), Word(start=3, end=3.9, text="d")])
    t = Transcript(source="x", language="en", duration=4, segments=[seg])
    turns = [SpeakerTurn(start=0, end=2.0, speaker="A"),
             SpeakerTurn(start=2.0, end=4, speaker="B")]
    out = assign_speakers(t, turns).segments
    assert [(s.speaker, s.text) for s in out] == [("A", "a b"), ("B", "c d")]


def test_no_diarization_defaults_single_speaker():
    t = assign_speakers(make_transcript(n=2), [])
    assert {s.speaker for s in t.segments} == {"SPEAKER_00"}


# ---------- keyframes (needs ffmpeg) ----------

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_keyframes_on_scene_changes(tmp_path: Path):
    from voicevault.ingest.media import detect_modality
    from voicevault.ingest.video import extract_keyframes

    vid = tmp_path / "slides.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=3",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=3",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=6",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2", "-shortest", str(vid)], check=True)
    assert detect_modality(vid) == "video"
    frames = extract_keyframes(vid, tmp_path / "frames", max_gap=100)
    assert [round(f.timestamp) for f in frames] == [0, 3]
    assert all(Path(f.path).exists() for f in frames)


# ---------- full pipeline wiring (Whisper + vision mocked) ----------

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_pipeline_end_to_end(tmp_path: Path, monkeypatch):
    import json

    from voicevault.ingest import pipeline

    vid = tmp_path / "Team Sync.mp4"
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=4",
        # Note: scene detection is brightness-based, so two slides with equal brightness
        # but different colours (e.g. red vs green) can be missed; max_gap covers that.
        "-f", "lavfi", "-i", "color=c=white:s=320x240:d=4",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=8",
        "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2", "-shortest", str(vid)], check=True)

    def fake_transcribe(wav, source, language=None, model_name=None):
        assert Path(wav).exists() and Path(wav).stat().st_size > 1000
        return Transcript(source=source, language="en", duration=8, segments=[
            TranscriptSegment(start=0.5, end=3.5, text="Let's review the roadmap."),
            TranscriptSegment(start=4.5, end=7.5, text="Revenue grew twelve percent."),
        ])

    def fake_caption(frames, **_):
        for f in frames:
            f.caption = f"Slide at {f.timestamp:.0f}s"
        return frames

    monkeypatch.setattr(pipeline, "transcribe", fake_transcribe)
    monkeypatch.setattr(pipeline, "caption_frames", fake_caption)

    r = pipeline.ingest(vid, diarization=False, out_dir=tmp_path / "out")
    assert r.modality == "video"
    assert len(r.keyframes) == 2
    assert len(r.chunks) == 1
    text = r.chunks[0].text
    assert "[0:00 SLIDE] Slide at 0s" in text and "[0:04 SLIDE] Slide at 4s" in text
    assert "SPEAKER_00] Revenue grew twelve percent." in text

    out = json.loads((tmp_path / "out" / "Team_Sync" / "ingest.json").read_text())
    assert out["chunks"][0]["frames"][1]["timestamp"] == 4.0
    assert set(out["timings"]) >= {"audio", "transcribe", "keyframes", "captions", "chunk"}
