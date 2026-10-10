"""ffmpeg / ffprobe helpers."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".webm"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install it: "
            "Windows `winget install ffmpeg`, macOS `brew install ffmpeg`, "
            "Ubuntu `sudo apt install ffmpeg`."
        )


def probe(path: Path) -> dict:
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def duration(path: Path) -> float:
    return float(probe(path)["format"].get("duration", 0.0))


def has_video_stream(path: Path) -> bool:
    streams = probe(path).get("streams", [])
    return any(
        s.get("codec_type") == "video"
        and s.get("disposition", {}).get("attached_pic", 0) == 0  # ignore album art
        for s in streams
    )


def detect_modality(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in VIDEO_EXTS and has_video_stream(path):
        return "video"
    if ext in AUDIO_EXTS or ext in VIDEO_EXTS:
        return "audio"
    raise ValueError(f"Unsupported file type: {path.name}")


def extract_audio(src: Path, dst: Path) -> Path:
    """Convert any media file to 16 kHz mono WAV (what Whisper and pyannote expect)."""
    require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )
    return dst
