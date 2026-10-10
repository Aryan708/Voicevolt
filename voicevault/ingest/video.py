"""Keyframe extraction tuned for meetings and lectures.

ffmpeg's built-in scene score mostly measures brightness change, so it misses the
most important event in a meeting: a new slide with the same background and new
text (measured scores of 0.05-0.26 against a usual 0.3 threshold). This detector
instead:

  1. samples the video at `fps` frames/second, downscaled to 160x90 RGB (streamed,
     so a 2-hour recording uses a few MB of memory)
  2. measures the *share of pixels that changed* against the LAST KEPT frame
     (so slow changes such as bullets appearing one by one still add up)
  3. keeps a frame when the picture has changed AND has then stopped changing:
     slides are static, a talking head never is, so webcam footage does not
     produce a flood of near-identical frames
  4. also keeps hard cuts (most pixels change at once) and forces one frame every
     `max_gap` seconds so nothing long goes unsampled

Full-resolution JPEGs are then grabbed only at the chosen timestamps.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import settings
from ..schemas import Keyframe
from .media import require_ffmpeg

log = logging.getLogger(__name__)

W, H = 160, 90
PIXEL_DELTA = 30  # a pixel "changed" if any RGB channel moved by more than this (0-255)


def changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > PIXEL_DELTA).mean())


def select_keyframes(
    frames,                       # iterable of HxWx3 uint8 arrays, evenly spaced
    fps: float,
    change: float = 0.012,        # >1.2% of pixels differ from the last kept frame
    stable: float = 0.004,        # and <0.4% differ from the previous sample
    cut: float = 0.30,            # >30% differ from the previous sample = hard cut
    max_gap: float = 30.0,
    min_gap: float = 1.0,
) -> list[float]:
    """Pure function (unit-tested): returns timestamps of frames worth keeping."""
    times: list[float] = []
    last_kept: Optional[np.ndarray] = None
    prev: Optional[np.ndarray] = None
    last_t = -1e9
    for i, f in enumerate(frames):
        t = i / fps
        if last_kept is None:
            times.append(0.0)
            last_kept, last_t = f, 0.0
        else:
            d_prev = changed_fraction(f, prev)
            d_kept = changed_fraction(f, last_kept)
            settled = d_prev < stable and d_kept > change      # new static content (a slide)
            hard_cut = d_prev > cut
            if (settled or hard_cut) and t - last_t >= min_gap:
                # for a settled slide, it was already on screen one sample earlier
                ts = round(max(last_t + min_gap, t - 1 / fps) if settled else t, 2)
                times.append(ts)
                last_kept, last_t = f, ts
            elif t - last_t >= max_gap:
                times.append(round(t, 2))
                last_kept, last_t = f, t
        prev = f
    return times


def _sample_frames(video: Path, fps: float):
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    size = W * H * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg frame sampling failed:\n{err[-2000:]}")


def _grab(video: Path, t: float, out: Path, width: int) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
                    "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out)], check=True)


def extract_keyframes(
    video_path: Path,
    out_dir: Path,
    scene_threshold: Optional[float] = None,
    max_gap: Optional[float] = None,
    width: int = 768,
    fps: Optional[float] = None,
) -> list[Keyframe]:
    require_ffmpeg()
    fps = fps or settings.keyframe_fps
    cut = settings.scene_threshold if scene_threshold is None else scene_threshold
    max_gap = settings.max_frame_gap_seconds if max_gap is None else max_gap
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    times = select_keyframes(_sample_frames(video_path, fps), fps,
                             change=settings.keyframe_change, stable=settings.keyframe_stable,
                             cut=cut, max_gap=max_gap)
    frames = []
    for i, t in enumerate(times, 1):
        path = out_dir / f"frame_{i:05d}.jpg"
        _grab(video_path, t, path, width)
        if path.exists():
            frames.append(Keyframe(timestamp=t, path=str(path)))
    log.info("Extracted %d keyframes from %s", len(frames), video_path.name)
    return frames
