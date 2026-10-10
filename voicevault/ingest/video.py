"""Keyframe extraction: scene-change detection with a guaranteed max gap."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import Keyframe
from .media import require_ffmpeg

log = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def extract_keyframes(
    video_path: Path,
    out_dir: Path,
    scene_threshold: Optional[float] = None,
    max_gap: Optional[float] = None,
    width: int = 768,
) -> list[Keyframe]:
    """Pick frames where the picture changes (new slide, new scene).

    One ffmpeg pass selects a frame when ANY of these is true:
      * it is the first frame
      * the scene-change score is above `scene_threshold`
      * `max_gap` seconds have passed since the last selected frame
    The last rule guarantees coverage for talking-head videos where
    nothing "changes" but the speaker may point at something.
    """
    require_ffmpeg()
    scene_threshold = settings.scene_threshold if scene_threshold is None else scene_threshold
    max_gap = settings.max_frame_gap_seconds if max_gap is None else max_gap
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()

    select = (
        f"isnan(prev_selected_t)"
        f"+gt(scene\\,{scene_threshold})"
        f"+gte(t-prev_selected_t\\,{max_gap})"
    )
    vf = f"select='{select}',showinfo,scale={width}:-2"
    cmd = [
        "ffmpeg", "-hide_banner", "-y", "-i", str(video_path),
        "-vf", vf, "-fps_mode", "vfr", "-q:v", "3",
        str(out_dir / "frame_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg keyframe extraction failed:\n{proc.stderr[-2000:]}")

    times = [float(m.group(1)) for line in proc.stderr.splitlines()
             if "Parsed_showinfo" in line and (m := _PTS_RE.search(line))]
    files = sorted(out_dir.glob("frame_*.jpg"))
    if len(times) != len(files):
        log.warning("Frame/timestamp count mismatch (%d vs %d); truncating.",
                    len(files), len(times))
    frames = [Keyframe(timestamp=round(t, 2), path=str(f)) for t, f in zip(times, files, strict=False)]
    log.info("Extracted %d keyframes from %s", len(frames), video_path.name)
    return frames
