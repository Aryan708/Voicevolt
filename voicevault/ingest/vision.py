"""Caption keyframes with a vision LLM (Anthropic or OpenAI), in parallel."""
from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import Keyframe

log = logging.getLogger(__name__)

PROMPT = (
    "You are indexing a frame from a meeting or lecture video so it can be searched later. "
    "Describe it in 2-4 sentences. Transcribe any visible slide title, headings, key numbers, "
    "chart labels or code exactly. Mention diagrams and what they show. "
    "If it is only a person talking with no useful visual content, reply exactly: TALKING_HEAD"
)


def _b64(path: str) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def _caption_anthropic(path: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    msg = client.messages.create(
        model=settings.anthropic_vision_model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                             "data": _b64(path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _caption_openai(path: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key or None)
    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}},
            ],
        }],
    )
    return (resp.choices[0].message.content or "").strip()


_PROVIDERS = {"anthropic": _caption_anthropic, "openai": _caption_openai}


def caption_frames(
    frames: list[Keyframe], provider: Optional[str] = None, max_workers: int = 4
) -> list[Keyframe]:
    provider = (provider or settings.vision_provider).lower()
    if provider == "none" or not frames:
        return frames
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown VISION_PROVIDER '{provider}'. Use anthropic | openai | none.")

    def work(f: Keyframe) -> Keyframe:
        try:
            cap = fn(f.path)
            f.caption = None if cap == "TALKING_HEAD" else cap
        except Exception as e:  # one bad frame should not kill the whole ingest
            log.warning("Caption failed for %s: %s", f.path, e)
        return f

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        frames = list(pool.map(work, frames))
    log.info("Captioned %d/%d frames with %s",
             sum(f.caption is not None for f in frames), len(frames), provider)
    return frames
