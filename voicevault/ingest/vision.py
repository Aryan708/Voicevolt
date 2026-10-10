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

DOC_PROMPT = (
    "You are indexing a page from a document, slide deck, scanned form or photo so it can be "
    "searched later. First transcribe ALL readable text exactly, keeping headings and table rows. "
    "Then, after a line '---', describe any charts, diagrams, stamps, signatures or photos in "
    "2-3 sentences, including numbers shown on charts. Text may be English or Hindi; keep the "
    "original script."
)

PROMPT = (
    "You are indexing a frame from a meeting or lecture video so it can be searched later. "
    "Describe it in 2-4 sentences. Transcribe any visible slide title, headings, key numbers, "
    "chart labels or code exactly. Mention diagrams and what they show. "
    "If it is only a person talking with no useful visual content, reply exactly: TALKING_HEAD"
)


def _b64(path: str) -> str:
    return base64.standard_b64encode(Path(path).read_bytes()).decode()


def _media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/jpeg")


def _caption_anthropic(path: str, prompt: str = PROMPT, max_tokens: int = 300) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    msg = client.messages.create(
        model=settings.anthropic_vision_model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": _media_type(path),
                                             "data": _b64(path)}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _caption_openai(path: str, prompt: str = PROMPT, max_tokens: int = 300) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key or None)
    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{_media_type(path)};base64,{_b64(path)}"}},
            ],
        }],
    )
    return (resp.choices[0].message.content or "").strip()


_PROVIDERS = {"anthropic": _caption_anthropic, "openai": _caption_openai}


def describe_image(path: str, provider: Optional[str] = None,
                   prompt: str = DOC_PROMPT, max_tokens: int = 1500) -> Optional[str]:
    """OCR + description of a single page/image. Returns None if vision is disabled or fails."""
    provider = (provider or settings.vision_provider).lower()
    if provider == "none":
        return None
    fn = _PROVIDERS.get(provider)
    if fn is None:
        raise ValueError(f"Unknown VISION_PROVIDER '{provider}'. Use anthropic | openai | none.")
    try:
        return fn(path, prompt=prompt, max_tokens=max_tokens) or None
    except Exception as e:
        log.warning("Vision describe failed for %s: %s", path, e)
        return None


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
