"""Central configuration loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass
class Settings:
    # Paths
    root: Path = Path(__file__).resolve().parent.parent
    raw_dir: Path = field(default=None)  # type: ignore[assignment]
    processed_dir: Path = field(default=None)  # type: ignore[assignment]

    # Whisper
    whisper_model: str = _env("WHISPER_MODEL", "small")
    whisper_device: str = _env("WHISPER_DEVICE", "auto")
    whisper_compute_type: str = _env("WHISPER_COMPUTE_TYPE", "int8")

    # Diarization
    hf_token: str = _env("HF_TOKEN")

    # Vision
    vision_provider: str = _env("VISION_PROVIDER", "anthropic").lower()
    anthropic_api_key: str = _env("ANTHROPIC_API_KEY")
    anthropic_vision_model: str = _env("ANTHROPIC_VISION_MODEL", "claude-haiku-4-5")
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_vision_model: str = _env("OPENAI_VISION_MODEL", "gpt-4o-mini")

    # Chunking
    chunk_seconds: float = float(_env("CHUNK_SECONDS", "45"))
    chunk_overlap_seconds: float = float(_env("CHUNK_OVERLAP_SECONDS", "8"))

    # Keyframes
    scene_threshold: float = float(_env("SCENE_THRESHOLD", "0.30"))
    max_frame_gap_seconds: float = float(_env("MAX_FRAME_GAP_SECONDS", "30"))

    def __post_init__(self) -> None:
        self.raw_dir = self.raw_dir or self.root / "data" / "raw"
        self.processed_dir = self.processed_dir or self.root / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
