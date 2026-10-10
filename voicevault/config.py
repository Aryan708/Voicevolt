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
    anthropic_vision_model: str = _env("ANTHROPIC_VISION_MODEL", "claude-haiku-5-5")
    openai_api_key: str = _env("OPENAI_API_KEY")
    openai_vision_model: str = _env("OPENAI_VISION_MODEL", "gpt-4o-mini")

    # Chunking
    chunk_seconds: float = float(_env("CHUNK_SECONDS", "45"))
    chunk_overlap_seconds: float = float(_env("CHUNK_OVERLAP_SECONDS", "8"))

    # Keyframes
    scene_threshold: float = float(_env("SCENE_THRESHOLD", "0.30"))
    max_frame_gap_seconds: float = float(_env("MAX_FRAME_GAP_SECONDS", "30"))

    # Embeddings (Week 2)
    embed_backend: str = _env("EMBED_BACKEND", "st").lower()  # st | hash (offline/testing)
    text_embed_model: str = _env("TEXT_EMBED_MODEL", "intfloat/multilingual-e5-small")
    clip_image_model: str = _env("CLIP_IMAGE_MODEL", "clip-ViT-B-32")
    clip_text_model: str = _env(
        "CLIP_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )
    embed_device: str = _env("EMBED_DEVICE", "")  # "" = auto
    index_dir: Path = field(default=None)  # type: ignore[assignment]

    # Retrieval
    rrf_k: int = int(_env("RRF_K", "60"))
    weight_dense: float = float(_env("WEIGHT_DENSE", "1.0"))
    weight_bm25: float = float(_env("WEIGHT_BM25", "0.8"))
    weight_visual: float = float(_env("WEIGHT_VISUAL", "0.6"))

    # LLM (Week 3)  provider: anthropic | openai | ollama (Llama etc. via OpenAI-compatible API)
    llm_provider: str = _env("LLM_PROVIDER", "anthropic").lower()
    llm_model: str = _env("LLM_MODEL", "")
    llm_base_url: str = _env("LLM_BASE_URL", "")  # e.g. http://localhost:11434/v1 for Ollama
    llm_max_tokens: int = int(_env("LLM_MAX_TOKENS", "700"))
    retrieval_k: int = int(_env("RETRIEVAL_K", "5"))

    # Voice
    tts_provider: str = _env("TTS_PROVIDER", "edge").lower()  # edge | openai | piper | none
    tts_voice_en: str = _env("TTS_VOICE_EN", "en-IN-NeerjaNeural")
    tts_voice_hi: str = _env("TTS_VOICE_HI", "hi-IN-SwaraNeural")
    piper_model_en: str = _env("PIPER_MODEL_EN", "")
    piper_model_hi: str = _env("PIPER_MODEL_HI", "")
    stt_provider: str = _env("STT_PROVIDER", "whisper").lower()  # whisper | fake
    stt_model: str = _env("STT_MODEL", "")       # blank = WHISPER_MODEL; or fine-tuned path
    vad_silence_ms: int = int(_env("VAD_SILENCE_MS", "700"))
    barge_in: bool = _env("BARGE_IN", "true").lower() in {"1", "true", "yes"}

    def default_llm_model(self) -> str:
        return self.llm_model or {
            "anthropic": "claude-haiku-5-5",
            "openai": "gpt-4o-mini",
            "ollama": "llama3.1:8b",
        }.get(self.llm_provider, "")

    @property
    def metrics_path(self) -> Path:
        return self.root / "data" / "logs" / "metrics.jsonl"

    @property
    def feedback_path(self) -> Path:
        return self.root / "data" / "logs" / "feedback.jsonl"

    def __post_init__(self) -> None:
        self.raw_dir = self.raw_dir or self.root / "data" / "raw"
        self.processed_dir = self.processed_dir or self.root / "data" / "processed"
        self.index_dir = self.index_dir or Path(_env("INDEX_DIR") or self.root / "data" / "index")
        self.processed_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
