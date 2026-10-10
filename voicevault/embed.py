"""Embedding models.

Two spaces:
  * text space  - multilingual sentence embedder for chunk text (English + Hindi)
  * CLIP space  - images (page renders, video keyframes) and the query text share
                  one space, so "bar chart of regional revenue" can find a slide even
                  when its text was never extracted.

`EMBED_BACKEND=hash` swaps in tiny deterministic embedders so the whole system
(and the test-suite) runs offline with no model downloads.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)  # works for Devanagari too


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class TextEmbedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class ImageEmbedder(Protocol):
    name: str
    dim: int

    def embed_images(self, paths: Sequence[str]) -> np.ndarray: ...
    def embed_text(self, texts: Sequence[str]) -> np.ndarray: ...


# ---------------------------------------------------------------- real models

def _device() -> str | None:
    return settings.embed_device or None  # sentence-transformers picks cuda/mps/cpu


class STTextEmbedder:
    """sentence-transformers text model, with the query/passage prefixes E5/BGE expect."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or settings.text_embed_model
        log.info("Loading text embedder %s", self.name)
        self.model = SentenceTransformer(self.name, device=_device())
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        lname = self.name.lower()
        if "e5" in lname:
            self._q, self._d = "query: ", "passage: "
        elif "bge" in lname and "m3" not in lname:
            self._q, self._d = "Represent this sentence for searching relevant passages: ", ""
        else:
            self._q = self._d = ""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode([self._d + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self.model.encode([self._q + text], normalize_embeddings=True))[0]


class CLIPEmbedder:
    """CLIP image encoder + (optionally multilingual) text encoder in the same space.

    `clip-ViT-B-32-multilingual-v1` was distilled to match CLIP's text tower, so a Hindi
    query like "क्षेत्रीय राजस्व का चार्ट" lands next to the right slide image.
    """

    def __init__(self, image_model: str | None = None, text_model: str | None = None,
                 batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        self.image_model_name = image_model or settings.clip_image_model
        self.text_model_name = text_model or settings.clip_text_model
        self.name = f"{self.image_model_name}+{self.text_model_name}"
        log.info("Loading CLIP %s", self.name)
        self.img_model = SentenceTransformer(self.image_model_name, device=_device())
        self.txt_model = (self.img_model if self.text_model_name == self.image_model_name
                          else SentenceTransformer(self.text_model_name, device=_device()))
        self.dim = self.img_model.get_sentence_embedding_dimension() or 512
        self.batch_size = batch_size

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(paths), self.batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + self.batch_size]]
            out.append(self.img_model.encode(imgs, batch_size=self.batch_size,
                                             normalize_embeddings=True))
            for im in imgs:
                im.close()
        return _normalize(np.vstack(out)) if out else np.zeros((0, self.dim), np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(self.txt_model.encode(list(texts), normalize_embeddings=True))


# ------------------------------------------------------- offline hash models

def _hash_vec(tokens: list[str], dim: int) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    for t in tokens:
        h = int(hashlib.md5(t.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 64) & 1 else -1.0
    return v if v.any() else np.ones(dim, np.float32)


class HashTextEmbedder:
    """Bag-of-words feature hashing. No semantics, but deterministic and offline."""

    name = "hash-text"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.stack([_hash_vec(tokenize(t), self.dim) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HashImageEmbedder:
    """Stand-in for CLIP. An image's "content" is read from a sidecar `<image>.label.txt`
    if present (used by tests to simulate what CLIP would see), else its file name."""

    name = "hash-image"

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        vecs = []
        for p in paths:
            side = Path(str(p) + ".label.txt")
            label = side.read_text() if side.exists() else Path(p).stem
            vecs.append(_hash_vec(tokenize(label), self.dim))
        return _normalize(np.stack(vecs)) if vecs else np.zeros((0, self.dim), np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.stack([_hash_vec(tokenize(t), self.dim) for t in texts]))


# ------------------------------------------------------------------ factory

@lru_cache(maxsize=1)
def get_text_embedder() -> TextEmbedder:
    return HashTextEmbedder() if settings.embed_backend == "hash" else STTextEmbedder()


@lru_cache(maxsize=1)
def get_image_embedder() -> ImageEmbedder:
    return HashImageEmbedder() if settings.embed_backend == "hash" else CLIPEmbedder()
