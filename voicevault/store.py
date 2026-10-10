"""Multimodal vector store on Qdrant (embedded/local mode - no server needed).

Two collections:
  chunks  - one point per Chunk, vector = text embedding, payload = full chunk
  images  - one point per image (page render or video keyframe), vector = CLIP
            embedding, payload points back to its chunk_id

Keeping images separate means a 45-second video chunk with 3 slides gets 3
visual entry points, and an image hit can be shown to the user directly.

Set QDRANT_URL to use a Qdrant server / Qdrant Cloud instead of local files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from qdrant_client import QdrantClient, models

from .schemas import Chunk

log = logging.getLogger(__name__)

CHUNKS, IMAGES = "chunks", "images"


def _pid(key: str) -> int:
    """Stable 63-bit point id from any string."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:15], 16)


def make_filter(source: Optional[str] = None,
                modality: Optional[str | list[str]] = None) -> Optional[models.Filter]:
    must = []
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if modality:
        vals = [modality] if isinstance(modality, str) else list(modality)
        must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=vals)))
    return models.Filter(must=must) if must else None


class VectorStore:
    def __init__(self, path: Path, text_dim: int, image_dim: int,
                 text_model: str = "", image_model: str = ""):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        url = os.getenv("QDRANT_URL")
        self.client = (QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
                       if url else QdrantClient(path=str(self.path / "qdrant")))
        self.version = 0  # bumped on every write so BM25 knows to rebuild
        self._check_meta(text_dim, image_dim, text_model, image_model)
        self._ensure(CHUNKS, text_dim)
        self._ensure(IMAGES, image_dim)

    # ---------------------------------------------------------------- setup
    def _check_meta(self, text_dim, image_dim, text_model, image_model) -> None:
        meta_path = self.path / "meta.json"
        meta = {"text_dim": text_dim, "image_dim": image_dim,
                "text_model": text_model, "image_model": image_model}
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old != meta:
                raise RuntimeError(
                    f"Index at {self.path} was built with {old}, but current embedders are "
                    f"{meta}. Re-index with `python scripts/index.py --rebuild`."
                )
        else:
            meta_path.write_text(json.dumps(meta, indent=2))

    def _ensure(self, name: str, dim: int) -> None:
        if not self.client.collection_exists(name):
            self.client.create_collection(
                name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            if os.getenv("QDRANT_URL"):  # payload indexes only matter on a Qdrant server
                for f in ("source", "modality", "chunk_id"):
                    self.client.create_payload_index(name, f, models.PayloadSchemaType.KEYWORD)

    @staticmethod
    def wipe(path: Path) -> None:
        """Delete the whole index (used by --rebuild, e.g. after changing embedding models)."""
        import shutil

        path = Path(path)
        url = os.getenv("QDRANT_URL")
        if url:
            client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
            for name in (CHUNKS, IMAGES):
                if client.collection_exists(name):
                    client.delete_collection(name)
            client.close()
        else:
            shutil.rmtree(path / "qdrant", ignore_errors=True)
        (path / "meta.json").unlink(missing_ok=True)

    # ---------------------------------------------------------------- write
    def delete_source(self, source: str) -> None:
        flt = models.FilterSelector(filter=make_filter(source=source))
        self.client.delete(CHUNKS, points_selector=flt)
        self.client.delete(IMAGES, points_selector=flt)
        self.version += 1

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        points = [
            models.PointStruct(id=_pid(c.id), vector=v.tolist(),
                               payload={**c.model_dump(), "chunk_id": c.id})
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(CHUNKS, points=points)
        self.version += 1

    def add_images(self, records: list[dict], vectors: np.ndarray) -> None:
        """records: {chunk_id, image_path, source, modality, timestamp|None, page|None}"""
        if not records:
            return
        points = [
            models.PointStruct(id=_pid(f"{r['chunk_id']}|{r['image_path']}"),
                               vector=v.tolist(), payload=r)
            for r, v in zip(records, vectors)
        ]
        self.client.upsert(IMAGES, points=points)
        self.version += 1

    # ----------------------------------------------------------------- read
    def search_text(self, vec: np.ndarray, k: int, flt=None) -> list[tuple[str, float]]:
        res = self.client.query_points(CHUNKS, query=vec.tolist(), limit=k,
                                       query_filter=flt, with_payload=["chunk_id"])
        return [(p.payload["chunk_id"], p.score) for p in res.points]

    def search_images(self, vec: np.ndarray, k: int, flt=None) -> list[tuple[str, float, str]]:
        res = self.client.query_points(IMAGES, query=vec.tolist(), limit=k,
                                       query_filter=flt, with_payload=True)
        return [(p.payload["chunk_id"], p.score, p.payload["image_path"]) for p in res.points]

    def get_chunks(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        pts = self.client.retrieve(CHUNKS, ids=[_pid(i) for i in ids], with_payload=True)
        return {p.payload["chunk_id"]: Chunk(**{k: v for k, v in p.payload.items()
                                                 if k != "chunk_id"}) for p in pts}

    def iter_chunks(self, flt=None, batch: int = 256) -> Iterator[Chunk]:
        offset = None
        while True:
            pts, offset = self.client.scroll(CHUNKS, scroll_filter=flt, limit=batch,
                                             offset=offset, with_payload=True)
            for p in pts:
                yield Chunk(**{k: v for k, v in p.payload.items() if k != "chunk_id"})
            if offset is None:
                break

    def stats(self) -> dict:
        by_source = Counter(c.source for c in self.iter_chunks())
        return {
            "chunks": self.client.count(CHUNKS).count,
            "images": self.client.count(IMAGES).count,
            "sources": dict(by_source),
        }

    def close(self) -> None:
        self.client.close()
