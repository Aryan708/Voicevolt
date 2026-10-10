"""Indexing + hybrid multimodal retrieval.

A query is answered by three retrievers, fused with weighted Reciprocal Rank Fusion:

  dense   - multilingual text embedding vs chunk text     (meaning, paraphrase, Hindi)
  bm25    - keyword match vs chunk text                    (names, numbers, invoice IDs)
  visual  - CLIP text embedding vs page/keyframe images    (charts, diagrams, scans)

RRF combines ranks, not raw scores, so the three very differently-scaled
similarity scores never have to be calibrated against each other.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embed import ImageEmbedder, TextEmbedder, get_image_embedder, get_text_embedder, tokenize
from .schemas import Chunk, IngestResult
from .store import VectorStore, make_filter

log = logging.getLogger(__name__)

_STOP = set("""a an the and or of to in on at for is are was were be been it this that with
as by from what which who whom how why when where did does do about into over under than
then there their they them we you he she his her our your i me my ka ki ke hai है का की के में
""".split())


def open_store(text_emb: Optional[TextEmbedder] = None,
               img_emb: Optional[ImageEmbedder] = None,
               rebuild: bool = False) -> tuple[VectorStore, TextEmbedder, ImageEmbedder]:
    text_emb = text_emb or get_text_embedder()
    img_emb = img_emb or get_image_embedder()
    if rebuild:
        VectorStore.wipe(settings.index_dir)
    store = VectorStore(settings.index_dir, text_emb.dim, img_emb.dim,
                        text_emb.name, img_emb.name)
    return store, text_emb, img_emb


# ------------------------------------------------------------------- indexing

class Indexer:
    def __init__(self, store: VectorStore, text_emb: TextEmbedder, img_emb: ImageEmbedder):
        self.store, self.text_emb, self.img_emb = store, text_emb, img_emb

    def index(self, result: IngestResult) -> dict:
        chunks = [c for c in result.chunks if c.text.strip() or c.image_paths()]
        self.store.delete_source(result.source)  # re-indexing a file replaces it

        text_vecs = self.text_emb.embed_documents([c.text or c.citation for c in chunks])
        self.store.add_chunks(chunks, text_vecs)

        # Skip "talking head" frames when captions were generated - they add visual noise.
        captioned = any(f.caption for f in result.keyframes)
        records = []
        for c in chunks:
            if c.image_path:
                records.append({"chunk_id": c.id, "image_path": c.image_path,
                                "source": c.source, "modality": c.modality,
                                "page": c.page, "timestamp": None})
            for f in c.frames:
                if captioned and not f.caption:
                    continue
                records.append({"chunk_id": c.id, "image_path": f.path,
                                "source": c.source, "modality": c.modality,
                                "page": None, "timestamp": f.timestamp})
        # A page split into several text chunks shares one image: embed each file once.
        uniq = list(dict.fromkeys(r["image_path"] for r in records))
        if uniq:
            vec_by_path = dict(zip(uniq, self.img_emb.embed_images(uniq)))
            self.store.add_images(records, np.stack([vec_by_path[r["image_path"]]
                                                     for r in records]))
        log.info("Indexed %s: %d chunks, %d images", result.source, len(chunks), len(records))
        return {"source": result.source, "chunks": len(chunks), "images": len(records)}


# ------------------------------------------------------------------ retrieval

@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)   # retriever -> 1-based rank
    image: Optional[str] = None                           # best image to show as proof

    @property
    def matched_by(self) -> str:
        return "+".join(sorted(self.ranks))


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [[t for t in tokenize(c.text) if t not in _STOP] or ["_"] for c in chunks]
        self.bm25 = BM25Okapi(docs) if chunks else None

    def search(self, query: str, k: int, source=None, modality=None) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        q = [t for t in tokenize(query) if t not in _STOP]
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        mods = {modality} if isinstance(modality, str) else set(modality or [])
        order = np.argsort(-scores)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if (source and c.source != source) or (mods and c.modality not in mods):
                continue
            out.append((c.id, float(scores[i])))
            if len(out) >= k:
                break
        return out


class HybridRetriever:
    def __init__(self, store: VectorStore, text_emb: TextEmbedder, img_emb: ImageEmbedder):
        self.store, self.text_emb, self.img_emb = store, text_emb, img_emb
        self._bm25: Optional[BM25Index] = None
        self._bm25_version = -1

    def _bm25_index(self) -> BM25Index:
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25Index(list(self.store.iter_chunks()))
            self._bm25_version = self.store.version
        return self._bm25

    def search(
        self,
        query: str,
        k: int = 5,
        source: Optional[str] = None,
        modality: Optional[str | list[str]] = None,
        retrievers: tuple[str, ...] = ("dense", "bm25", "visual"),
    ) -> list[Hit]:
        pool = max(k * 4, 20)
        flt = make_filter(source=source, modality=modality)
        ranked: dict[str, list[str]] = {}
        best_image: dict[str, str] = {}

        if "dense" in retrievers:
            ranked["dense"] = [cid for cid, _ in
                               self.store.search_text(self.text_emb.embed_query(query), pool, flt)]
        if "bm25" in retrievers:
            ranked["bm25"] = [cid for cid, _ in
                              self._bm25_index().search(query, pool, source, modality)]
        if "visual" in retrievers:
            qv = self.img_emb.embed_text([query])[0]
            seen: list[str] = []
            for cid, _, img in self.store.search_images(qv, pool, flt):
                if cid not in best_image:   # first (best) image per chunk
                    best_image[cid] = img
                    seen.append(cid)
            ranked["visual"] = seen

        weights = {"dense": settings.weight_dense, "bm25": settings.weight_bm25,
                   "visual": settings.weight_visual}
        fused = rrf(ranked, weights, settings.rrf_k)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        chunks = self.store.get_chunks([cid for cid, _ in top])

        hits = []
        for cid, score in top:
            if cid not in chunks:
                continue
            c = chunks[cid]
            ranks = {name: lst.index(cid) + 1 for name, lst in ranked.items() if cid in lst}
            imgs = c.image_paths()
            hits.append(Hit(chunk=c, score=round(score, 5), ranks=ranks,
                            image=best_image.get(cid) or (imgs[0] if imgs else None)))
        return hits


def rrf(ranked: dict[str, list[str]], weights: dict[str, float], k: int = 60) -> dict[str, float]:
    """Weighted Reciprocal Rank Fusion: score(d) = sum_r  w_r / (k + rank_r(d))."""
    scores: dict[str, float] = {}
    for name, ids in ranked.items():
        w = weights.get(name, 1.0)
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
    return scores
