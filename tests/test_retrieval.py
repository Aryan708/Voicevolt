"""Week 2 tests: document ingestion, vector store, hybrid retrieval.

Use the offline hash embedders, so they run without model downloads.
HashImageEmbedder reads `<image>.label.txt` to simulate what CLIP "sees".
"""
from pathlib import Path

import pytest
from PIL import Image

from voicevault.config import settings
from voicevault.embed import HashImageEmbedder, HashTextEmbedder
from voicevault.retrieval import HybridRetriever, Indexer, open_store, rrf
from voicevault.schemas import Chunk, IngestResult, Keyframe


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "index")
    store, temb, iemb = open_store(HashTextEmbedder(), HashImageEmbedder())
    yield store, Indexer(store, temb, iemb), HybridRetriever(store, temb, iemb), tmp_path
    store.close()


def img(path: Path, sees: str) -> str:
    """Create an image and record what a CLIP model would 'see' in it."""
    Image.new("RGB", (64, 64), "white").save(path)
    Path(str(path) + ".label.txt").write_text(sees)
    return str(path)


def corpus(tmp: Path) -> list[IngestResult]:
    f1 = img(tmp / "f1.jpg", "bar chart regional revenue north south")
    f2 = img(tmp / "f2.jpg", "person talking webcam")
    meeting = IngestResult(source="sync.mp4", modality="video", keyframes=[
        Keyframe(timestamp=754, path=f1, caption="Bar chart: Q3 revenue by region"),
        Keyframe(timestamp=790, path=f2, caption=None)], chunks=[
        Chunk(id="m1", source="sync.mp4", modality="video", start=750, end=795,
              text="[12:30 SPEAKER_01] The Chennai launch slipped to Q4 because of vendor delays.",
              frames=[Keyframe(timestamp=754, path=f1, caption="Bar chart: Q3 revenue by region"),
                      Keyframe(timestamp=790, path=f2, caption=None)]),
        Chunk(id="m2", source="sync.mp4", modality="video", start=800, end=845,
              text="[13:20 SPEAKER_00] Hiring plan: two ML engineers in Bengaluru next quarter."),
    ])
    doc = IngestResult(source="policy.pdf", modality="document", chunks=[
        Chunk(id="d1", source="policy.pdf", modality="document", page=3,
              text="[policy.pdf p.3] Travel reimbursement is capped at 5000 rupees per trip.",
              image_path=img(tmp / "p3.png", "text page policy")),
    ])
    scan = IngestResult(source="scan.pdf", modality="document", chunks=[
        Chunk(id="s1", source="scan.pdf", modality="document", page=1,
              text="[scan.pdf p.1]",  # no text layer, no OCR - only the image can find it
              image_path=img(tmp / "s1.png", "scanned invoice 4471 vendor stamp")),
    ])
    return [meeting, doc, scan]


# ------------------------------------------------------------------ fusion

def test_rrf_rewards_agreement():
    scores = rrf({"dense": ["a", "b", "c"], "bm25": ["b", "a"], "visual": ["c"]},
                 {"dense": 1, "bm25": 1, "visual": 1}, k=60)
    assert max(scores, key=scores.get) in {"a", "b"}
    assert scores["b"] > scores["c"]          # found by 2 retrievers beats found by 1
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)


# --------------------------------------------------------------- retrieval

def test_keyword_question_finds_meeting_with_timestamp(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    hits = retriever.search("Why did the Chennai launch slip?", k=3)
    top = hits[0]
    assert top.chunk.id == "m1"
    assert top.chunk.citation == "sync.mp4 @ 12:30-13:15"
    assert "bm25" in top.ranks and "dense" in top.ranks


def test_visual_only_document_found_through_image(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    hits = retriever.search("scanned invoice 4471", k=3)
    assert hits[0].chunk.id == "s1"
    assert hits[0].ranks.get("visual") == 1
    assert hits[0].image.endswith("s1.png")


def test_chart_query_returns_matching_keyframe_as_proof(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    hits = retriever.search("regional revenue bar chart", k=3, retrievers=("visual",))
    assert hits[0].chunk.id == "m1"
    assert hits[0].image.endswith("f1.jpg")


def test_talking_head_frames_not_indexed(kb):
    store, indexer, retriever, tmp = kb
    stats = indexer.index(corpus(tmp)[0])
    assert stats["images"] == 1  # f1 only; f2 had no caption (TALKING_HEAD)


def test_filters(kb):
    store, indexer, retriever, tmp = kb
    for r in corpus(tmp):
        indexer.index(r)
    hits = retriever.search("reimbursement rupees", k=5, modality="video")
    assert hits and all(h.chunk.modality == "video" for h in hits)
    hits = retriever.search("launch", k=5, source="policy.pdf")
    assert all(h.chunk.source == "policy.pdf" for h in hits)


def test_reindex_replaces_not_duplicates(kb):
    store, indexer, retriever, tmp = kb
    meeting = corpus(tmp)[0]
    indexer.index(meeting)
    indexer.index(meeting)
    s = store.stats()
    assert s["sources"] == {"sync.mp4": 2}
    assert s["images"] == 1


def test_bm25_refreshes_after_new_index(kb):
    store, indexer, retriever, tmp = kb
    meeting, doc, _ = corpus(tmp)
    indexer.index(meeting)
    assert retriever.search("reimbursement", k=1, retrievers=("bm25",)) == []
    indexer.index(doc)
    assert retriever.search("reimbursement", k=1, retrievers=("bm25",))[0].chunk.id == "d1"


def test_embedder_change_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "index_dir", tmp_path / "idx")
    store, *_ = open_store(HashTextEmbedder(dim=256), HashImageEmbedder())
    store.close()
    with pytest.raises(RuntimeError, match="--rebuild"):
        open_store(HashTextEmbedder(dim=128), HashImageEmbedder())
    store, *_ = open_store(HashTextEmbedder(dim=128), HashImageEmbedder(), rebuild=True)
    store.close()


# ------------------------------------------------------- document ingestion

def test_scanned_pdf_and_image_ingest(tmp_path):
    from voicevault.ingest.documents import ingest_document

    scan = tmp_path / "scan.pdf"
    Image.new("RGB", (400, 300), "white").save(scan)
    r = ingest_document(scan, vision=False, out_dir=tmp_path / "out")
    assert r.modality == "document" and len(r.chunks) == 1
    assert r.chunks[0].page == 1 and Path(r.chunks[0].image_path).exists()

    photo = tmp_path / "whiteboard.jpg"
    Image.new("RGB", (300, 200), "gray").save(photo)
    r = ingest_document(photo, vision=False, out_dir=tmp_path / "out")
    assert r.modality == "image" and r.chunks[0].citation == "whiteboard.jpg p.1"


def test_text_pdf_uses_text_layer_and_splits_long_pages(tmp_path):
    canvas = pytest.importorskip("reportlab.pdfgen.canvas")
    from voicevault.ingest import documents

    pdf = tmp_path / "report.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 750, "Q3 Revenue Report: North region grew 12 percent.")
    c.showPage()
    y = 800
    for i in range(60):
        c.drawString(40, y, f"Line {i}: " + "detailed vendor contract clause text " * 3)
        y -= 12
        if y < 40:
            break
    c.save()

    r = documents.ingest_document(pdf, vision=False, out_dir=tmp_path / "out")
    p1 = [ch for ch in r.chunks if ch.page == 1]
    p2 = [ch for ch in r.chunks if ch.page == 2]
    assert "North region grew 12 percent" in p1[0].text
    assert len(p2) >= 2, "long page should be split"
    assert len({ch.image_path for ch in p2}) == 1, "split pieces share the page image"
    assert all(len(ch.text) <= documents.MAX_CHUNK_CHARS + 40 for ch in p2)
