import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from voicevault.config import settings  # noqa: E402
from voicevault.embed import HashImageEmbedder, HashTextEmbedder  # noqa: E402
from voicevault.retrieval import HybridRetriever, Indexer, open_store  # noqa: E402
from voicevault.schemas import Chunk, IngestResult, Keyframe  # noqa: E402


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




ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True, scope="session")
def _repo_stays_clean():
    """Fail the run if any test writes into the repository's data/ folder."""
    def snapshot():
        return {p for p in (ROOT / "data").rglob("*") if p.name != ".gitkeep"}
    before = snapshot()
    yield
    leaked = snapshot() - before
    assert not leaked, f"tests wrote into the repo: {sorted(map(str, leaked))[:5]}"
