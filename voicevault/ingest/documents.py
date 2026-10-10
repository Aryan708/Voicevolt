"""Ingest PDFs, slide decks, scans and images into page-level chunks.

Every page is rendered to an image (for CLIP search and for showing the user
the page as proof) and gets text from one of two places:
  * the PDF's own text layer, when it has one (fast, free, exact)
  * a vision LLM reading the page image, for scans, photos and image-only slides
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ..config import settings
from ..schemas import Chunk, IngestResult
from .vision import describe_image

log = logging.getLogger(__name__)

PDF_EXTS = {".pdf"}
OFFICE_EXTS = {".pptx", ".ppt", ".docx", ".doc", ".odp", ".key"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
DOC_EXTS = PDF_EXTS | OFFICE_EXTS | IMAGE_EXTS

MIN_TEXT_CHARS = 40     # below this, a PDF page is treated as scanned / image-only
MAX_CHUNK_CHARS = 1800  # long pages are split, all pieces point at the same page image


def _cid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def office_to_pdf(src: Path, out_dir: Path) -> Path:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if not exe:
        raise RuntimeError(
            f"{src.suffix} needs LibreOffice to convert to PDF "
            "(or export it to PDF yourself and ingest the PDF)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(src)],
                   check=True, capture_output=True, timeout=300)
    pdf = out_dir / f"{src.stem}.pdf"
    if not pdf.exists():
        raise RuntimeError(f"LibreOffice did not produce {pdf.name}")
    return pdf


def render_pdf(pdf: Path, out_dir: Path, scale: float = 1.6) -> list[tuple[int, Path, str]]:
    """Return (page_number, page_image, text_layer) for each page."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    pages = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            img_path = out_dir / f"page_{i + 1:04d}.png"
            page.render(scale=scale).to_pil().convert("RGB").save(img_path, optimize=True)
            textpage = page.get_textpage()
            text = textpage.get_text_range().replace("\r\n", "\n").strip()
            textpage.close()
            page.close()
            pages.append((i + 1, img_path, text))
    finally:
        doc.close()
    return pages


def _split(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on paragraph boundaries, falling back to hard cuts for huge paragraphs."""
    if len(text) <= limit:
        return [text]
    pieces, buf = [], ""
    for para in text.split("\n\n"):
        while len(para) > limit:
            pieces.append(para[:limit])
            para = para[limit:]
        if len(buf) + len(para) + 2 > limit and buf:
            pieces.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def ingest_document(
    path: str | Path,
    vision: bool = True,
    out_dir: Optional[Path] = None,
    max_workers: int = 4,
) -> IngestResult:
    path = Path(path).resolve()
    ext = path.suffix.lower()
    if ext not in DOC_EXTS:
        raise ValueError(f"Unsupported document type: {path.name}")

    from .pipeline import _slug  # shared helper

    work = (out_dir or settings.processed_dir) / _slug(path.stem)
    pages_dir = work / "pages"
    if pages_dir.exists():
        shutil.rmtree(pages_dir)

    if ext in IMAGE_EXTS:
        from PIL import Image

        pages_dir.mkdir(parents=True, exist_ok=True)
        img_path = pages_dir / "page_0001.png"
        Image.open(path).convert("RGB").save(img_path)
        pages = [(1, img_path, "")]
        modality = "image"
    else:
        pdf = office_to_pdf(path, work) if ext in OFFICE_EXTS else path
        pages = render_pdf(pdf, pages_dir)
        modality = "document"

    def page_text(item: tuple[int, Path, str]) -> tuple[int, Path, str, str]:
        num, img, text = item
        if len(text) >= MIN_TEXT_CHARS:
            return num, img, text, "text_layer"
        if vision:
            described = describe_image(str(img))
            if described:
                return num, img, described, "vision"
        return num, img, text, "none"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        processed = list(pool.map(page_text, pages))

    chunks: list[Chunk] = []
    stats = {"text_layer": 0, "vision": 0, "none": 0}
    for num, img, text, how in processed:
        stats[how] += 1
        header = f"[{path.name} p.{num}]"
        for k, piece in enumerate(_split(text) if text else [""]):
            chunks.append(Chunk(
                id=_cid(path.name, num, k), source=path.name, modality=modality,  # type: ignore[arg-type]
                text=f"{header} {piece}".strip(), page=num, image_path=str(img),
            ))

    log.info("%s: %d pages (%s) -> %d chunks", path.name, len(pages),
             ", ".join(f"{k}={v}" for k, v in stats.items() if v), len(chunks))
    if stats["none"] and not vision:
        log.info("  %d page(s) had no text layer; enable vision to OCR them.", stats["none"])

    result = IngestResult(source=path.name, modality=modality,  # type: ignore[arg-type]
                          chunks=chunks, timings={})
    work.mkdir(parents=True, exist_ok=True)
    (work / "ingest.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
