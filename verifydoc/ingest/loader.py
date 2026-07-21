"""Document loaders: PDF (via pdfplumber, optional extra), images, plain text.

``document_from_text`` is dependency-free and powers unit tests and the
synthetic benchmark; PDF/image ingestion needs ``pip install verifydoc[pdf]``.
"""

from __future__ import annotations

from pathlib import Path

from verifydoc.types import Document, Page, Word

_PDF_SUFFIXES = {".pdf"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def document_from_text(doc_id: str, pages_text: list[str], line_height: float = 0.04) -> Document:
    """Build a Document from raw page texts with a synthetic geometry layer.

    Each line becomes a row of words laid out left-to-right; bboxes are
    normalized. This gives heuristic adapters and the grounder real spans and
    boxes to work with — no PDF stack required.
    """
    pages = []
    for page_no, text in enumerate(pages_text):
        words: list[Word] = []
        lines = text.split("\n")
        for row, line in enumerate(lines):
            y0 = min(0.98, row * line_height)
            y1 = min(0.99, y0 + line_height * 0.9)
            col = 0
            for token in line.split(" "):
                if token:
                    x0 = min(0.98, 0.02 + col * 0.012)
                    x1 = min(0.99, x0 + max(1, len(token)) * 0.012)
                    words.append(Word(text=token, bbox=(x0, y0, x1, y1)))
                col += len(token) + 1
        pages.append(Page(page=page_no, width=612.0, height=792.0, text=text, words=words))
    return Document(doc_id=doc_id, pages=pages)


def ingest_path(path: str | Path) -> Document:
    """Ingest a file by extension: .pdf (extra), images (extra), else UTF-8 text."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in _PDF_SUFFIXES:
        return _ingest_pdf(p)
    if suffix in _IMAGE_SUFFIXES:
        return _ingest_image(p)
    return document_from_text(p.stem, p.read_text(encoding="utf-8").split("\f"))


def _ingest_pdf(path: Path) -> Document:  # pragma: no cover - needs [pdf] extra
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "PDF ingestion requires the pdf extra: pip install 'verifydoc[pdf]'"
        ) from exc
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages):
            width, height = float(page.width), float(page.height)
            words = [
                Word(
                    text=w["text"],
                    bbox=(
                        float(w["x0"]) / width,
                        float(w["top"]) / height,
                        float(w["x1"]) / width,
                        float(w["bottom"]) / height,
                    ),
                )
                for w in page.extract_words()
            ]
            pages.append(
                Page(
                    page=page_no,
                    width=width,
                    height=height,
                    text=page.extract_text() or "",
                    words=words,
                )
            )
    return Document(doc_id=path.stem, source_path=str(path), pages=pages)


def _ingest_image(path: Path) -> Document:  # pragma: no cover - needs [pdf] extra
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Image ingestion requires the pdf extra: pip install 'verifydoc[pdf]'"
        ) from exc
    with Image.open(path) as img:
        width, height = img.size
    page = Page(page=0, width=float(width), height=float(height), text=None, image_path=str(path))
    return Document(doc_id=path.stem, source_path=str(path), pages=[page])
