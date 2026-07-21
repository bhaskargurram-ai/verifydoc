"""RapidOCR adapter (PP-OCR models via ONNX Runtime; CPU or GPU).

RapidOCR ships the PaddleOCR PP-OCR detection/recognition models exported to
ONNX, so it runs through onnxruntime with no PaddlePaddle dependency — which
means it is architecture-independent (works where paddle's precompiled GPU
kernels do not, e.g. very new GPUs, and on CPU everywhere). Each recognized
box carries a recognition confidence, which feeds the token-prob signal via
the shared, unit-tested ``_ocr_common`` pathway.

Install: ``pip install rapidocr onnxruntime`` (or ``rapidocr-onnxruntime``).
"""

from __future__ import annotations

from typing import Any

from verifydoc.adapters._ocr_common import OCRToken, predictions_from_ocr_tokens
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Page, Schema


class RapidOCRAdapter(ExtractorAdapter):
    name = "rapidocr"

    def __init__(self) -> None:  # pragma: no cover - needs SDK
        engine, api = _load_engine()
        self._engine = engine
        self._api = api

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:  # pragma: no cover
        tokens: list[OCRToken] = []
        for page in doc.pages:
            if page.image_path is None:
                continue
            tokens.extend(self._page_tokens(page))
        return predictions_from_ocr_tokens(doc.doc_id, tokens, schema)

    def _page_tokens(self, page: Page) -> list[OCRToken]:  # pragma: no cover - needs SDK
        raw = self._engine(page.image_path)
        tokens = []
        for text, score, poly in _iter_results(raw, self._api):
            tok = _token_from_poly(text, score, poly, page)
            if tok is not None:
                tokens.append(tok)
        return tokens


def _load_engine() -> tuple[Any, str]:  # pragma: no cover - needs SDK
    """Return (engine, api) supporting both rapidocr v2 and v1 packages."""
    try:  # v2 unified package: engine(img) -> result object with .boxes/.txts/.scores
        from rapidocr import RapidOCR

        return RapidOCR(), "v2"
    except ImportError:
        pass
    try:  # v1: engine(img) -> (list[[box, text, score]], elapse)
        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR(), "v1"
    except ImportError as exc:
        raise ImportError(
            "RapidOCRAdapter requires rapidocr: pip install rapidocr onnxruntime"
        ) from exc


def _iter_results(raw: Any, api: str) -> list[tuple[str, float, Any]]:  # pragma: no cover
    """Normalize either package's output into (text, score, 4-point poly) tuples."""
    if api == "v2":
        if raw is None or raw.boxes is None:
            return []
        return [
            (str(txt), float(score), box)
            for box, txt, score in zip(raw.boxes, raw.txts, raw.scores)
        ]
    result = raw[0] if isinstance(raw, tuple) else raw
    if not result:
        return []
    return [(str(text), float(score), box) for box, text, score in result]


def _token_from_poly(
    text: str, score: float, poly: Any, page: Page
) -> OCRToken | None:  # pragma: no cover - exercised via SDK paths
    xs = [float(pt[0]) for pt in poly]
    ys = [float(pt[1]) for pt in poly]
    bbox = (
        max(0.0, min(xs) / page.width),
        max(0.0, min(ys) / page.height),
        min(1.0, max(xs) / page.width),
        min(1.0, max(ys) / page.height),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or not text.strip():
        return None
    return OCRToken(text=text, bbox=bbox, score=max(1e-6, min(1.0, score)), page=page.page)
