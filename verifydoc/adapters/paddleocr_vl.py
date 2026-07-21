"""PaddleOCR adapter (local, single-GPU default extractor).

Supports both the PaddleOCR 3.x pipeline API (``predict``) and the legacy
2.x API (``ocr``); detection is automatic at construction. All SDK use is
isolated here; the token -> field mapping is the shared, unit-tested
``_ocr_common`` pathway, which also feeds recognition scores into the
token-prob confidence signal.
"""

from __future__ import annotations

from typing import Any

from verifydoc.adapters._ocr_common import OCRToken, predictions_from_ocr_tokens
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Page, Schema


class PaddleOCRVLAdapter(ExtractorAdapter):
    name = "paddleocr-vl"

    def __init__(self, lang: str = "en") -> None:  # pragma: no cover - needs SDK
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCRVLAdapter requires the paddleocr package: pip install paddleocr"
            ) from exc
        try:  # 3.x pipeline API
            self._ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            self._api = "predict"
        except TypeError:  # 2.x legacy API
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            self._api = "ocr"

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:  # pragma: no cover
        tokens: list[OCRToken] = []
        for page in doc.pages:
            if page.image_path is None:
                continue
            tokens.extend(
                self._page_tokens(page)
                if self._api == "predict"
                else self._page_tokens_legacy(page)
            )
        return predictions_from_ocr_tokens(doc.doc_id, tokens, schema)

    def _page_tokens(self, page: Page) -> list[OCRToken]:  # pragma: no cover - needs SDK
        result = self._ocr.predict(page.image_path)
        if not result:
            return []
        res: Any = result[0]
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res.get("rec_polys") if hasattr(res, "get") else res["rec_polys"]
        if polys is None:
            polys = res["dt_polys"]
        return [
            tok
            for text, score, poly in zip(texts, scores, polys)
            if (tok := _token_from_poly(text, float(score), poly, page)) is not None
        ]

    def _page_tokens_legacy(self, page: Page) -> list[OCRToken]:  # pragma: no cover - needs SDK
        result = self._ocr.ocr(page.image_path, cls=True)
        tokens = []
        for line in result[0] or []:
            quad, (text, score) = line
            tok = _token_from_poly(text, float(score), quad, page)
            if tok is not None:
                tokens.append(tok)
        return tokens


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
