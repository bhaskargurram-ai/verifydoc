"""PaddleOCR-VL adapter (local, single-GPU default extractor).

Requires ``pip install paddleocr`` and a GPU (or slow CPU). All SDK use is
isolated here; the token -> field mapping is the shared, unit-tested
``_ocr_common`` pathway.
"""

from __future__ import annotations

from verifydoc.adapters._ocr_common import OCRToken, predictions_from_ocr_tokens
from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema


class PaddleOCRVLAdapter(ExtractorAdapter):
    name = "paddleocr-vl"

    def __init__(self, lang: str = "en") -> None:  # pragma: no cover - needs SDK
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "PaddleOCRVLAdapter requires the paddleocr package: pip install paddleocr"
            ) from exc
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:  # pragma: no cover
        tokens: list[OCRToken] = []
        for page in doc.pages:
            if page.image_path is None:
                continue
            result = self._ocr.ocr(page.image_path, cls=True)
            for line in result[0] or []:
                quad, (text, score) = line
                xs = [pt[0] for pt in quad]
                ys = [pt[1] for pt in quad]
                tokens.append(
                    OCRToken(
                        text=text,
                        bbox=(
                            min(xs) / page.width,
                            min(ys) / page.height,
                            max(xs) / page.width,
                            max(ys) / page.height,
                        ),
                        score=float(score),
                        page=page.page,
                    )
                )
        return predictions_from_ocr_tokens(doc.doc_id, tokens, schema)
