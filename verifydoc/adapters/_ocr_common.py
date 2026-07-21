"""Shared OCR-token plumbing for local OCR adapters (PaddleOCR-VL, dots.ocr).

OCR engines return (text, bbox, recognition-score) tokens per page. This
module clusters tokens into reading-order lines, builds a text-layer
``Document``, and lets the shared field-finding heuristic run on top —
attaching each token's recognition score as ``token_logprobs`` so the
token-prob confidence signal works for local OCR models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from verifydoc.adapters.text_search import TextSearchAdapter
from verifydoc.types import Document, FieldPrediction, Page, Schema, Word


@dataclass
class OCRToken:
    """One recognized token: text, normalized bbox, recognition score in (0, 1]."""

    text: str
    bbox: tuple[float, float, float, float]
    score: float
    page: int = 0


def document_from_ocr_tokens(doc_id: str, tokens: list[OCRToken]) -> Document:
    """Cluster OCR tokens into lines (by vertical overlap) and build a Document."""
    pages: dict[int, list[OCRToken]] = {}
    for tok in tokens:
        pages.setdefault(tok.page, []).append(tok)

    out_pages = []
    for page_no in sorted(pages):
        toks = sorted(pages[page_no], key=lambda t: (t.bbox[1], t.bbox[0]))
        lines: list[list[OCRToken]] = []
        for tok in toks:
            placed = False
            for line in lines:
                y0 = sum(t.bbox[1] for t in line) / len(line)
                y1 = sum(t.bbox[3] for t in line) / len(line)
                mid = (tok.bbox[1] + tok.bbox[3]) / 2
                if y0 <= mid <= y1:
                    line.append(tok)
                    placed = True
                    break
            if not placed:
                lines.append([tok])
        for line in lines:
            line.sort(key=lambda t: t.bbox[0])
        text = "\n".join(" ".join(t.text for t in line) for line in lines)
        words = [Word(text=t.text, bbox=t.bbox) for line in lines for t in line]
        out_pages.append(Page(page=page_no, width=1.0, height=1.0, text=text, words=words))
    return Document(doc_id=doc_id, pages=out_pages)


def predictions_from_ocr_tokens(
    doc_id: str, tokens: list[OCRToken], schema: Schema
) -> list[FieldPrediction]:
    """Field predictions from OCR tokens: line clustering + label search +
    per-field token log-scores for the token-prob confidence signal."""
    doc = document_from_ocr_tokens(doc_id, tokens)
    preds = TextSearchAdapter().extract(doc, schema)
    by_text: dict[str, list[float]] = {}
    for tok in tokens:
        by_text.setdefault(tok.text.casefold(), []).append(tok.score)
    out = []
    for pred in preds:
        logprobs = [
            math.log(max(1e-6, min(1.0, by_text[part.casefold()][0])))
            for part in str(pred.value).split()
            if part.casefold() in by_text
        ]
        meta = dict(pred.meta)
        if logprobs:
            meta["token_logprobs"] = logprobs
        out.append(pred.model_copy(update={"meta": meta}))
    return out
