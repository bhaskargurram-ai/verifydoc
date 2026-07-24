"""DocILE loader (business-document KILE, 55 field classes) -> VerifyDocBench.

DocILE annotates invoices with Key Information Localization and Extraction
(KILE) fields — each field has a page, a relative bbox (0-1), a field type,
and the text. We build a text layer from the OCR word boxes, flatten the
KILE annotations into scored gold fields, and locate each gold value on the
page (numeric-aware for amount-like fields).

Loads via the HuggingFace hub (``pip install 'verifydoc[data]'``); cached
under ``data/``. Unit tests use fixture rows only (golden rule #5) — the
pure helpers below are what's tested.

License note (benchmark/card.md): DocILE is released for research use; we
ship the loader + our added splits, never the PDFs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.datasets.synthetic import BenchDocument
from verifydoc.grounding.attach import _locate
from verifydoc.types import Document, FieldGold, Page, Schema, Word

# Subset of DocILE's 55 KILE field types that map cleanly to our schema.
# Full list: github.com/rossumai/docile — we expose the common invoice header
# fields; line-item fields (LIR) are out of scope for this loader.
DOCILE_FIELDS: dict[str, dict[str, Any]] = {
    "invoice_number": {"type": "string", "x-scoring": "exact"},
    "invoice_date": {"type": "string", "x-aliases": ["date"]},
    "due_date": {"type": "string"},
    "vendor": {"type": "string", "x-scoring": "semantic"},
    "customer": {"type": "string", "x-scoring": "semantic"},
    "iban": {"type": "string"},
    "vat_number": {"type": "string", "x-aliases": ["vat", "tax_id"]},
    "amount_total": {"type": "number", "x-numeric-tol": 0.01, "x-aliases": ["total"]},
    "amount_due": {"type": "number", "x-numeric-tol": 0.01, "x-aliases": ["due"]},
    "currency": {"type": "string"},
}

DOCILE_SCHEMA_RAW: dict = {
    "type": "object",
    "properties": {
        path: {k: v for k, v in spec.items() if k != "x-aliases"}
        | ({"x-aliases": spec["x-aliases"]} if "x-aliases" in spec else {})
        for path, spec in DOCILE_FIELDS.items()
    },
}


def document_from_words(
    doc_id: str,
    words: list[dict[str, Any]],
    width: float,
    height: float,
    page: int = 0,
) -> Document:
    """Build a text-layer Document from DocILE OCR word boxes.

    Each word is ``{"text": str, "bbox": [x0, y0, x1, y1]}`` with bbox in
    relative coordinates (0-1).
    """
    page_words: list[Word] = []
    lines: list[str] = []
    for w in words:
        text = str(w.get("text", "")).strip()
        bbox = w.get("bbox")
        if not text or not bbox:
            continue
        x0, y0, x1, y1 = bbox
        norm = (
            max(0.0, min(1.0, x0)),
            max(0.0, min(1.0, y0)),
            max(0.0, min(1.0, x1)),
            max(0.0, min(1.0, y1)),
        )
        if norm[2] > norm[0] and norm[3] > norm[1]:
            page_words.append(Word(text=text, bbox=norm))
            lines.append(text)
    return Document(
        doc_id=doc_id,
        pages=[
            Page(
                page=page,
                width=float(width),
                height=float(height),
                text="\n".join(lines),
                words=page_words,
            )
        ],
    )


def golds_from_annotations(
    annotations: list[dict[str, Any]], schema: Schema, doc: Document
) -> list[FieldGold]:
    """Turn DocILE KILE field annotations into scored gold fields with boxes.

    Each annotation is ``{"page": int, "bbox": [l,t,r,b], "fieldtype": str,
    "text": str}``. Line-item fields (those with ``line_item_id``) are
    skipped — only KILE header fields are mapped.
    """
    golds: list[FieldGold] = []
    seen: set[str] = set()
    for ann in annotations:
        if ann.get("line_item_id") is not None:
            continue
        fieldtype = ann.get("fieldtype", "")
        if fieldtype not in DOCILE_FIELDS:
            continue
        value = str(ann.get("text", "")).strip()
        if not value:
            continue
        path = fieldtype
        if path in seen:
            continue
        seen.add(path)
        leaf = schema.leaf(path)
        scoring = leaf.scoring if leaf else (
            "numeric" if DOCILE_FIELDS[path]["type"] == "number" else "exact"
        )
        numeric_tol = leaf.numeric_tol if leaf else 1e-6
        gold_box = _locate(value, doc, min_support=0.6, penalty_mode="none")
        golds.append(
            FieldGold(
                path=path,
                value=value,
                scoring=scoring,
                numeric_tol=numeric_tol,
                gold_box=gold_box,
            )
        )
    return golds


def load(
    split: str = "val",
    limit: int | None = 100,
    cache_dir: str | Path = "data",
) -> list[BenchDocument]:  # pragma: no cover - network on first call
    """Load DocILE from the HF hub (``rossumai/docile`` or compatible)."""
    schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile-invoice")
    cache = Path(cache_dir) / f"docile_{split}_{limit or 'all'}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("DocILE loader requires: pip install 'verifydoc[data]'") from exc
        ds = load_dataset("rossumai/docile", split=split, streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            width = float(row.get("page_width", 1.0))
            height = float(row.get("page_height", 1.0))
            words = [
                {"text": w.get("text", ""), "bbox": w.get("bbox", [0, 0, 0, 0])}
                for w in row.get("ocr_words", [])
            ]
            rows.append(
                {
                    "words": words,
                    "annotations": row.get("annotations", []),
                    "width": width,
                    "height": height,
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")

    out: list[BenchDocument] = []
    for i, row in enumerate(rows):
        doc = document_from_words(
            f"docile-{split}-{i:05d}", row["words"], row["width"], row["height"]
        )
        golds = golds_from_annotations(row["annotations"], schema, doc)
        if golds and doc.pages[0].words:
            out.append(BenchDocument(doc=doc, schema=schema, golds=golds))
    return out
