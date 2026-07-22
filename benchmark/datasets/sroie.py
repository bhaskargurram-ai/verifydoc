"""SROIE v2 loader (scanned receipts; ICDAR 2019 Task 3) -> VerifyDocBench.

SROIE pairs each receipt with an OCR text layer (per-line boxes) and four gold
key fields: ``company``, ``date``, ``address``, ``total``. We build the text
layer from the box annotations, take gold values from the entities file, and
locate each gold value on the page (numeric-aware for ``total``).

Loads via the HuggingFace hub (``pip install 'verifydoc[data]'``); cached under
``data/``. Unit tests use fixture rows only (golden rule #5) — the pure helpers
below are what's tested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.datasets.synthetic import BenchDocument
from verifydoc.grounding.attach import _locate
from verifydoc.types import Document, FieldGold, Page, Schema, Word

SROIE_SCHEMA_RAW: dict = {
    "type": "object",
    "required": ["company", "total"],
    "properties": {
        "company": {"type": "string", "x-scoring": "semantic"},
        "date": {"type": "string"},
        "address": {"type": "string", "x-scoring": "semantic"},
        "total": {"type": "number", "x-numeric-tol": 0.01, "x-aliases": ["total", "amount"]},
    },
}


def document_from_lines(
    doc_id: str, lines: list[dict[str, Any]], width: int, height: int
) -> Document:
    """Build a text-layer Document from SROIE line boxes.

    Each ``line`` is ``{"text": str, "box": [x0, y0, x1, y1]}`` in pixel coords.
    """
    words: list[Word] = []
    text_lines: list[str] = []
    for line in lines:
        text = str(line.get("text", "")).strip()
        box = line.get("box")
        if not text or not box:
            continue
        x0, y0, x1, y1 = box
        bbox = (
            max(0.0, x0 / width),
            max(0.0, y0 / height),
            min(1.0, x1 / width),
            min(1.0, y1 / height),
        )
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            words.append(Word(text=text, bbox=bbox))
            text_lines.append(text)
    page = Page(
        page=0, width=float(width), height=float(height), text="\n".join(text_lines), words=words
    )
    return Document(doc_id=doc_id, pages=[page])


def golds_from_entities(entities: dict[str, Any], schema: Schema, doc: Document) -> list[FieldGold]:
    """Turn SROIE's four gold entities into scored gold fields with located boxes."""
    golds = []
    for path in ("company", "date", "address", "total"):
        value = entities.get(path)
        if value is None or str(value).strip() == "":
            continue
        leaf = schema.leaf(path)
        golds.append(
            FieldGold(
                path=path,
                value=value,
                scoring=leaf.scoring if leaf else "exact",
                numeric_tol=leaf.numeric_tol if leaf else 1e-6,
                gold_box=_locate(str(value), doc, min_support=0.6, penalize_ambiguity=False),
            )
        )
    return golds


def load(
    split: str = "test", limit: int | None = 100, cache_dir: str | Path = "data"
) -> list[BenchDocument]:  # pragma: no cover - network on first call
    """Load SROIE from the HF hub (``darentang/sroie`` or compatible)."""
    schema = Schema.from_json_schema(SROIE_SCHEMA_RAW, name="sroie-receipt")
    cache = Path(cache_dir) / f"sroie_{split}_{limit or 'all'}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("SROIE loader requires: pip install 'verifydoc[data]'") from exc
        ds = load_dataset("darentang/sroie", split=split, streaming=True)
        rows = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            width, height = row["image"].size if row.get("image") else (1.0, 1.0)
            lines = [
                {"text": t, "box": b}
                for t, b in zip(row.get("words", row.get("text", [])), row.get("bboxes", []))
            ]
            rows.append(
                {
                    "lines": lines,
                    "entities": row.get("entities", {}),
                    "width": width,
                    "height": height,
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")

    out = []
    for i, row in enumerate(rows):
        doc = document_from_lines(
            f"sroie-{split}-{i:05d}", row["lines"], row["width"], row["height"]
        )
        golds = golds_from_entities(row["entities"], schema, doc)
        if golds and doc.pages[0].words:
            out.append(BenchDocument(doc=doc, schema=schema, golds=golds))
    return out
