"""CORD v2 loader (receipts, CC-BY-4.0) -> VerifyDocBench items with REAL
text layers and gold boxes.

Pages are built from CORD's ``valid_line`` word quads (normalized to image
size), gold values from ``gt_parse`` (flattened to leaf paths with our path
convention, so arrays like ``menu[0].price`` score natively), and gold boxes
are located on the real page. Downloads stream from the HF hub
(pip install 'verifydoc[data]'); results are cached as JSON under ``data/``
so a run downloads once. Never imported by unit tests (golden rule #5) —
pure helpers below are tested with fixture rows instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.datasets.synthetic import BenchDocument
from verifydoc.grounding.attach import _locate
from verifydoc.types import Document, FieldGold, Page, Schema, Word, flatten_json

CORD_SCHEMA_RAW: dict = {
    "type": "object",
    "properties": {
        "menu": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "nm": {"type": "string", "x-scoring": "semantic"},
                    "cnt": {"type": "string"},
                    "price": {"type": "number", "x-numeric-tol": 0.01},
                },
            },
        },
        "sub_total": {
            "type": "object",
            "properties": {
                "subtotal_price": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["subtotal", "sub total", "sub-total"],
                },
                "tax_price": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["tax", "ppn", "pb1"],
                },
                "discount_price": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["discount", "diskon"],
                },
                "service_price": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["service", "service charge"],
                },
            },
        },
        "total": {
            "type": "object",
            "properties": {
                "total_price": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["total", "grand total", "amount due"],
                },
                "cashprice": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["cash", "tunai"],
                },
                "changeprice": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["change", "kembali", "kembalian"],
                },
                "creditcardprice": {
                    "type": "number",
                    "x-numeric-tol": 0.01,
                    "x-aliases": ["credit card", "debit", "card"],
                },
                "menuqty_cnt": {"type": "string", "x-aliases": ["qty", "items", "item qty"]},
            },
        },
    },
}


def document_from_valid_lines(
    doc_id: str, valid_lines: list[dict[str, Any]], width: int, height: int
) -> Document:
    """Build a real text-layer Document from CORD ``valid_line`` word quads."""
    words: list[Word] = []
    lines: list[str] = []
    for line in valid_lines:
        line_words = line.get("words") or []
        texts = []
        for w in line_words:
            quad = w["quad"]
            xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
            ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
            bbox = (
                max(0.0, min(xs) / width),
                max(0.0, min(ys) / height),
                min(1.0, max(xs) / width),
                min(1.0, max(ys) / height),
            )
            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                words.append(Word(text=w["text"], bbox=bbox))
                texts.append(w["text"])
        if texts:
            lines.append(" ".join(texts))
    page = Page(
        page=0, width=float(width), height=float(height), text="\n".join(lines), words=words
    )
    return Document(doc_id=doc_id, pages=[page])


def golds_from_gt_parse(gt_parse: dict[str, Any], schema: Schema, doc: Document) -> list[FieldGold]:
    """Flatten gt_parse into scored gold fields; locate gold boxes on the page."""
    golds = []
    for path, value in flatten_json(gt_parse).items():
        if value is None or str(value).strip() == "":
            continue
        leaf = schema.leaf(path)
        golds.append(
            FieldGold(
                path=path,
                value=value,
                scoring=leaf.scoring if leaf else "exact",
                numeric_tol=leaf.numeric_tol if leaf else 1e-6,
                gold_box=_locate(str(value), doc, min_support=0.7, penalty_mode="none"),
            )
        )
    return golds


def load(
    split: str = "validation",
    limit: int | None = 100,
    cache_dir: str | Path = "data",
    with_images: bool = False,
) -> list[BenchDocument]:  # pragma: no cover - network on first call
    """Load CORD v2 (streamed; JSON-cached locally after the first download).

    ``with_images=True`` additionally exports each receipt PNG (needed by
    image-reading OCR adapters) and sets ``page.image_path``.
    """
    schema = Schema.from_json_schema(CORD_SCHEMA_RAW, name="cord-receipt")
    cache = Path(cache_dir) / f"cord_{split}_{limit or 'all'}.json"
    image_dir = Path(cache_dir) / "cord_images" / split
    need_images = with_images and not image_dir.exists()
    if cache.exists() and not need_images:
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("CORD loader requires: pip install 'verifydoc[data]'") from exc
        stream = load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True)
        rows = []
        if with_images:
            image_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(stream):
            if limit is not None and i >= limit:
                break
            gt = json.loads(row["ground_truth"])
            width, height = row["image"].size
            if with_images:
                row["image"].save(image_dir / f"{i:05d}.png")
            rows.append(
                {
                    "gt_parse": gt["gt_parse"],
                    "valid_line": gt["valid_line"],
                    "width": width,
                    "height": height,
                }
            )
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")

    out = []
    for i, row in enumerate(rows):
        doc = document_from_valid_lines(
            f"cord-{split}-{i:05d}", row["valid_line"], row["width"], row["height"]
        )
        image_path = image_dir / f"{i:05d}.png"
        if with_images and image_path.exists():
            doc.pages[0].image_path = str(image_path)
        golds = golds_from_gt_parse(row["gt_parse"], schema, doc)
        if golds and doc.pages[0].words:
            out.append(BenchDocument(doc=doc, schema=schema, golds=golds))
    return out
