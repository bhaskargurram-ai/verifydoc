"""FUNSD loader (199 scanned forms, noisy real-world KIE) -> VerifyDocBench.

Question->answer links become gold fields: the path is a slug of the question
text, the value is the linked answer text, and the gold box is the exact
union of the answer's word boxes (no locating needed — FUNSD annotates them).
Download (~16 MB zip) happens once into ``data/``; unit tests use fixture
annotations only (golden rule #5).

License note (benchmark/card.md): FUNSD permits research use; we ship this
loader + our added splits, never the images.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmark.datasets.synthetic import BenchDocument
from verifydoc.eval.extraction import parse_number
from verifydoc.types import Document, FieldGold, Grounding, Page, Schema, SchemaLeaf, Word

FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(question: str) -> str:
    """Question text -> a stable field path ('DATE:' -> 'date')."""
    slug = _SLUG_RE.sub("_", question.casefold()).strip("_")
    return slug or "field"


def _norm_box(box: list[int], width: int, height: int) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = box
    bbox = (
        max(0.0, x0 / width),
        max(0.0, y0 / height),
        min(1.0, x1 / width),
        min(1.0, y1 / height),
    )
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def bench_from_annotation(
    doc_id: str, annotation: dict[str, Any], width: int, height: int
) -> BenchDocument | None:
    """Build one benchmark item from a FUNSD annotation dict."""
    entries = {e["id"]: e for e in annotation.get("form", [])}

    words: list[Word] = []
    lines: list[str] = []
    for entry in annotation.get("form", []):
        texts = []
        for w in entry.get("words", []):
            bbox = _norm_box(w["box"], width, height)
            if bbox is not None and w["text"].strip():
                words.append(Word(text=w["text"], bbox=bbox))
                texts.append(w["text"])
        if texts:
            lines.append(" ".join(texts))
    doc = Document(
        doc_id=doc_id,
        pages=[
            Page(
                page=0, width=float(width), height=float(height), text="\n".join(lines), words=words
            )
        ],
    )

    # question -> linked answers, in annotation order
    answers_by_question: dict[int, list[int]] = {}
    for entry in annotation.get("form", []):
        for a, b in entry.get("linking", []):
            qa = (a, b)
            if (
                entries.get(qa[0], {}).get("label") == "question"
                and entries.get(qa[1], {}).get("label") == "answer"
            ):
                answers_by_question.setdefault(qa[0], [])
                if qa[1] not in answers_by_question[qa[0]]:
                    answers_by_question[qa[0]].append(qa[1])

    golds: list[FieldGold] = []
    leaves: list[SchemaLeaf] = []
    seen_paths: set[str] = set()
    for q_id, a_ids in answers_by_question.items():
        question = entries[q_id]["text"].strip()
        # DECISION: multiple linked answers concatenate in annotation order.
        value = " ".join(entries[a]["text"].strip() for a in a_ids if entries[a]["text"].strip())
        if not question or not value:
            continue
        path = slugify(question)
        n = 2
        while path in seen_paths:
            path = f"{slugify(question)}_{n}"
            n += 1
        seen_paths.add(path)

        answer_boxes = [
            _norm_box(w["box"], width, height) for a in a_ids for w in entries[a].get("words", [])
        ]
        answer_boxes = [b for b in answer_boxes if b is not None]
        gold_box = (
            Grounding(
                page=0,
                bbox=(
                    min(b[0] for b in answer_boxes),
                    min(b[1] for b in answer_boxes),
                    max(b[2] for b in answer_boxes),
                    max(b[3] for b in answer_boxes),
                ),
                support=1.0,
            )
            if answer_boxes
            else None
        )
        scoring = "numeric" if parse_number(value) is not None else "semantic"
        golds.append(
            FieldGold(path=path, value=value, scoring=scoring, numeric_tol=1e-6, gold_box=gold_box)
        )
        leaves.append(SchemaLeaf(path=path, type="string", scoring=scoring))

    if not golds or not words:
        return None
    schema = Schema(name=f"funsd-{doc_id}", leaves=leaves)
    return BenchDocument(doc=doc, schema=schema, golds=golds)


def load(
    split: str = "testing", limit: int | None = None, cache_dir: str | Path = "data"
) -> list[BenchDocument]:  # pragma: no cover - network on first call
    """Load FUNSD (downloads + unzips once into ``cache_dir``)."""
    from PIL import Image

    root = Path(cache_dir) / "funsd"
    if not root.exists():
        import io
        import urllib.request
        import zipfile

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FUNSD_URL) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(root)

    ann_dir = root / "dataset" / f"{split}_data" / "annotations"
    img_dir = root / "dataset" / f"{split}_data" / "images"
    out: list[BenchDocument] = []
    for i, ann_path in enumerate(sorted(ann_dir.glob("*.json"))):
        if limit is not None and i >= limit:
            break
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        img_path = img_dir / f"{ann_path.stem}.png"
        with Image.open(img_path) as img:
            width, height = img.size
        item = bench_from_annotation(f"funsd-{split}-{ann_path.stem}", annotation, width, height)
        if item is not None:
            item.doc.pages[0].image_path = str(img_path)  # for image-reading OCR adapters
            out.append(item)
    return out
