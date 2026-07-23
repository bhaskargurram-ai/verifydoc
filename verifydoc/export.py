"""Export verified results for downstream systems (issue #30).

Flattens one or many :class:`VerifiedResult` into per-field rows carrying the
trust contract (``confidence``, ``decision``, ``grounded``, ``page``) so a
pipeline can, e.g., load only auto-accepted fields into a database and route the
rest to a review table. Pure/stdlib — no pandas dependency.

    from verifydoc import verify_batch
    from verifydoc.export import to_csv, to_records

    results = verify_batch(docs, schema=SCHEMA)
    to_csv(results, "verified.csv")
    accepted = [r for r in to_records(results) if r["decision"] == "accept"]
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from verifydoc.pipeline import VerifiedResult

FIELDNAMES = ["doc_id", "field", "value", "confidence", "decision", "grounded", "page"]


def to_records(results: Sequence[VerifiedResult]) -> list[dict[str, Any]]:
    """One flat row per field across all results (stable column order)."""
    rows: list[dict[str, Any]] = []
    for result in results:
        for field in result.fields:
            rows.append(
                {
                    "doc_id": result.doc_id,
                    "field": field.path,
                    "value": field.value,
                    "confidence": round(field.confidence, 4),
                    "decision": field.decision,
                    "grounded": field.grounding is not None,
                    "page": field.grounding.page if field.grounding else None,
                }
            )
    return rows


def to_csv(results: Sequence[VerifiedResult], path: str | Path) -> Path:
    """Write per-field rows to a CSV file; returns the path."""
    dest = Path(path)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in to_records(results):
            writer.writerow(row)
    return dest


def to_jsonl(results: Sequence[VerifiedResult], path: str | Path) -> Path:
    """Write one JSON object per document (``VerifiedResult.to_dict``) per line."""
    dest = Path(path)
    with dest.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    return dest
