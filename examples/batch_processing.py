"""Batch-verify many documents against one schema with verify_batch().

Runs offline with the local text-search adapter on a few inline receipts; swap
in real files (``ingest_path`` / a folder glob) and any adapter for production.

    python examples/batch_processing.py
"""

from __future__ import annotations

from typing import Any

from verifydoc import verify_batch
from verifydoc.ingest import document_from_text

RECEIPTS = {
    "r1": "Corner Cafe\nDate: 2024-05-01\nTotal: 7.70\n",
    "r2": "Bloom Florist\nDate: 2024-05-02\nTotal: 42.00\n",
    "r3": "Metro Garage\nDate: 2024-05-03\nTotal: 18.50\n",
}

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


def main() -> None:
    # In production: docs = [ingest_path(p) for p in Path("invoices").glob("*.pdf")]
    docs = [document_from_text(name, [text]) for name, text in RECEIPTS.items()]
    results = verify_batch(docs, schema=SCHEMA, threshold=0.8)

    total_accept = total_review = 0
    for r in results:
        total_accept += r.n_accepted
        total_review += r.n_review
        print(f"{r.doc_id}: {r.n_accepted} accepted, {r.n_review} to review")
    print(f"\nBatch: {len(results)} docs, {total_accept} auto-accepted, {total_review} to review.")
    # Straight-through-processing rate = auto-accepted / total fields.


if __name__ == "__main__":
    main()
