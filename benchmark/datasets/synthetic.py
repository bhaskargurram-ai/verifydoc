"""Synthetic invoice corpus: seeded, offline, with gold values AND gold boxes.

This is the CI-runnable slice of VerifyDocBench: every document is generated
from a template, so per-field correctness is computable automatically and the
gold source box is recovered from the synthetic text layout. Public-dataset
loaders (CORD et al.) extend this interface.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from verifydoc.grounding.attach import _locate
from verifydoc.ingest import document_from_text
from verifydoc.types import Document, FieldGold, Schema

VENDORS = [
    "ACME Corporation",
    "Globex Industries",
    "Initech Solutions",
    "Umbrella Supply Co",
    "Stark Manufacturing",
    "Wayne Logistics",
]

INVOICE_SCHEMA_RAW: dict = {
    "type": "object",
    "required": ["invoice_id", "vendor", "date", "subtotal", "tax", "total"],
    "properties": {
        "invoice_id": {"type": "string"},
        "vendor": {"type": "string", "x-scoring": "semantic"},
        "date": {"type": "string"},
        "subtotal": {"type": "number", "x-numeric-tol": 0.01},
        "tax": {"type": "number", "x-numeric-tol": 0.01},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


@dataclass
class BenchDocument:
    """One benchmark item: the document, its schema, and gold fields with boxes."""

    doc: Document
    schema: Schema
    golds: list[FieldGold]


def generate(n_docs: int = 40, seed: int = 0) -> list[BenchDocument]:
    """Deterministically generate ``n_docs`` synthetic invoices."""
    rng = random.Random(seed)
    schema = Schema.from_json_schema(INVOICE_SCHEMA_RAW, name="synthetic-invoice")
    out = []
    for i in range(n_docs):
        vendor = rng.choice(VENDORS)
        invoice_id = f"INV-{2024 + i % 2}-{rng.randrange(10_000):04d}"
        date = f"{2024 + i % 2}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
        subtotal = round(rng.uniform(50, 9000), 2)
        tax = round(subtotal * 0.0825, 2)
        total = round(subtotal + tax, 2)
        text = (
            f"{vendor.upper()}\n"
            f"Invoice ID: {invoice_id}\n"
            f"Vendor: {vendor}\n"
            f"Date: {date}\n"
            f"Subtotal: {subtotal:.2f}\n"
            f"Tax: {tax:.2f}\n"
            f"Total: {total:.2f}\n"
            f"Payment due within 30 days"
        )
        doc = document_from_text(f"synth-{seed}-{i:04d}", [text])
        values: dict[str, object] = {
            "invoice_id": invoice_id,
            "vendor": vendor,
            "date": date,
            "subtotal": f"{subtotal:.2f}",
            "tax": f"{tax:.2f}",
            "total": f"{total:.2f}",
        }
        golds = []
        for leaf in schema.leaves:
            value = values[leaf.path]
            gold_box = _locate(str(value), doc, min_support=0.99, penalize_ambiguity=False)
            golds.append(
                FieldGold(
                    path=leaf.path,
                    value=value,
                    scoring=leaf.scoring,
                    numeric_tol=leaf.numeric_tol,
                    gold_box=gold_box,
                )
            )
        out.append(BenchDocument(doc=doc, schema=schema, golds=golds))
    return out
