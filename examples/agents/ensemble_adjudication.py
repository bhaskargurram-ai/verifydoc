"""Multi-extractor ensemble: run several extractors and adjudicate per field.

`verifydoc.agents.ensemble_verify` runs each adapter through the pipeline, then a
judge fuses them per field: where extractors agree, confidence rises; where they
disagree, the best-grounded reading wins and the dissent is recorded. Genuine
splits stay `review`.

    python examples/agents/ensemble_adjudication.py
"""

from __future__ import annotations

from typing import Any

from verifydoc.adapters.canned import CannedAdapter
from verifydoc.agents import ensemble_verify
from verifydoc.ingest import document_from_text

DOCUMENT = document_from_text("invoice", ["Vendor: ACME Supplies\nTotal: 1,234.50"])
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string", "x-scoring": "semantic"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


def main() -> None:
    # Stand-ins for three different extractors (an OCR pipeline, a VLM, an API model).
    # Two read the on-page total correctly; one hallucinates it.
    extractors = [
        ("ocr", CannedAdapter({"vendor": "ACME Supplies", "total": "1,234.50"})),
        ("vlm", CannedAdapter({"vendor": "ACME Supplies", "total": "1,234.50"})),
        ("api", CannedAdapter({"vendor": "ACME Supplies", "total": "1,432.50"})),
    ]
    names = [n for n, _ in extractors]
    result = ensemble_verify(
        DOCUMENT, SCHEMA, [a for _, a in extractors], names=names, threshold=0.8
    )
    for f in result.fields:
        ens = f.meta.get("ensemble", {})
        print(
            f"{f.path} = {f.value!r}  {f.decision}  "
            f"(agreement {ens.get('agreement', 0):.2f}, dissent {ens.get('dissent', [])})"
        )


if __name__ == "__main__":
    main()
