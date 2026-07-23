"""Verify a Pydantic-AI (or any Pydantic) extraction with VerifyDoc.

Pydantic-AI agents return a validated ``BaseModel``. Validation proves the JSON
*parses*; it says nothing about whether the values are *correct*. VerifyDoc adds
the missing trust layer: per-field calibrated confidence + source grounding +
an accept/review decision.

Runs with no framework installed — it uses a plain BaseModel to stand in for a
Pydantic-AI result, so the pattern is copy-pasteable as-is.

    python examples/pydantic_ai_example.py
"""

from __future__ import annotations

from pydantic import BaseModel

from verifydoc.integrations.instructor import verify_instructor_result

DOCUMENT = """ACME SUPPLIES INVOICE
Invoice #: INV-2024-0912
Vendor: ACME Supplies Ltd
Date: 2024-03-04
Total: $1,234.50
"""


class Invoice(BaseModel):
    invoice_id: str
    vendor: str
    total: float


def main() -> None:
    # In real use this object comes from `agent.run_sync(...).data` (Pydantic-AI),
    # `client.chat.completions.create(response_model=Invoice, ...)` (Instructor),
    # Outlines, Marvin, etc. — anything that yields a BaseModel.
    extracted = Invoice(invoice_id="INV-2024-0912", vendor="ACME Supplies Ltd", total=1234.50)

    result = verify_instructor_result(DOCUMENT, extracted, threshold=0.8)
    for f in result.fields:
        flag = "OK  " if f.decision == "accept" else "REVIEW"
        where = f" @page {f.grounding.page}" if f.grounding else ""
        print(f"[{flag}] {f.path:12} = {f.value!r:22} conf={f.confidence:.2f}{where}")
    print(f"\n{result.n_accepted} accepted, {result.n_review} to review")


if __name__ == "__main__":
    main()
