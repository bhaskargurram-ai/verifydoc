"""Add a VerifyDoc trust layer around any LlamaIndex / DSPy / Haystack extractor.

These frameworks give you a ``str -> dict`` extraction step. ``VerifiedExtractor``
wraps any such callable and returns the values *plus* per-field confidence,
grounding, and accept/review — without VerifyDoc depending on the framework.

Runs with nothing extra installed: a stub extractor stands in for your
LlamaIndex ``LLMTextCompletionProgram`` / DSPy module / Haystack pipeline.

    python examples/llamaindex_example.py
"""

from __future__ import annotations

from typing import Any

from verifydoc.integrations.langchain import VerifiedExtractor

DOCUMENT = """RECEIPT - Corner Cafe
Date: 2024-05-01
Item: Flat White   4.50
Item: Croissant     3.20
Total: 7.70
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "date": {"type": "string"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


def my_llamaindex_program(text: str) -> dict[str, Any]:
    """Stand-in for a LlamaIndex program / DSPy module / Haystack pipeline that
    maps document text to a dict of extracted fields."""
    return {"date": "2024-05-01", "total": 7.70}


def main() -> None:
    # VerifiedExtractor wraps any `str -> dict` (or `-> BaseModel`) extractor.
    extractor = VerifiedExtractor(my_llamaindex_program, schema=SCHEMA, threshold=0.8)
    result = extractor(DOCUMENT)
    for f in result.fields:
        flag = "OK  " if f.decision == "accept" else "REVIEW"
        where = f" @page {f.grounding.page}" if f.grounding else ""
        print(f"[{flag}] {f.path:8} = {f.value!r:14} conf={f.confidence:.2f}{where}")
    print(f"\n{result.n_accepted} accepted, {result.n_review} to review")


if __name__ == "__main__":
    main()
