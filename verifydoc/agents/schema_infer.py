"""Schema inference: propose a scored JSON schema from a raw document.

Writing the schema is the friction in "extract structured JSON from this doc."
This layer removes it: point at a document with **no schema** and get a proposed
one (field names + types + ``x-scoring`` rules), then run verified extraction —
``verify_auto`` returns both.

The proposer is pluggable behind :class:`SchemaProposer`. The zero-dependency
default (:class:`HeuristicSchemaProposer`) reads ``Label: value`` lines and infers
each leaf's type/scoring: an id-like label → ``exact`` string, a numeric value →
``number`` with a tolerance, a date → ``exact`` string, otherwise a ``semantic``
string. Plug an LLM proposer for messy, unlabelled layouts.

No model SDK is imported here; an LLM proposer is injected by the caller.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from verifydoc.pipeline import DEFAULT_THRESHOLD, VerifiedResult, verify
from verifydoc.types import Document

# "Label: value" — a label of letters/spaces/#/_/-/& then a colon then a value.
_LABEL_RE = re.compile(r"^\s*([A-Za-z][\w /&.#-]{0,40}?)\s*:\s*(\S.*?)\s*$")
# a purely-numeric amount (optional currency, thousands separators, decimals, %).
_NUMERIC_RE = re.compile(r"^[$€£]?\s?-?[\d,]+(?:\.\d+)?\s?%?$")
# common date shapes.
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$")
_ID_TOKENS = ("id", "no", "number", "ref", "invoice", "order", "#")


@runtime_checkable
class SchemaProposer(Protocol):
    """Propose a JSON Schema (with ``x-scoring``/``x-numeric-tol``) from text."""

    def propose(self, text: str) -> dict[str, Any]: ...


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _leaf_for(label: str, value: str) -> dict[str, Any]:
    label_l = label.lower()
    if any(tok in label_l for tok in _ID_TOKENS):
        return {"type": "string", "x-scoring": "exact"}
    if _NUMERIC_RE.match(value):
        return {"type": "number", "x-numeric-tol": 0.01}
    if _DATE_RE.match(value):
        return {"type": "string", "x-scoring": "exact"}
    return {"type": "string", "x-scoring": "semantic"}


class HeuristicSchemaProposer:
    """Zero-dependency proposer: infer leaves from ``Label: value`` lines."""

    def propose(self, text: str) -> dict[str, Any]:
        props: dict[str, Any] = {}
        for line in text.splitlines():
            m = _LABEL_RE.match(line)
            if not m:
                continue
            name = _slug(m.group(1))
            if not name or name in props:
                continue
            props[name] = _leaf_for(m.group(1), m.group(2))
        return {"type": "object", "properties": props}


def _text_of(source: str | Document) -> str:
    if isinstance(source, Document):
        return "\n".join(p.text for p in source.pages if p.text)
    return source


def infer_schema(source: str | Document, proposer: SchemaProposer | None = None) -> dict[str, Any]:
    """Propose a scored JSON schema for a document (text or :class:`Document`)."""
    proposer = proposer or HeuristicSchemaProposer()
    return proposer.propose(_text_of(source))


def verify_auto(
    source: Document,
    *,
    proposer: SchemaProposer | None = None,
    adapter: Any = None,
    threshold: float = DEFAULT_THRESHOLD,
    k: int = 1,
    calibrator: Any = None,
) -> tuple[dict[str, Any], VerifiedResult]:
    """Infer a schema from the document, then verify against it.

    Returns ``(inferred_schema, result)``. ``source`` must be an ingested
    :class:`Document` (use ``ingest_path`` / ``document_from_text``), so the same
    text drives both inference and grounding.
    """
    schema = infer_schema(source, proposer)
    result = verify(
        source, schema, adapter=adapter, k=k, threshold=threshold, calibrator=calibrator
    )
    return schema, result
