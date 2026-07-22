"""Add VerifyDoc's trust layer to an Instructor / Pydantic extraction.

Instructor (and Pydantic-AI, Outlines, Marvin) guarantee a value is well-typed;
they do not tell you whether it is *correct* or *where it came from*. This
wraps their output: given the document text and the extracted Pydantic model,
it returns a per-field trust report (calibrated confidence + grounding +
accept/review) alongside the original object.

Usage::

    import instructor
    from verifydoc.integrations.instructor import verify_instructor_result

    obj = client.chat.completions.create(response_model=Invoice, ...)  # Instructor
    report = verify_instructor_result(document_text, obj)
    for f in report.fields:
        if f.decision == "review":
            escalate(f.path, f.value, f.grounding)

No Instructor dependency is imported here — any ``pydantic.BaseModel`` instance
works, so this also covers Pydantic-AI / Outlines / Marvin outputs.
"""

from __future__ import annotations

from typing import Any

from verifydoc.pipeline import DEFAULT_THRESHOLD, VerifiedResult, verify
from verifydoc.types import Schema, flatten_json


def verify_instructor_result(
    document: str,
    result: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    **verify_kwargs: Any,
) -> VerifiedResult:
    """Verify an already-extracted Pydantic object against its source document.

    ``result`` is a ``pydantic.BaseModel`` instance (from Instructor et al.).
    The model's field values are treated as a canned extraction and scored for
    grounding/confidence/abstention against ``document`` — no re-extraction, no
    extra model call. Returns the standard :class:`VerifiedResult`.
    """
    if not hasattr(result, "model_dump"):
        raise TypeError("result must be a pydantic BaseModel instance")
    from verifydoc.ingest import document_from_text

    schema = Schema.from_pydantic(type(result))
    flat = flatten_json(result.model_dump())
    doc = document_from_text("extraction", [document])
    return verify(doc, schema, adapter=_CannedAdapter(flat), threshold=threshold, **verify_kwargs)


class _CannedAdapter:
    """Adapter that replays already-extracted field values (no model call)."""

    name = "instructor-canned"

    def __init__(self, flat_values: dict[str, Any]) -> None:
        self._flat = flat_values

    def extract(self, doc: Any, schema: Schema) -> list[Any]:
        from verifydoc.types import FieldPrediction

        return [
            FieldPrediction(path=path, value=value)
            for path, value in self._flat.items()
            if value is not None
        ]

    def extract_samples(self, doc: Any, schema: Schema, k: int = 1) -> list[list[Any]]:
        return [self.extract(doc, schema)]
