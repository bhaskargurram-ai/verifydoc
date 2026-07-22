"""Add VerifyDoc's trust layer to a LangChain extraction chain.

LangChain's ``with_structured_output`` / extraction chains return a dict or
Pydantic object with no per-field trust signal. ``VerifiedExtractor`` wraps any
callable that maps ``document_text -> dict`` (an extraction chain, a runnable,
your own function) and returns the extracted values *plus* a VerifyDoc trust
report — so a chain can branch on ``accept``/``review`` instead of trusting
blindly.

Usage::

    from verifydoc.integrations.langchain import VerifiedExtractor

    extractor = VerifiedExtractor(my_chain.invoke, schema=InvoiceSchemaOrModel)
    result = extractor(document_text)
    result.n_review          # fields to escalate
    result.to_dict()         # nested value + confidence + grounding + decision

No langchain dependency is imported; any ``callable(str) -> dict`` works.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from verifydoc.pipeline import DEFAULT_THRESHOLD, VerifiedResult, verify
from verifydoc.types import Schema, flatten_json


class VerifiedExtractor:
    """Wrap a ``document -> dict`` extractor with VerifyDoc's trust layer."""

    def __init__(
        self,
        extract_fn: Callable[[str], Any],
        schema: Schema | dict[str, Any] | type,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        **verify_kwargs: Any,
    ) -> None:
        self._fn = extract_fn
        self._schema = _coerce_schema(schema)
        self._threshold = threshold
        self._verify_kwargs = verify_kwargs

    def __call__(self, document: str) -> VerifiedResult:
        from verifydoc.ingest import document_from_text

        raw = self._fn(document)
        flat = flatten_json(raw.model_dump() if hasattr(raw, "model_dump") else raw)
        return verify(
            document_from_text("extraction", [document]),
            self._schema,
            adapter=_CannedAdapter(flat),
            threshold=self._threshold,
            **self._verify_kwargs,
        )


def _coerce_schema(schema: Schema | dict[str, Any] | type) -> Schema:
    if isinstance(schema, Schema):
        return schema
    if isinstance(schema, dict):
        return Schema.from_json_schema(schema)
    return Schema.from_pydantic(schema)  # a Pydantic model class


class _CannedAdapter:
    name = "langchain-canned"

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
