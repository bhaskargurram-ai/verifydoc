"""Canned adapter: replay already-extracted field values (no model call).

Used by the framework integrations (Instructor/LangChain), which have an
extraction in hand and only need VerifyDoc to score it for grounding /
confidence / abstention.
"""

from __future__ import annotations

from typing import Any

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema


class CannedAdapter(ExtractorAdapter):
    """Emit predictions from a pre-computed ``{leaf_path: value}`` mapping."""

    name = "canned"

    def __init__(self, flat_values: dict[str, Any]) -> None:
        self._flat = flat_values

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        return [
            FieldPrediction(path=path, value=value)
            for path, value in self._flat.items()
            if value is not None
        ]
