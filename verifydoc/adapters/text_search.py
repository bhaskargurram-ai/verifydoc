"""Dependency-free heuristic extractor: label/value search over the text layer.

The honest floor baseline: for each schema leaf, find a line whose text
contains the leaf's label ("invoice_id" -> "invoice id") and take what follows
the separator as the value. Real OCR/VLM adapters reuse this field-finding on
top of their own text layers; the paper reports it as the no-model baseline.

# DECISION: the heuristic baseline skips array leaves (``items[].price``) —
# repeating-group detection is extractor territory, not a text heuristic's;
# array handling is exercised through the mock and model adapters.
"""

from __future__ import annotations

import re

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.types import Document, FieldPrediction, Schema


class TextSearchAdapter(ExtractorAdapter):
    name = "text-search"

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        preds: list[FieldPrediction] = []
        for leaf in schema.leaves:
            if "[]" in leaf.path:
                continue
            label = leaf.path.split(".")[-1].replace("_", " ").casefold()
            found = self._find(doc, label)
            if found is not None:
                value, page_no, line = found
                preds.append(
                    FieldPrediction(
                        path=leaf.path,
                        value=value,
                        confidence=0.5,
                        meta={"source_page": page_no, "source_line": line},
                    )
                )
        return preds

    @staticmethod
    def _find(doc: Document, label: str) -> tuple[str, int, str] | None:
        # word-boundary match so "total" never fires inside "Subtotal"
        pattern = re.compile(rf"\b{re.escape(label)}\b")
        for page in doc.pages:
            if not page.text:
                continue
            for line in page.text.split("\n"):
                match = pattern.search(line.casefold())
                if match is None:
                    continue
                value = line[match.end() :].lstrip(" \t:—–-").strip()
                if value:
                    return value, page.page, line
        return None
