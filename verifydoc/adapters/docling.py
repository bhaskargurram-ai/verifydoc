"""Docling/MinerU output adapter: reuse a parse you already ran.

Consumes either a live ``docling`` conversion (SDK installed) or an exported
markdown/text file from Docling/MinerU/Marker, then runs the shared
field-finding heuristic over the parsed text.
"""

from __future__ import annotations

from pathlib import Path

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.adapters.text_search import TextSearchAdapter
from verifydoc.ingest import document_from_text
from verifydoc.types import Document, FieldPrediction, Schema


class DoclingAdapter(ExtractorAdapter):
    name = "docling"

    def __init__(self, parsed_output: str | Path | None = None) -> None:
        """``parsed_output``: path to an exported markdown/text parse. When
        omitted, the docling SDK is used to convert the source directly."""
        self._parsed_output = Path(parsed_output) if parsed_output is not None else None

    def extract(self, doc: Document, schema: Schema) -> list[FieldPrediction]:
        if self._parsed_output is not None:
            text = self._parsed_output.read_text(encoding="utf-8")
        else:  # pragma: no cover - needs docling SDK
            text = self._convert_with_sdk(doc)
        parsed_doc = document_from_text(doc.doc_id, text.split("\f"))
        return TextSearchAdapter().extract(parsed_doc, schema)

    @staticmethod
    def _convert_with_sdk(doc: Document) -> str:  # pragma: no cover - needs docling SDK
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError(
                "DoclingAdapter without parsed_output requires docling: pip install docling"
            ) from exc
        if doc.source_path is None:
            raise ValueError("document has no source_path for docling conversion")
        result = DocumentConverter().convert(doc.source_path)
        return str(result.document.export_to_markdown())
