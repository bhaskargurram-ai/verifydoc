"""Offline tests for the DocILE loader helpers (fixture rows, no network)."""

import pytest

from benchmark.datasets.docile import (
    DOCILE_SCHEMA_RAW,
    document_from_words,
    golds_from_annotations,
)
from verifydoc.types import Schema

WIDTH, HEIGHT = 612.0, 792.0

WORDS = [
    {"text": "INVOICE", "bbox": [0.1, 0.05, 0.3, 0.08]},
    {"text": "#INV-2024-0042", "bbox": [0.1, 0.10, 0.4, 0.13]},
    {"text": "Date:", "bbox": [0.1, 0.15, 0.2, 0.18]},
    {"text": "2024-03-15", "bbox": [0.22, 0.15, 0.4, 0.18]},
    {"text": "ACME", "bbox": [0.1, 0.25, 0.3, 0.28]},
    {"text": "GmbH", "bbox": [0.32, 0.25, 0.45, 0.28]},
    {"text": "Total:", "bbox": [0.1, 0.80, 0.2, 0.83]},
    {"text": "1,234.56", "bbox": [0.22, 0.80, 0.4, 0.83]},
]

ANNOTATIONS = [
    {
        "page": 0,
        "bbox": [0.1, 0.10, 0.4, 0.13],
        "fieldtype": "invoice_number",
        "text": "INV-2024-0042",
    },
    {"page": 0, "bbox": [0.22, 0.15, 0.4, 0.18], "fieldtype": "invoice_date", "text": "2024-03-15"},
    {"page": 0, "bbox": [0.1, 0.25, 0.45, 0.28], "fieldtype": "vendor", "text": "ACME GmbH"},
    {"page": 0, "bbox": [0.22, 0.80, 0.4, 0.83], "fieldtype": "amount_total", "text": "1,234.56"},
    # line-item field — should be skipped
    {
        "page": 0,
        "bbox": [0.5, 0.5, 0.6, 0.55],
        "fieldtype": "line_item_description",
        "text": "Widget",
        "line_item_id": 1,
    },
    # unknown field type — should be skipped
    {"page": 0, "bbox": [0.5, 0.6, 0.6, 0.65], "fieldtype": "unknown_field", "text": "???"},
]


class TestDocumentFromWords:
    def test_text_layer_and_boxes(self):
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        assert "INVOICE" in (doc.pages[0].text or "")
        assert len(doc.pages[0].words) == 8
        w = doc.pages[0].words[0]
        assert 0 <= w.bbox[0] < w.bbox[2] <= 1

    def test_empty_words_dropped(self):
        doc = document_from_words("d2", [{"text": "", "bbox": [0.1, 0.1, 0.2, 0.2]}], WIDTH, HEIGHT)
        assert doc.pages[0].words == []


class TestGoldsFromAnnotations:
    def test_kile_fields_with_boxes(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        golds = {g.path: g for g in golds_from_annotations(ANNOTATIONS, schema, doc)}
        assert set(golds) == {"invoice_number", "invoice_date", "vendor", "amount_total"}

    def test_amount_total_numeric_scoring(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        total = next(
            g for g in golds_from_annotations(ANNOTATIONS, schema, doc) if g.path == "amount_total"
        )
        assert total.scoring == "numeric"
        assert total.numeric_tol == pytest.approx(0.01)

    def test_vendor_semantic_scoring(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        vendor = next(
            g for g in golds_from_annotations(ANNOTATIONS, schema, doc) if g.path == "vendor"
        )
        assert vendor.scoring == "semantic"

    def test_line_item_fields_skipped(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        golds = golds_from_annotations(ANNOTATIONS, schema, doc)
        assert "line_item_description" not in {g.path for g in golds}

    def test_unknown_field_types_skipped(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        golds = golds_from_annotations(ANNOTATIONS, schema, doc)
        assert "unknown_field" not in {g.path for g in golds}

    def test_empty_text_skipped(self):
        schema = Schema.from_json_schema(DOCILE_SCHEMA_RAW, name="docile")
        doc = document_from_words("d1", WORDS, WIDTH, HEIGHT)
        golds = golds_from_annotations(
            [{"page": 0, "bbox": [0.1, 0.1, 0.2, 0.2], "fieldtype": "vendor", "text": ""}],
            schema,
            doc,
        )
        assert golds == []
