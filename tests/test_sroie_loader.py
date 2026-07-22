"""Offline tests for SROIE loader helpers (fixture rows, no network, rule #5)."""

import pytest

from benchmark.datasets.sroie import (
    SROIE_SCHEMA_RAW,
    document_from_lines,
    golds_from_entities,
)
from verifydoc.types import Schema

W, H = 800, 1000
LINES = [
    {"text": "SUPERSTORE SDN BHD", "box": [100, 40, 500, 70]},
    {"text": "NO 12 JALAN MAJU", "box": [100, 80, 480, 110]},
    {"text": "DATE: 01/03/2024", "box": [100, 200, 380, 230]},
    {"text": "TOTAL 45.50", "box": [100, 560, 320, 590]},
]
ENTITIES = {
    "company": "SUPERSTORE SDN BHD",
    "date": "01/03/2024",
    "address": "NO 12 JALAN MAJU",
    "total": "45.50",
}


class TestDocumentFromLines:
    def test_text_layer_and_boxes(self):
        doc = document_from_lines("r1", LINES, W, H)
        assert "SUPERSTORE SDN BHD" in (doc.pages[0].text or "")
        assert len(doc.pages[0].words) == 4
        w = doc.pages[0].words[0]
        assert 0 <= w.bbox[0] < w.bbox[2] <= 1

    def test_empty_lines_dropped(self):
        doc = document_from_lines("r2", [{"text": "", "box": [1, 2, 3, 4]}], W, H)
        assert doc.pages[0].words == []


class TestGoldsFromEntities:
    def test_scored_golds_with_boxes(self):
        schema = Schema.from_json_schema(SROIE_SCHEMA_RAW, name="sroie")
        doc = document_from_lines("r1", LINES, W, H)
        golds = {g.path: g for g in golds_from_entities(ENTITIES, schema, doc)}
        assert set(golds) == {"company", "date", "address", "total"}
        assert golds["total"].scoring == "numeric"
        assert golds["company"].scoring == "semantic"
        # company appears verbatim -> located
        assert golds["company"].gold_box is not None
        assert golds["company"].gold_box.support == pytest.approx(1.0)

    def test_total_numeric_scoring(self):
        from verifydoc.eval.extraction import value_correct

        schema = Schema.from_json_schema(SROIE_SCHEMA_RAW)
        doc = document_from_lines("r1", LINES, W, H)
        total = next(g for g in golds_from_entities(ENTITIES, schema, doc) if g.path == "total")
        assert value_correct("45.50", total)
        assert value_correct("$45.50", total)  # currency symbol stripped
        assert value_correct("45.505", total)  # within 0.01 tolerance
        assert not value_correct("45.60", total)

    def test_missing_entities_skipped(self):
        schema = Schema.from_json_schema(SROIE_SCHEMA_RAW)
        doc = document_from_lines("r1", LINES, W, H)
        golds = golds_from_entities({"company": "X", "total": ""}, schema, doc)
        assert {g.path for g in golds} == {"company"}
