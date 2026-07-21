"""Tests for verifydoc.types: contracts, schema walking, path utilities."""

import pytest
from pydantic import ValidationError

from verifydoc.types import (
    Document,
    FieldGold,
    FieldPrediction,
    Grounding,
    Page,
    Schema,
    flatten_json,
    schema_path,
    unflatten_json,
)


class TestGrounding:
    def test_valid(self):
        g = Grounding(page=0, bbox=(0.1, 0.2, 0.4, 0.3), char_span=(5, 12), support=0.9)
        assert g.support == 0.9

    def test_bad_bbox_rejected(self):
        with pytest.raises(ValidationError):
            Grounding(page=0, bbox=(0.5, 0.2, 0.4, 0.3))

    def test_bad_span_rejected(self):
        with pytest.raises(ValidationError):
            Grounding(page=0, char_span=(12, 5))

    def test_negative_page_rejected(self):
        with pytest.raises(ValidationError):
            Grounding(page=-1)


class TestFieldPrediction:
    def test_defaults(self):
        p = FieldPrediction(path="total")
        assert p.decision == "review"
        assert 0.0 <= p.confidence <= 1.0

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            FieldPrediction(path="total", confidence=1.5)
        with pytest.raises(ValidationError):
            FieldPrediction(path="total", confidence=-0.1)

    def test_bad_decision_rejected(self):
        with pytest.raises(ValidationError):
            FieldPrediction(path="total", decision="maybe")


class TestFieldGold:
    def test_scoring_rules(self):
        g = FieldGold(path="total", value=42.5, scoring="numeric", numeric_tol=0.01)
        assert g.scoring == "numeric"
        with pytest.raises(ValidationError):
            FieldGold(path="total", scoring="fuzzy")


class TestDocument:
    def test_pages(self):
        doc = Document(
            doc_id="d1",
            pages=[Page(page=0, width=612, height=792, text="hello")],
        )
        assert doc.n_pages == 1

    def test_bad_page_geometry(self):
        with pytest.raises(ValidationError):
            Page(page=0, width=0, height=10)


class TestSchema:
    RAW = {
        "type": "object",
        "required": ["invoice_id", "total"],
        "properties": {
            "invoice_id": {"type": "string"},
            "vendor": {"type": "string", "x-scoring": "semantic"},
            "total": {"type": "number", "x-numeric-tol": 0.01},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
            },
        },
    }

    def test_leaf_extraction(self):
        schema = Schema.from_json_schema(self.RAW, name="invoice")
        paths = {leaf.path for leaf in schema.leaves}
        assert paths == {"invoice_id", "vendor", "total", "items[].name", "items[].price"}

    def test_scoring_annotations(self):
        schema = Schema.from_json_schema(self.RAW)
        by_path = {leaf.path: leaf for leaf in schema.leaves}
        assert by_path["invoice_id"].scoring == "exact"
        assert by_path["vendor"].scoring == "semantic"
        assert by_path["total"].scoring == "numeric"
        assert by_path["total"].numeric_tol == 0.01
        assert by_path["items[].price"].scoring == "numeric"

    def test_required_propagation(self):
        schema = Schema.from_json_schema(self.RAW)
        by_path = {leaf.path: leaf for leaf in schema.leaves}
        assert by_path["invoice_id"].required
        assert not by_path["vendor"].required
        assert by_path["items[].name"].required
        assert not by_path["items[].price"].required

    def test_concrete_path_lookup(self):
        schema = Schema.from_json_schema(self.RAW)
        leaf = schema.leaf("items[3].price")
        assert leaf is not None and leaf.path == "items[].price"
        assert schema.leaf("nonexistent") is None


class TestPathUtils:
    def test_schema_path(self):
        assert schema_path("items[12].price") == "items[].price"
        assert schema_path("total") == "total"

    def test_flatten(self):
        obj = {"a": 1, "b": {"c": "x"}, "items": [{"n": "p"}, {"n": "q"}]}
        flat = flatten_json(obj)
        assert flat == {"a": 1, "b.c": "x", "items[0].n": "p", "items[1].n": "q"}

    def test_roundtrip(self):
        obj = {"a": 1, "b": {"c": "x"}, "items": [{"n": "p", "v": 2.5}, {"n": "q", "v": 3.0}]}
        assert unflatten_json(flatten_json(obj)) == obj

    def test_unflatten_sparse_array(self):
        assert unflatten_json({"xs[1]": "b"}) == {"xs": [None, "b"]}
