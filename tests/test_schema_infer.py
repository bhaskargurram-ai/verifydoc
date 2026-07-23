"""Tests for schema inference (offline, heuristic proposer)."""

import pytest

from verifydoc.agents import HeuristicSchemaProposer, SchemaProposer, infer_schema, verify_auto
from verifydoc.ingest import document_from_text

INVOICE = "\n".join(
    [
        "Invoice #: INV-2024-0912",
        "Date: 2024-03-04",
        "Subtotal: 1,100.00",
        "Tax: 134.50",
        "Total: $1,234.50",
    ]
)


class TestHeuristicProposer:
    def test_infers_types_and_scoring(self):
        props = infer_schema(INVOICE)["properties"]
        assert set(props) == {"invoice", "date", "subtotal", "tax", "total"}
        # id-like label → exact string
        assert props["invoice"] == {"type": "string", "x-scoring": "exact"}
        # numeric values → number with a tolerance
        assert props["total"]["type"] == "number" and "x-numeric-tol" in props["total"]
        assert props["subtotal"]["type"] == "number"
        # a date value → exact string
        assert props["date"] == {"type": "string", "x-scoring": "exact"}

    def test_numeric_id_label_stays_exact_not_number(self):
        # an id-like label wins even when the value is numeric
        props = infer_schema("Invoice Number: 12345")["properties"]
        assert props["invoice_number"] == {"type": "string", "x-scoring": "exact"}

    def test_free_text_is_semantic(self):
        props = infer_schema("Vendor: ACME Supplies Ltd")["properties"]
        assert props["vendor"] == {"type": "string", "x-scoring": "semantic"}

    def test_unlabelled_lines_are_skipped(self):
        props = infer_schema("ACME SUPPLIES INVOICE\nTotal: 10")["properties"]
        assert list(props) == ["total"]  # the banner line has no "label:" and is skipped

    def test_first_label_wins_on_duplicates(self):
        props = infer_schema("Total: 10\nTotal: 20")["properties"]
        assert list(props) == ["total"]

    def test_default_proposer_satisfies_protocol(self):
        assert isinstance(HeuristicSchemaProposer(), SchemaProposer)


class TestVerifyAuto:
    def test_infers_then_verifies(self):
        doc = document_from_text("invoice", [INVOICE])
        schema, result = verify_auto(doc, threshold=0.8)
        assert schema["properties"]  # a schema was proposed
        paths = {f.path for f in result.fields}
        assert "total" in paths and "date" in paths
        # the grounded numeric/date fields should verify (present in the text)
        by = {f.path: f for f in result.fields}
        assert by["total"].decision == "accept"

    def test_custom_proposer_is_used(self):
        class FixedProposer:
            def propose(self, text):
                return {"type": "object", "properties": {"total": {"type": "number"}}}

        doc = document_from_text("d", ["Total: 5"])
        schema, result = verify_auto(doc, proposer=FixedProposer())
        assert list(schema["properties"]) == ["total"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
