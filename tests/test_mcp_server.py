"""Tests for the MCP server's argument marshalling (offline, no MCP SDK needed)."""

import json

import pytest

from verifydoc.mcp_server import _run_verify

SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}
DOC = "Invoice ID: INV-9\nTotal: $12.50\n"


class TestRunVerify:
    def test_raw_text_and_dict_schema(self):
        out = _run_verify(DOC, SCHEMA, threshold=0.8)
        by_path = {f["path"]: f for f in out["fields"]}
        assert by_path["invoice_id"]["value"] == "INV-9"
        assert by_path["total"]["value"] == "$12.50"
        assert out["n_accepted"] + out["n_review"] == len(out["fields"])
        assert "review" in out["summary"]

    def test_schema_as_json_string(self):
        out = _run_verify(DOC, json.dumps(SCHEMA))
        assert {f["path"] for f in out["fields"]} == {"invoice_id", "total"}

    def test_file_paths(self, tmp_path):
        doc_f = tmp_path / "d.txt"
        doc_f.write_text(DOC, encoding="utf-8")
        schema_f = tmp_path / "s.json"
        schema_f.write_text(json.dumps(SCHEMA), encoding="utf-8")
        out = _run_verify(str(doc_f), str(schema_f))
        assert out["doc_id"] == "d"

    def test_fields_carry_the_contract(self):
        out = _run_verify(DOC, SCHEMA)
        for f in out["fields"]:
            assert set(f) >= {"path", "value", "confidence", "decision", "grounding"}
            assert f["decision"] in ("accept", "review")
            assert 0.0 <= f["confidence"] <= 1.0

    def test_output_is_json_serializable(self):
        json.dumps(_run_verify(DOC, SCHEMA))  # must not raise

    def test_bad_adapter_rejected(self):
        with pytest.raises(ValueError):
            _run_verify(DOC, SCHEMA, adapter="nope")
