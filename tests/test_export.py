"""Tests for the export layer (offline)."""

import csv
import json

from verifydoc import verify_batch
from verifydoc.export import FIELDNAMES, to_csv, to_jsonl, to_records
from verifydoc.ingest import document_from_text

TEXT = "Corner Cafe\nDate: 2024-05-01\nTotal: 7.70\n"
SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}


def _results():
    docs = [document_from_text("a", [TEXT]), document_from_text("b", [TEXT])]
    return verify_batch(docs, schema=SCHEMA)


class TestToRecords:
    def test_one_row_per_field_with_trust_columns(self):
        rows = to_records(_results())
        assert rows, "expected at least one field row"
        r = rows[0]
        assert set(r) == set(FIELDNAMES)
        assert r["doc_id"] in {"a", "b"}
        assert r["decision"] in {"accept", "review"}
        assert isinstance(r["grounded"], bool)

    def test_empty(self):
        assert to_records([]) == []


class TestFileWriters:
    def test_to_csv(self, tmp_path):
        out = to_csv(_results(), tmp_path / "v.csv")
        with out.open(encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        assert reader and list(reader[0].keys()) == FIELDNAMES

    def test_to_jsonl(self, tmp_path):
        results = _results()
        out = to_jsonl(results, tmp_path / "v.jsonl")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(results)
        assert json.loads(lines[0])["doc_id"] == "a"
