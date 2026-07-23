"""Tests for the typer CLI (offline, text-search adapter — no network)."""

import json

from typer.testing import CliRunner

import verifydoc
from verifydoc.cli import app

runner = CliRunner()

DOC = "Corner Cafe\nDate: 2024-05-01\nTotal: 7.70\n"
SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}


def _files(tmp_path):
    doc = tmp_path / "doc.txt"
    doc.write_text(DOC, encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
    return doc, schema


class TestVersion:
    def test_prints_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert verifydoc.__version__ in result.output


class TestExtract:
    def test_to_stdout_emits_valid_json(self, tmp_path):
        doc, schema = _files(tmp_path)
        result = runner.invoke(app, ["extract", str(doc), "--schema", str(schema)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "fields" in data and "n_accepted" in data

    def test_writes_out_file(self, tmp_path):
        doc, schema = _files(tmp_path)
        out = tmp_path / "result.json"
        result = runner.invoke(app, ["extract", str(doc), "-s", str(schema), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "accepted" in result.output
        assert "fields" in json.loads(out.read_text(encoding="utf-8"))

    def test_unknown_adapter_fails(self, tmp_path):
        doc, schema = _files(tmp_path)
        result = runner.invoke(app, ["extract", str(doc), "-s", str(schema), "-a", "nope"])
        assert result.exit_code != 0

    def test_missing_source_fails(self, tmp_path):
        _, schema = _files(tmp_path)
        result = runner.invoke(app, ["extract", str(tmp_path / "nope.txt"), "-s", str(schema)])
        assert result.exit_code != 0


class TestIaa:
    def test_reports_kappa(self, tmp_path):
        a = tmp_path / "a.json"
        a.write_text(json.dumps({"annotator": "a", "labels": {"f1": 1, "f2": 0, "f3": 1}}))
        b = tmp_path / "b.json"
        b.write_text(json.dumps({"annotator": "b", "labels": {"f1": 1, "f2": 0, "f3": 0}}))
        result = runner.invoke(app, ["iaa", str(a), str(b)])
        assert result.exit_code == 0
        assert "kappa" in result.output.lower()
