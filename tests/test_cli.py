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


class TestBatch:
    def _folder(self, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        (d / "a.txt").write_text(DOC, encoding="utf-8")
        (d / "b.txt").write_text("Bistro\nTotal: 12.00\n", encoding="utf-8")
        schema = tmp_path / "schema.json"
        schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
        return d, schema

    def test_writes_one_json_per_doc_plus_summary(self, tmp_path):
        d, schema = self._folder(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, ["batch", str(d), "-s", str(schema), "-o", str(out)])
        assert result.exit_code == 0
        assert (out / "a.json").exists() and (out / "b.json").exists()
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        assert summary["n_docs"] == 2
        assert {r["file"] for r in summary["documents"]} == {"a.txt", "b.txt"}
        assert summary["total_accepted"] + summary["total_review"] >= 2
        assert "2 docs" in result.output

    def test_per_doc_json_is_a_verified_result(self, tmp_path):
        d, schema = self._folder(tmp_path)
        out = tmp_path / "out"
        runner.invoke(app, ["batch", str(d), "-s", str(schema), "-o", str(out)])
        data = json.loads((out / "a.json").read_text(encoding="utf-8"))
        assert "fields" in data and "n_accepted" in data

    def test_glob_filters_files(self, tmp_path):
        d, schema = self._folder(tmp_path)
        (d / "notes.md").write_text("ignore me", encoding="utf-8")
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["batch", str(d), "-s", str(schema), "-o", str(out), "--glob", "*.txt"]
        )
        assert result.exit_code == 0
        assert not (out / "notes.json").exists()
        assert json.loads((out / "summary.json").read_text())["n_docs"] == 2

    def test_empty_folder_exits_nonzero(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        _, schema = self._folder(tmp_path)
        result = runner.invoke(app, ["batch", str(empty), "-s", str(schema)])
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
