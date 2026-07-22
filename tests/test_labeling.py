"""Tests for the human-labeling IAA aggregation and CLI."""

import json

import pytest
from typer.testing import CliRunner

from verifydoc.cli import app
from verifydoc.labeling import iaa_report, load_annotations


class TestIAAReport:
    def test_perfect_agreement(self):
        ann = {"a": {"f1": 1, "f2": 0, "f3": 1}, "b": {"f1": 1, "f2": 0, "f3": 1}}
        r = iaa_report(ann)
        assert r.n_annotators == 2 and r.n_items == 3
        assert r.fleiss == pytest.approx(1.0)
        assert r.mean_pairwise_cohen == pytest.approx(1.0)
        assert "almost perfect" in r.interpret()

    def test_only_co_labeled_items_scored(self):
        ann = {"a": {"f1": 1, "f2": 0, "x": 1}, "b": {"f1": 1, "f2": 1}}
        r = iaa_report(ann)
        assert r.n_items == 2  # f1, f2 shared; x dropped

    def test_three_annotators_fleiss(self):
        ann = {
            "a": {"f1": 1, "f2": 0, "f3": 1, "f4": 0},
            "b": {"f1": 1, "f2": 0, "f3": 0, "f4": 0},
            "c": {"f1": 1, "f2": 1, "f3": 1, "f4": 0},
        }
        r = iaa_report(ann)
        assert r.n_annotators == 3
        assert -1.0 <= r.fleiss <= 1.0
        assert len(r.pairwise_cohen) == 3  # 3 pairs

    def test_validation(self):
        with pytest.raises(ValueError):
            iaa_report({"a": {"f1": 1}})  # <2 annotators
        with pytest.raises(ValueError):
            iaa_report({"a": {"f1": 1}, "b": {"f2": 0}})  # no overlap


class TestLoadAndCLI:
    def _write(self, tmp_path, name, labels):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps({"annotator": name, "labels": labels}), encoding="utf-8")
        return p

    def test_load_annotations(self, tmp_path):
        p = self._write(tmp_path, "alice", {"f1": 1, "f2": 0})
        ann = load_annotations([p])
        assert ann["alice"] == {"f1": 1, "f2": 0}

    def test_cli_iaa(self, tmp_path):
        a = self._write(tmp_path, "alice", {"f1": 1, "f2": 0, "f3": 1})
        b = self._write(tmp_path, "bob", {"f1": 1, "f2": 0, "f3": 0})
        result = CliRunner().invoke(app, ["iaa", str(a), str(b)])
        assert result.exit_code == 0
        assert "kappa" in result.output.lower()
        assert "alice vs bob" in result.output
