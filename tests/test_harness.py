"""Tests for the benchmark harness: end-to-end, offline, deterministic."""

import json

import pytest

from benchmark.datasets import synthetic
from verifydoc.eval.harness import run_benchmark

TINY_CFG = {
    "seed": 3,
    "n_docs": 14,
    "k": 3,
    "error_rate": 0.25,
    "omit_rate": 0.05,
    "hallucinate_rate": 0.05,
    "calibration_fraction": 0.5,
    "alphas": [0.05],
    "n_boot": 30,
}


class TestSyntheticDataset:
    def test_deterministic(self):
        a = synthetic.generate(n_docs=5, seed=1)
        b = synthetic.generate(n_docs=5, seed=1)
        assert [x.doc.doc_id for x in a] == [y.doc.doc_id for y in b]
        assert a[0].golds[0].value == b[0].golds[0].value

    def test_gold_boxes_present(self):
        (item,) = synthetic.generate(n_docs=1, seed=0)
        boxed = [g for g in item.golds if g.gold_box is not None]
        assert len(boxed) == len(item.golds)  # every gold value is locatable
        for g in boxed:
            assert g.gold_box.support >= 0.99

    def test_schema_totals_consistent(self):
        (item,) = synthetic.generate(n_docs=1, seed=2)
        values = {
            g.path: float(str(g.value))
            for g in item.golds
            if g.path != "invoice_id" and g.path != "vendor" and g.path != "date"
        }
        assert values["total"] == pytest.approx(values["subtotal"] + values["tax"], abs=0.011)


class TestHarness:
    def test_end_to_end_writes_tables_and_figures(self, tmp_path):
        summary = run_benchmark(TINY_CFG, tmp_path)
        for table in (
            "extraction.md",
            "calibration.md",
            "selective.md",
            "conformal.md",
            "grounding.md",
        ):
            content = (tmp_path / table).read_text(encoding="utf-8")
            assert content.startswith("#") and "|" in content
        assert (tmp_path / "rc_curves.png").exists()
        assert (tmp_path / "reliability.png").exists()
        assert summary["n_cal"] + summary["n_test"] == TINY_CFG["n_docs"]

    def test_deterministic_summary(self, tmp_path):
        a = run_benchmark(TINY_CFG, tmp_path / "a")
        b = run_benchmark(TINY_CFG, tmp_path / "b")
        assert json.dumps(a, default=str, sort_keys=True) == json.dumps(
            b, default=str, sort_keys=True
        )

    def test_signals_carry_real_signal(self, tmp_path):
        """The USP sanity check: informative signals must rank errors better
        than the deliberately-uninformative verbalized signal."""
        summary = run_benchmark(TINY_CFG, tmp_path)
        assert summary["best_coverage"]["signal"] in ("combined", "consensus", "grounding")

    def test_grounding_gap_positive(self, tmp_path):
        """Grounded-correctness hypothesis: well-grounded fields are more
        often correct than ungrounded ones (corrupted values don't appear on
        the page, so they can't be grounded)."""
        summary = run_benchmark(TINY_CFG, tmp_path)
        assert summary["grounding"]["grounding_gap"] > 0.2

    def test_conformal_guarantee_row_present(self, tmp_path):
        run_benchmark(TINY_CFG, tmp_path)
        content = (tmp_path / "conformal.md").read_text(encoding="utf-8")
        assert "guarantee_held" in content
