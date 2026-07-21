"""Harness extractor dispatch: any registered adapter, still offline."""

import pytest

from verifydoc.eval.harness import run_benchmark

CFG = {
    "seed": 2,
    "n_docs": 8,
    "k": 2,
    "extractor": "text-search",
    "calibration_fraction": 0.5,
    "alphas": [0.05],
    "n_boot": 20,
}


class TestExtractorDispatch:
    def test_text_search_extractor_runs(self, tmp_path):
        summary = run_benchmark(CFG, tmp_path)
        assert summary["extractor"] == "text-search"
        assert (tmp_path / "selective.md").exists()
        # the deterministic heuristic reads the synthetic invoices near-perfectly
        assert summary["extraction"]["f1"] > 0.9

    def test_unknown_extractor_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown adapter"):
            run_benchmark({**CFG, "extractor": "nope"}, tmp_path)

    def test_unknown_dataset_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown dataset"):
            run_benchmark({**CFG, "dataset": "nope"}, tmp_path)
