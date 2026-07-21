"""Tests for the learned signal combiner and its harness row."""

import numpy as np
import pytest

from verifydoc.confidence.learned import LearnedCombiner
from verifydoc.eval.harness import run_benchmark


def make_data(n=400, seed=0):
    """One informative signal (grounding), one useless (verbalized)."""
    rng = np.random.default_rng(seed)
    correct = rng.random(n) < 0.7
    signals = [
        {
            "consensus": None,
            "grounding": (0.9 if c else 0.1) + rng.normal(0, 0.05),
            "token_prob": None,
            "verbalized": 0.9 + rng.normal(0, 0.03),
        }
        for c in correct
    ]
    return signals, correct.astype(int).tolist()


class TestLearnedCombiner:
    def test_learns_the_informative_signal(self):
        signals, correct = make_data()
        model = LearnedCombiner().fit(signals[:200], correct[:200])
        preds = model.predict(signals[200:])
        acc = float(np.mean((preds > 0.5) == (np.array(correct[200:]) == 1)))
        assert acc > 0.95

    def test_missing_indicator_features(self):
        # all-None signals still featurize and fit
        signals = [{"consensus": None, "grounding": None}] * 10
        model = LearnedCombiner().fit(signals, [1, 0] * 5)
        assert 0.0 <= float(model.predict(signals)[0]) <= 1.0

    def test_degenerate_split_predicts_base_rate(self):
        signals, _ = make_data(n=10)
        model = LearnedCombiner().fit(signals, [1] * 10)
        assert model.predict(signals)[0] == pytest.approx(1.0)

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            LearnedCombiner().predict([{}])

    def test_validation(self):
        with pytest.raises(ValueError):
            LearnedCombiner().fit([], [])


class TestHarnessLearnedRow:
    CFG = {
        "seed": 5,
        "n_docs": 14,
        "k": 3,
        "error_rate": 0.25,
        "calibration_fraction": 0.5,
        "alphas": [0.05],
        "n_boot": 20,
    }

    def test_learned_row_in_tables(self, tmp_path):
        run_benchmark(self.CFG, tmp_path)
        selective = (tmp_path / "selective.md").read_text(encoding="utf-8")
        assert "learned" in selective

    def test_learned_row_can_be_disabled(self, tmp_path):
        run_benchmark({**self.CFG, "learned_combiner": False}, tmp_path)
        selective = (tmp_path / "selective.md").read_text(encoding="utf-8")
        assert "learned" not in selective
