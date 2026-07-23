"""Tests for the grouped-conformal characterization diagnostic (calibration-split
predictor of the coverage gain). Controlled fixtures pin the two directions:
heterogeneous groups -> recommend grouped; homogeneous -> recommend pooled."""

import numpy as np
import pytest

from verifydoc.calibration.characterization import (
    CharacterizationReport,
    characterize,
    predict_coverage_gain,
)
from verifydoc.calibration.grouped_conformal import grounded_group
from verifydoc.types import FieldPrediction, Grounding


def _fp(conf, support, correct_p, rng):
    g = Grounding(page=0, support=support) if support is not None else None
    pred = FieldPrediction(path="f", value="v", confidence=conf, grounding=g)
    return pred, int(rng.random() < correct_p)


def _heterogeneous(rng, n):
    """Grounded fields reliable (~95%) with discriminative confidence; ungrounded
    unreliable (~50%). Group-conditional conformal should help here."""
    preds, corr = [], []
    for _ in range(n):
        if rng.random() < 0.6:
            p, y = _fp(0.5 + 0.5 * rng.random(), 0.9, 0.95, rng)
        else:
            p, y = _fp(0.5 + 0.5 * rng.random(), 0.1, 0.5, rng)
        preds.append(p)
        corr.append(y)
    return preds, np.array(corr, dtype=float)


def _homogeneous(rng, n):
    """Both groups have the same error rate -> conditioning cannot help."""
    preds, corr = [], []
    for _ in range(n):
        support = 0.9 if rng.random() < 0.5 else 0.1
        p, y = _fp(0.5 + 0.5 * rng.random(), support, 0.8, rng)
        preds.append(p)
        corr.append(y)
    return preds, np.array(corr, dtype=float)


class TestCharacterize:
    def test_recommends_grouped_when_heterogeneous(self):
        rng = np.random.default_rng(0)
        preds, corr = _heterogeneous(rng, 600)
        rep = characterize(preds, corr, grounded_group, alpha=0.10)
        assert isinstance(rep, CharacterizationReport)
        assert rep.error_separation > 0.2  # grounded ~5% err vs ungrounded ~50%
        assert rep.predicted_gain > 0.02
        assert rep.recommend is True
        assert set(rep.group_error_rates) == {"grounded", "ungrounded"}

    def test_recommends_pooled_when_homogeneous(self):
        rng = np.random.default_rng(1)
        preds, corr = _homogeneous(rng, 600)
        rep = characterize(preds, corr, grounded_group, alpha=0.10)
        assert rep.error_separation < 0.1
        assert rep.predicted_gain < 0.02
        assert rep.recommend is False

    def test_summary_string(self):
        rng = np.random.default_rng(2)
        preds, corr = _heterogeneous(rng, 200)
        assert "coverage gain" in characterize(preds, corr).summary()


class TestPredictCoverageGain:
    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(3)
        preds, corr = _heterogeneous(rng, 300)
        a = predict_coverage_gain(preds, corr, grounded_group, alpha=0.10, seed=5)
        b = predict_coverage_gain(preds, corr, grounded_group, alpha=0.10, seed=5)
        assert a == b

    def test_length_mismatch_raises(self):
        rng = np.random.default_rng(4)
        preds, _ = _heterogeneous(rng, 10)
        with pytest.raises(ValueError):
            predict_coverage_gain(preds, [1, 0], grounded_group)
