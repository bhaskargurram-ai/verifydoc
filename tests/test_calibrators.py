"""Tests for the calibrators: numeric behavior, guardrails, split enforcement."""

import numpy as np
import pytest

from verifydoc.calibration import (
    ConformalAbstention,
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
    assert_disjoint,
    split_calibration,
)
from verifydoc.eval.calibration import ece, nll

# Overconfident fixture: model says .9 everywhere but is right only 60%.
RNG = np.random.default_rng(42)
N = 500
OVER_CONF = np.full(N, 0.9)
OVER_CORR = (RNG.random(N) < 0.6).astype(float)


class TestTemperature:
    def test_softens_overconfidence(self):
        cal = TemperatureScaling().fit(OVER_CONF, OVER_CORR)
        assert cal.temperature_ > 1.0
        # with a single confidence level, NLL is minimized at p' = accuracy
        mapped = cal.transform([0.9])[0]
        assert mapped == pytest.approx(OVER_CORR.mean(), abs=0.01)

    def test_reduces_nll_and_ece(self):
        cal = TemperatureScaling().fit(OVER_CONF, OVER_CORR)
        raw = list(OVER_CONF)
        cooked = list(cal.transform(OVER_CONF))
        corr = list(OVER_CORR.astype(int))
        assert nll(cooked, corr) < nll(raw, corr)
        assert ece(cooked, corr) < ece(raw, corr)

    def test_well_calibrated_left_alone(self):
        conf = np.array([0.2, 0.4, 0.6, 0.8] * 50)
        corr = (np.random.default_rng(0).random(200) < conf).astype(float)
        cal = TemperatureScaling().fit(conf, corr)
        assert 0.5 < cal.temperature_ < 2.0

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            TemperatureScaling().transform([0.5])


class TestPlatt:
    def test_fits_shifted_scores(self):
        cal = PlattScaling().fit(OVER_CONF, OVER_CORR)
        assert cal.transform([0.9])[0] == pytest.approx(OVER_CORR.mean(), abs=0.02)

    def test_degenerate_split_identity(self):
        cal = PlattScaling().fit([0.7, 0.8], [1, 1])
        assert cal.transform([0.7])[0] == pytest.approx(0.7, abs=1e-4)


class TestIsotonic:
    def test_maps_to_empirical_accuracy(self):
        conf = [0.1, 0.2, 0.8, 0.9]
        corr = [0, 0, 1, 1]
        cal = IsotonicCalibrator().fit(conf, corr)
        out = cal.transform([0.15, 0.85])
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(1.0)

    def test_monotone(self):
        conf = RNG.random(300)
        corr = (RNG.random(300) < conf).astype(float)
        cal = IsotonicCalibrator().fit(conf, corr)
        grid = np.linspace(0, 1, 50)
        mapped = cal.transform(grid)
        assert (np.diff(mapped) >= -1e-12).all()


class TestHistogram:
    def test_hand_computed_bins(self):
        # 2 bins: [0,.5) acc 0 ; (.5,1] acc 2/3
        cal = HistogramBinning(n_bins=2).fit([0.9, 0.9, 0.6, 0.1], [1, 0, 1, 0])
        assert cal.transform([0.3])[0] == pytest.approx(0.0)
        assert cal.transform([0.7])[0] == pytest.approx(2 / 3)

    def test_empty_bin_falls_back_to_global(self):
        cal = HistogramBinning(n_bins=4).fit([0.9, 0.9], [1, 0])
        assert cal.transform([0.1])[0] == pytest.approx(0.5)  # global accuracy


class TestConformal:
    def test_hand_computed_threshold(self):
        # sorted conf [.9,.8,.7,.6,.5], correct [1,1,1,0,0]
        # bounds (errors+1)/(k+1): .5, .333, .25, .4, .5 -> first <= .25 at k=3
        cal = ConformalAbstention(alpha=0.25).fit([0.9, 0.8, 0.7, 0.6, 0.5], [1, 1, 1, 0, 0])
        assert cal.threshold_ == pytest.approx(0.7)
        assert cal.abstention_rate_ == pytest.approx(0.4)
        assert list(cal.accept([0.75, 0.65])) == [True, False]

    def test_impossible_target_reviews_everything(self):
        cal = ConformalAbstention(alpha=0.01).fit([0.9, 0.8], [0, 0])
        assert cal.threshold_ == float("inf")
        assert cal.abstention_rate_ == 1.0
        assert not cal.accept([0.99]).any()

    def test_risk_guarantee_holds_empirically(self):
        # exchangeable cal/test: average achieved risk must be <= alpha
        rng = np.random.default_rng(7)
        risks = []
        for _trial in range(200):
            conf = rng.random(400)
            corr = (rng.random(400) < conf).astype(float)
            cal_c, test_c = conf[:200], conf[200:]
            cal_y, test_y = corr[:200], corr[200:]
            policy = ConformalAbstention(alpha=0.10).fit(cal_c, cal_y)
            mask = policy.accept(test_c)
            if mask.any():
                risks.append(1.0 - test_y[mask].mean())
        assert np.mean(risks) <= 0.10 + 0.01

    def test_guarantee_string(self):
        cal = ConformalAbstention(alpha=0.05).fit([0.99] * 50, [1] * 50)
        assert "0.050" in cal.guarantee

    def test_bad_alpha(self):
        with pytest.raises(ValueError):
            ConformalAbstention(alpha=1.5)


class TestSplits:
    def test_disjoint_and_complete(self):
        ids = [f"doc{i}" for i in range(20)]
        cal, test = split_calibration(ids, 0.3, seed=1)
        assert set(cal) | set(test) == set(ids)
        assert not set(cal) & set(test)
        assert len(cal) == 6

    def test_deterministic(self):
        ids = list(range(50))
        assert split_calibration(ids, seed=9) == split_calibration(ids, seed=9)

    def test_assert_disjoint_raises(self):
        with pytest.raises(ValueError, match="never see test"):
            assert_disjoint(["a", "b"], ["b", "c"])
        assert_disjoint(["a"], ["b"])  # no raise

    def test_split_validation(self):
        with pytest.raises(ValueError):
            split_calibration(["only"], 0.5)
        with pytest.raises(ValueError):
            split_calibration([1, 2, 3], 1.5)
