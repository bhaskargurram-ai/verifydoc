"""Numeric regression tests for eval/calibration.py against hand-computed values."""

import numpy as np
import pytest

from verifydoc.eval.calibration import (
    adaptive_ece,
    brier,
    ece,
    mce,
    nll,
    reliability_bins,
    tce,
)

# Fixture A: two occupied bins under 5 equal-width bins.
#   bin (0.4, 0.6]: confs {0.55, 0.5} acc 1.0, mean conf 0.525 -> gap 0.475
#   bin (0.8, 1.0]: confs {0.9, 0.85} acc 0.5, mean conf 0.875 -> gap 0.375
CONF_A = [0.9, 0.85, 0.55, 0.5]
CORR_A = [1, 0, 1, 1]

# Fixture B: equal-width and equal-mass binning disagree.
CONF_B = [0.9, 0.8, 0.7, 0.1]
CORR_B = [1, 1, 0, 0]


class TestECE:
    def test_hand_computed(self):
        assert ece(CONF_A, CORR_A, n_bins=5) == pytest.approx(0.5 * 0.475 + 0.5 * 0.375)

    def test_equal_width_vs_equal_mass(self):
        # width-2 bins: 0.25*|0-0.1| + 0.75*|2/3-0.8| = 0.125
        assert ece(CONF_B, CORR_B, n_bins=2) == pytest.approx(0.125)
        # mass-2 bins: 0.5*|0-0.4| + 0.5*|1-0.85| = 0.275
        assert adaptive_ece(CONF_B, CORR_B, n_bins=2) == pytest.approx(0.275)

    def test_perfectly_calibrated(self):
        conf = [0.5, 0.5, 0.5, 0.5]
        corr = [1, 0, 1, 0]
        assert ece(conf, corr, n_bins=15) == pytest.approx(0.0)
        assert adaptive_ece(conf, corr, n_bins=1) == pytest.approx(0.0)

    def test_mce_is_worst_bin(self):
        assert mce(CONF_A, CORR_A, n_bins=5) == pytest.approx(0.475)

    def test_validation(self):
        with pytest.raises(ValueError):
            ece([1.5], [1])
        with pytest.raises(ValueError):
            ece([0.5, 0.5], [1])
        with pytest.raises(ValueError):
            ece([0.5], [2])
        with pytest.raises(ValueError):
            ece([], [])


class TestProperScores:
    def test_brier_hand_computed(self):
        # ((0.1)^2 + (0.2)^2 + (0.7)^2 + (0.4)^2) / 4 = 0.7/4
        assert brier([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]) == pytest.approx(0.175)

    def test_nll_hand_computed(self):
        # -(ln .9 + ln .8 + ln .3 + ln .6)/4
        assert nll([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]) == pytest.approx(0.510825624, abs=1e-8)

    def test_nll_clips_extremes(self):
        assert nll([1.0], [0]) < 30  # clipped, not inf


class TestTCE:
    def test_overconfident_hand_computed(self):
        # all conf .9, actual acc .75:
        #   alpha=0.10 -> accept all, achieved risk .25, gap .15
        #   alpha=0.05 -> predicted risk .1 > .05, accept none, gap .05
        val = tce([0.9] * 4, [1, 1, 1, 0], alphas=(0.05, 0.10))
        assert val == pytest.approx(0.10)

    def test_perfect_confidence_zero_tce(self):
        val = tce([1.0, 1.0, 1.0, 0.0], [1, 1, 1, 0], alphas=(0.0,))
        assert val == pytest.approx(0.0)

    def test_threshold_respects_tie_boundaries(self):
        # ties at .9 cannot be split: either all four accepted or none
        val = tce([0.9] * 4, [1, 1, 1, 1], alphas=(0.10,))
        assert val == pytest.approx(0.10)  # accepted all, achieved 0, |0-.1|

    def test_tune_on_separate_split(self):
        # cal split: prefix [.99,.99,.99,.9] has predicted risk .0325 <= .05,
        # so the tuned threshold is 0.9 -> every test field is accepted.
        val = tce(
            [0.9] * 4,
            [1, 1, 1, 0],
            alphas=(0.05,),
            tune_conf=[0.99, 0.99, 0.99, 0.9],
        )
        assert val == pytest.approx(abs(0.25 - 0.05))

    def test_tuned_threshold_too_high_accepts_nothing(self):
        val = tce([0.9] * 4, [1, 1, 1, 0], alphas=(0.05,), tune_conf=[0.99] * 4)
        assert val == pytest.approx(0.05)  # threshold .99 > test confs -> empty


class TestReliabilityBins:
    def test_bins_and_counts(self):
        bins = reliability_bins(CONF_A, CORR_A, n_bins=5)
        assert len(bins) == 2  # empty bins skipped
        assert bins[0].count == 2 and bins[0].accuracy == 1.0
        assert bins[1].mean_confidence == pytest.approx(0.875)

    def test_zero_confidence_in_first_bin(self):
        bins = reliability_bins([0.0, 1.0], [0, 1], n_bins=2)
        assert bins[0].lo == 0.0 and bins[0].count == 1


class TestSmoothECE:
    def test_perfectly_calibrated_near_zero(self):
        from verifydoc.eval.calibration import smooth_ece

        rng = np.random.default_rng(0)
        conf = rng.random(500)
        correct = (rng.random(500) < conf).astype(int)
        assert smooth_ece(conf, correct) < 0.1

    def test_overconfident_positive(self):
        from verifydoc.eval.calibration import smooth_ece

        # says 0.9, right 60% -> smoothed residual ~ -0.3
        assert smooth_ece([0.9] * 200, ([1] * 120 + [0] * 80)) > 0.2

    def test_validation(self):
        from verifydoc.eval.calibration import smooth_ece

        with pytest.raises(ValueError):
            smooth_ece([1.5], [1])
