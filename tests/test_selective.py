"""Numeric regression tests for eval/selective.py against hand-computed values.

Canonical 4-field toy set: conf = [.9, .8, .7, .6], correct = [1, 1, 0, 1].
Sorted prefixes give risks 0, 0, 1/3, 1/4:
  AURC        = (0 + 0 + 1/3 + 1/4) / 4 = 7/48
  oracle AURC = (0 + 0 + 0   + 1/4) / 4 = 1/16
  E-AURC      = 7/48 - 3/48            = 1/12
"""

import math

import numpy as np
import pytest

from verifydoc.eval.selective import (
    accuracy_at_k,
    aupr,
    aurc,
    auroc,
    coverage_at_risk,
    e_aurc,
    fpr_at_tpr,
    oracle_aurc,
    rc_curve,
    risk_at_coverage,
)

CONF = [0.9, 0.8, 0.7, 0.6]
CORR = [1, 1, 0, 1]


class TestRCCurve:
    def test_hand_computed_curve(self):
        coverage, risk = rc_curve(CONF, CORR)
        assert np.allclose(coverage, [0.25, 0.5, 0.75, 1.0])
        assert np.allclose(risk, [0.0, 0.0, 1 / 3, 0.25])

    def test_aurc(self):
        assert aurc(CONF, CORR) == pytest.approx(7 / 48)

    def test_oracle_aurc(self):
        assert oracle_aurc(CORR) == pytest.approx(1 / 16)

    def test_e_aurc(self):
        assert e_aurc(CONF, CORR) == pytest.approx(1 / 12)

    def test_perfect_ranking_zero_excess(self):
        assert e_aurc([0.9, 0.8, 0.7, 0.1], [1, 1, 1, 0]) == pytest.approx(0.0)


class TestOperatingPoints:
    def test_coverage_at_zero_risk(self):
        cov, thr = coverage_at_risk(CONF, CORR, alpha=0.0)
        assert cov == pytest.approx(0.5)
        assert thr == pytest.approx(0.8)

    def test_coverage_at_quarter_risk(self):
        cov, _ = coverage_at_risk(CONF, CORR, alpha=0.25)
        assert cov == pytest.approx(1.0)  # k=4 risk .25 qualifies; k=3 (1/3) does not

    def test_coverage_none_qualifies(self):
        cov, thr = coverage_at_risk([0.9, 0.8], [0, 0], alpha=0.1)
        assert cov == 0.0 and thr == float("inf")

    def test_ties_cut_only_at_boundaries(self):
        # top tie group {.9, .9} contains an error: risk 0 unreachable
        cov, _ = coverage_at_risk([0.9, 0.9, 0.5], [1, 0, 1], alpha=0.0)
        assert cov == 0.0

    def test_risk_at_coverage(self):
        assert risk_at_coverage(CONF, CORR, 0.75) == pytest.approx(1 / 3)
        assert risk_at_coverage(CONF, CORR, 1.0) == pytest.approx(0.25)
        with pytest.raises(ValueError):
            risk_at_coverage(CONF, CORR, 0.0)

    def test_accuracy_at_k(self):
        assert accuracy_at_k(CONF, CORR, 0.5) == pytest.approx(1.0)
        assert accuracy_at_k(CONF, CORR, 0.75) == pytest.approx(2 / 3)


class TestErrorDetection:
    def test_auroc_hand_computed(self):
        # correct confs {.9,.8,.6} vs error {.7}: 2 of 3 pairs ranked right
        assert auroc(CONF, CORR) == pytest.approx(2 / 3)

    def test_auroc_with_ties(self):
        # scores 1-conf: correct {.1}, error {.1}: single tied pair -> 0.5
        assert auroc([0.9, 0.9], [1, 0]) == pytest.approx(0.5)

    def test_aupr_hand_computed(self):
        # detection scores desc: .4(neg) .3(pos) .2(neg) .1(neg)
        # single positive found at precision 1/2 -> AP = 0.5
        assert aupr(CONF, CORR) == pytest.approx(0.5)

    def test_aupr_perfect(self):
        assert aupr([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)

    def test_fpr_at_95_tpr(self):
        # flagging the only error (score .3) also flags the .4-scored correct field
        assert fpr_at_tpr(CONF, CORR, 0.95) == pytest.approx(1 / 3)

    def test_degenerate_returns_nan(self):
        assert math.isnan(auroc([0.9, 0.8], [1, 1]))
        assert math.isnan(aupr([0.9, 0.8], [0, 0]))
        assert math.isnan(fpr_at_tpr([0.9], [1]))


class TestAUGRC:
    def test_hand_computed(self):
        from verifydoc.eval.selective import augrc

        # conf=[.9,.8,.7,.6], correct=[1,1,0,1]: cum_err=[0,0,1,1], /4 -> [0,0,.25,.25]
        assert augrc(CONF, CORR) == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)

    def test_perfect_ranking_low(self):
        from verifydoc.eval.selective import augrc

        # all errors last -> undetected failures accumulate only at the end
        good = augrc([0.9, 0.8, 0.7, 0.1], [1, 1, 1, 0])
        bad = augrc([0.9, 0.8, 0.7, 0.1], [0, 1, 1, 1])
        assert good < bad
