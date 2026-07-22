"""Tests for grounding-conditioned (Mondrian) conformal risk control.

Includes the two claims the paper makes: (1) the per-group guarantee holds
empirically under exchangeability, and (2) conditioning on grounding lifts
coverage over pooled conformal at the same target risk.
"""

import numpy as np
import pytest

from verifydoc.calibration import ConformalAbstention, GroupConformalAbstention, grounded_group
from verifydoc.types import FieldPrediction, Grounding


def _pred(conf, support):
    g = Grounding(page=0, support=support) if support is not None else None
    return FieldPrediction(path="f", value="v", confidence=conf, grounding=g)


class TestGroupTaxonomy:
    def test_grounded_group_split(self):
        assert grounded_group(_pred(0.9, 0.8)) == "grounded"
        assert grounded_group(_pred(0.9, 0.2)) == "ungrounded"
        assert grounded_group(_pred(0.9, None)) == "ungrounded"


class TestFitAndAccept:
    def test_per_group_thresholds_differ(self):
        # grounded fields: high accuracy -> low threshold; ungrounded: strict
        rng = np.random.default_rng(0)
        preds, corr = [], []
        for _ in range(400):
            preds.append(_pred(0.6 + 0.4 * rng.random(), 0.9))  # grounded
            corr.append(int(rng.random() < 0.97))
        for _ in range(400):
            preds.append(_pred(0.6 + 0.4 * rng.random(), 0.1))  # ungrounded
            corr.append(int(rng.random() < 0.55))
        policy = GroupConformalAbstention(alpha=0.05).fit(preds, corr)
        assert "grounded" in policy.thresholds_ and "ungrounded" in policy.thresholds_
        # grounded group can accept at a lower confidence bar than ungrounded
        assert policy.threshold_for("grounded") <= policy.threshold_for("ungrounded")

    def test_empty_group_falls_back_to_pooled(self):
        preds = [_pred(0.9, 0.9) for _ in range(20)]  # all grounded
        policy = GroupConformalAbstention(alpha=0.1).fit(preds, [1] * 20)
        # an ungrounded field at query time uses the pooled threshold, not a crash
        mask = policy.accept([_pred(0.95, 0.1)])
        assert mask.shape == (1,)

    def test_requires_fit(self):
        with pytest.raises(RuntimeError):
            GroupConformalAbstention().accept([_pred(0.9, 0.9)])

    def test_validation(self):
        with pytest.raises(ValueError):
            GroupConformalAbstention().fit([_pred(0.9, 0.9)], [1, 0])
        with pytest.raises(ValueError):
            GroupConformalAbstention(alpha=1.5).fit([_pred(0.9, 0.9)], [1])


class TestGuaranteeAndCoverage:
    def _make(self, rng, n):
        """A population where grounded fields are reliable and ungrounded are not."""
        preds, corr = [], []
        for _ in range(n):
            if rng.random() < 0.6:  # grounded
                c = 0.5 + 0.5 * rng.random()
                y = int(rng.random() < 0.95)
                preds.append(_pred(c, 0.9))
            else:  # ungrounded
                c = 0.5 + 0.5 * rng.random()
                y = int(rng.random() < 0.5)
                preds.append(_pred(c, 0.1))
            corr.append(y)
        return preds, np.array(corr, dtype=float)

    def test_per_group_guarantee_holds(self):
        rng = np.random.default_rng(7)
        alpha = 0.10
        risks = {"grounded": [], "ungrounded": []}
        for _ in range(150):
            cal_p, cal_y = self._make(rng, 300)
            test_p, test_y = self._make(rng, 300)
            policy = GroupConformalAbstention(alpha=alpha).fit(cal_p, cal_y)
            mask = policy.accept(test_p)
            groups = np.array([grounded_group(p) for p in test_p])
            for g in ("grounded", "ungrounded"):
                sel = mask & (groups == g)
                if sel.any():
                    risks[g].append(1.0 - test_y[sel].mean())
        # a group that (correctly) abstains fully trivially satisfies the guarantee;
        # where a group accepts, its average selective risk must be <= alpha.
        for g in ("grounded", "ungrounded"):
            if risks[g]:
                assert np.mean(risks[g]) <= alpha + 0.02, f"group {g} risk too high"
        assert risks["grounded"], "grounded group should accept a meaningful fraction"

    def test_conditioning_lifts_coverage_vs_pooled(self):
        rng = np.random.default_rng(3)
        alpha = 0.10
        grouped_cov, pooled_cov = [], []
        for _ in range(120):
            cal_p, cal_y = self._make(rng, 400)
            test_p, test_y = self._make(rng, 400)
            grouped = GroupConformalAbstention(alpha=alpha).fit(cal_p, cal_y)
            conf = np.array([p.confidence for p in cal_p])
            pooled = ConformalAbstention(alpha=alpha).fit(conf, cal_y)
            test_conf = np.array([p.confidence for p in test_p])
            grouped_cov.append(grouped.accept(test_p).mean())
            pooled_cov.append(pooled.accept(test_conf).mean())
        # grounding-conditioned conformal accepts strictly more at the same risk target
        assert np.mean(grouped_cov) > np.mean(pooled_cov) + 0.02

    def test_pooled_risk_also_controlled(self):
        # per-group control implies pooled control
        rng = np.random.default_rng(11)
        alpha = 0.10
        risks = []
        for _ in range(150):
            cal_p, cal_y = self._make(rng, 300)
            test_p, test_y = self._make(rng, 300)
            policy = GroupConformalAbstention(alpha=alpha).fit(cal_p, cal_y)
            mask = policy.accept(test_p)
            if mask.any():
                risks.append(1.0 - test_y[mask].mean())
        assert np.mean(risks) <= alpha + 0.02
