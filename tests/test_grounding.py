"""Numeric regression tests for eval/grounding.py against hand-computed values."""

import pytest

from verifydoc.eval.grounding import (
    box_grounding_accuracy,
    grounding_conditioned_correctness,
    iou,
    mean_iou,
    pairwise_ious,
    span_f1,
    span_grounding_f1,
)

UNIT = (0.0, 0.0, 1.0, 1.0)
HALF_SHIFT = (0.5, 0.0, 1.5, 1.0)  # inter .5, union 1.5 -> IoU 1/3


class TestIoU:
    def test_hand_computed(self):
        assert iou(UNIT, HALF_SHIFT) == pytest.approx(1 / 3)
        assert iou(UNIT, UNIT) == pytest.approx(1.0)

    def test_disjoint(self):
        assert iou(UNIT, (2.0, 2.0, 3.0, 3.0)) == 0.0

    def test_degenerate(self):
        assert iou((0, 0, 0, 0), UNIT) == 0.0


class TestBoxMetrics:
    PREDS = [UNIT, HALF_SHIFT, None, UNIT]
    GOLDS = [UNIT, UNIT, UNIT, None]

    def test_pairwise(self):
        vals = pairwise_ious(self.PREDS, self.GOLDS)
        assert vals[0] == pytest.approx(1.0)
        assert vals[1] == pytest.approx(1 / 3)
        assert vals[2] == 0.0  # missing pred box vs gold box = miss
        assert vals[3] is None  # no gold box: excluded

    def test_accuracy_at_tau(self):
        # of 3 gold-boxed fields: IoUs {1.0, 1/3, 0.0}
        assert box_grounding_accuracy(self.PREDS, self.GOLDS, tau=0.5) == pytest.approx(1 / 3)
        assert box_grounding_accuracy(self.PREDS, self.GOLDS, tau=0.3) == pytest.approx(2 / 3)

    def test_mean_iou_only_both(self):
        # both-box fields: IoUs {1.0, 1/3} -> mean 2/3
        assert mean_iou(self.PREDS, self.GOLDS) == pytest.approx(2 / 3)

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            pairwise_ious([None], [])


class TestSpanF1:
    def test_hand_computed(self):
        # spans [0,10) vs [5,15): overlap 5 -> P = R = .5 -> F1 = .5
        assert span_f1((0, 10), (5, 15)) == pytest.approx(0.5)

    def test_exact_and_disjoint(self):
        assert span_f1((3, 7), (3, 7)) == pytest.approx(1.0)
        assert span_f1((0, 5), (5, 10)) == 0.0
        assert span_f1(None, (0, 5)) == 0.0

    def test_macro_average(self):
        preds = [(0, 10), (3, 7), None]
        golds = [(5, 15), (3, 7), None]  # third has no gold: excluded
        assert span_grounding_f1(preds, golds) == pytest.approx(0.75)


class TestGroundingConditioned:
    def test_hypothesis_report(self):
        correct = [1, 1, 0, 0]
        preds = [UNIT, UNIT, HALF_SHIFT, None]
        golds = [UNIT, UNIT, UNIT, UNIT]
        report = grounding_conditioned_correctness(correct, preds, golds, tau=0.5)
        assert report.accuracy_grounded == pytest.approx(1.0)  # fields 0,1
        assert report.accuracy_ungrounded == pytest.approx(0.0)  # fields 2,3
        assert report.n_grounded == 2 and report.n_ungrounded == 2
        assert report.gap == pytest.approx(1.0)
