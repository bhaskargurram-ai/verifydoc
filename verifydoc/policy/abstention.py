"""Accept/review policy: confidence threshold tuned to a target selective risk.

Threshold sources (both fit on the CALIBRATION split only, golden rule #4):
- ``empirical``: the lowest threshold whose calibration selective risk <= alpha
  (max coverage, no finite-sample cushion);
- ``conformal``: the split-conformal threshold with the finite-sample
  correction — a distribution-free E[risk] <= alpha guarantee, at the cost of
  more forced review (report both, per PROJECT.md §5.G).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from verifydoc.calibration.conformal import ConformalAbstention
from verifydoc.eval.selective import coverage_at_risk
from verifydoc.types import FieldPrediction


@dataclass
class AbstentionPolicy:
    """A fitted operating point."""

    threshold: float
    target_risk: float
    method: Literal["empirical", "conformal"]
    expected_coverage: float  # coverage on the calibration split

    def __call__(self, predictions: list[FieldPrediction]) -> list[FieldPrediction]:
        return apply_policy(predictions, self.threshold)


def threshold_for_target_risk(
    cal_conf: Sequence[float],
    cal_correct: Sequence[int],
    alpha: float,
    method: Literal["empirical", "conformal"] = "conformal",
) -> AbstentionPolicy:
    """Fit the accept threshold for a target selective risk on calibration data."""
    if method == "empirical":
        coverage, threshold = coverage_at_risk(cal_conf, cal_correct, alpha)
        return AbstentionPolicy(
            threshold=threshold, target_risk=alpha, method=method, expected_coverage=coverage
        )
    conformal = ConformalAbstention(alpha=alpha).fit(cal_conf, cal_correct)
    return AbstentionPolicy(
        threshold=conformal.threshold_,
        target_risk=alpha,
        method=method,
        expected_coverage=1.0 - conformal.abstention_rate_,
    )


def apply_policy(predictions: list[FieldPrediction], threshold: float) -> list[FieldPrediction]:
    """Set decisions: accept iff confidence >= threshold and a value exists.

    Omitted fields (value None) are always ``review`` — an omission is exactly
    what a human must resolve; confidence there scores the omission itself.
    """
    return [
        pred.model_copy(
            update={
                "decision": (
                    "accept"
                    if pred.value is not None and pred.confidence >= threshold
                    else "review"
                )
            }
        )
        for pred in predictions
    ]
