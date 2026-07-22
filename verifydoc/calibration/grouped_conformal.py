"""Group-conditional (Mondrian) conformal risk control (PROJECT.md §5; novel).

Standard split-conformal abstention (``ConformalAbstention``) fits a single
acceptance threshold and controls the *pooled* selective risk. But fields are
not exchangeable across provenance: a well-grounded value is far more likely
correct than an ungrounded one (empirically ~85% vs ~1%). Pooling wastes
coverage --- to hold the pooled risk it must apply one conservative bar to
everyone.

**Grounding-conditioned conformal** partitions fields into groups by a
provenance signal (by default: grounded vs ungrounded at an IoU/support
threshold) and fits a *separate* conformal threshold per group on the
calibration split. Each group then carries its own finite-sample guarantee

    E[ selective risk | group g ] <= alpha,

which also implies the pooled guarantee, while accepting well-grounded fields
at a lower confidence bar and ungrounded fields at a stricter one. The result
is higher coverage at the same guaranteed risk (a Neyman--Pearson-style gain
from conditioning). This is the Mondrian/group-conditional conformal idea
(Vovk et al.) instantiated with *provenance* as the taxonomy --- new for
document extraction.

# DECISION: groups are defined by a caller-supplied ``group_of(pred)`` so the
# taxonomy is pluggable (grounded/ungrounded, support bins, field type). A
# group with no calibration data falls back to the pooled threshold, so the
# method never does worse than pooled conformal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from verifydoc.calibration.conformal import ConformalAbstention
from verifydoc.types import FieldPrediction

GroupFn = Callable[[FieldPrediction], str]


def grounded_group(pred: FieldPrediction, min_support: float = 0.5) -> str:
    """Default taxonomy: 'grounded' if provenance support >= threshold else 'ungrounded'."""
    if pred.grounding is not None and pred.grounding.support >= min_support:
        return "grounded"
    return "ungrounded"


@dataclass
class GroupConformalAbstention:
    """Per-group split-conformal accept policy with a per-group risk guarantee."""

    alpha: float = 0.05
    group_of: GroupFn = grounded_group
    thresholds_: dict[str, float] = field(default_factory=dict)
    abstention_: dict[str, float] = field(default_factory=dict)
    pooled_threshold_: float = float("inf")
    _fitted: bool = False

    def fit(
        self, preds: Sequence[FieldPrediction], correct: Sequence[int]
    ) -> GroupConformalAbstention:
        """Fit a conformal threshold per group on the CALIBRATION split only."""
        if len(preds) != len(correct):
            raise ValueError("preds and correct must be the same length")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        conf = np.array([p.confidence for p in preds], dtype=float)
        corr = np.asarray(correct, dtype=float)
        groups = np.array([self.group_of(p) for p in preds])

        # pooled fallback threshold (for empty/degenerate groups)
        pooled = ConformalAbstention(alpha=self.alpha).fit(conf, corr)
        self.pooled_threshold_ = pooled.threshold_

        self.thresholds_ = {}
        self.abstention_ = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            policy = ConformalAbstention(alpha=self.alpha).fit(conf[mask], corr[mask])
            self.thresholds_[str(g)] = policy.threshold_
            self.abstention_[str(g)] = policy.abstention_rate_
        self._fitted = True
        return self

    def threshold_for(self, group: str) -> float:
        return self.thresholds_.get(group, self.pooled_threshold_)

    def accept(self, preds: Sequence[FieldPrediction]) -> np.ndarray:
        """Boolean accept mask: each field cleared against its own group threshold."""
        if not self._fitted:
            raise RuntimeError(
                "GroupConformalAbstention must be fit on the calibration split first"
            )
        return np.array(
            [p.confidence >= self.threshold_for(self.group_of(p)) for p in preds], dtype=bool
        )

    @property
    def guarantee(self) -> str:
        parts = ", ".join(
            f"{g}: thr={t:.3f} (abstain {self.abstention_.get(g, 1.0):.0%})"
            for g, t in sorted(self.thresholds_.items())
        )
        return f"E[selective risk | group] <= {self.alpha:.3f} per group; {parts}"
