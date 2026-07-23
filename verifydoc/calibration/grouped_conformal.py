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
        pooled = ConformalAbstention(alpha=self.alpha).fit(conf.tolist(), corr.tolist())
        self.pooled_threshold_ = pooled.threshold_

        self.thresholds_ = {}
        self.abstention_ = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() == 0:
                continue
            policy = ConformalAbstention(alpha=self.alpha).fit(
                conf[mask].tolist(), corr[mask].tolist()
            )
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


# ---------------------------------------------------------------------------
# Richer provenance taxonomies (the paper's grouping ablation). Every taxonomy
# uses only inference-available fields (value, grounding.support) — never the
# gold box — so the group is known at accept time.
# ---------------------------------------------------------------------------


def _num(value: object) -> float | None:
    """Best-effort numeric parse (self-contained; no eval dependency)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = "".join(ch for ch in str(value) if ch.isdigit() or ch in ".-")
    if s in ("", ".", "-", "-.", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def support_bin_group(pred: FieldPrediction, edges: Sequence[float] = (0.5, 0.8)) -> str:
    """Provenance-quality taxonomy: bin grounded fields by support strength.

    Ungrounded fields form their own group; grounded fields split at ``edges``
    into support bins, so a spuriously-grounded low-support field lands in a
    stricter group than a confidently-located one — finer than binary grounded.
    """
    if pred.grounding is None:
        return "ungrounded"
    s = float(pred.grounding.support)
    return f"supp{int(np.searchsorted(np.asarray(edges, dtype=float), s, side='right'))}"


def value_length_group(pred: FieldPrediction, edges: Sequence[int] = (3, 8)) -> str:
    """Value-length taxonomy: short values (bare numerics) ground coincidentally
    on many tokens, so they are the uncertifiable group. Binning by string
    length quarantines them so longer, reliably-grounded values can still
    certify. Combine with grounding via :func:`combine_groups`.
    """
    n = len(str(pred.value if pred.value is not None else ""))
    return f"len{int(np.searchsorted(np.asarray(edges, dtype=int), n, side='right'))}"


def field_type_group(pred: FieldPrediction) -> str:
    """Coarse value-type taxonomy: numeric vs text vs empty."""
    if pred.value is None or str(pred.value).strip() == "":
        return "empty"
    return "numeric" if _num(pred.value) is not None else "text"


def combine_groups(*fns: GroupFn) -> GroupFn:
    """Cross-product taxonomy: label is each fn's label joined by ``|``."""

    def _combined(pred: FieldPrediction) -> str:
        return "|".join(fn(pred) for fn in fns)

    return _combined


# Named catalog shared by the harness grouping-ablation and GroupPartitionSelector.
GROUP_TAXONOMIES: dict[str, GroupFn] = {
    "grounded": grounded_group,
    "support_bin": support_bin_group,
    "value_length": value_length_group,
    "field_type": field_type_group,
    "grounded_x_length": combine_groups(grounded_group, value_length_group),
    "support_x_type": combine_groups(support_bin_group, field_type_group),
}


@dataclass
class GroupPartitionSelector:
    """Pick the provenance taxonomy that maximizes coverage at the target risk.

    Validity-preserving data use: the partition is CHOSEN on a selection split
    and the per-group thresholds are REFIT on a disjoint fit split, so the final
    policy's per-group guarantee still holds on test (the partition is fixed
    w.r.t. the fit split). ``candidates`` defaults to :data:`GROUP_TAXONOMIES`.

    # DECISION: selecting the taxonomy and fitting thresholds on the SAME split
    # would inflate coverage and break the guarantee; the select/fit split keeps
    # it honest (a small price in calibration data for distribution-free validity).
    """

    alpha: float = 0.05
    candidates: dict[str, GroupFn] | None = None
    selected_: str = ""
    policy_: GroupConformalAbstention | None = None
    coverage_: dict[str, float] = field(default_factory=dict)

    def fit(
        self,
        select_preds: Sequence[FieldPrediction],
        select_correct: Sequence[int],
        fit_preds: Sequence[FieldPrediction],
        fit_correct: Sequence[int],
    ) -> GroupPartitionSelector:
        cands = self.candidates or GROUP_TAXONOMIES
        self.coverage_ = {}
        for name, fn in cands.items():
            policy = GroupConformalAbstention(alpha=self.alpha, group_of=fn).fit(
                select_preds, select_correct
            )
            self.coverage_[name] = float(policy.accept(select_preds).mean())
        self.selected_ = max(self.coverage_, key=lambda k: self.coverage_[k])
        self.policy_ = GroupConformalAbstention(
            alpha=self.alpha, group_of=cands[self.selected_]
        ).fit(fit_preds, fit_correct)
        return self

    def accept(self, preds: Sequence[FieldPrediction]) -> np.ndarray:
        if self.policy_ is None:
            raise RuntimeError("GroupPartitionSelector must be fit on cal splits first")
        return self.policy_.accept(preds)
