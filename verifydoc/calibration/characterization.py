"""When does provenance-conditioned conformal beat pooled? (PROJECT.md §5; novel).

Group-conditional (Mondrian) conformal is only worth its finite-sample cost when
the groups actually differ in a way the confidence signal can exploit. This
module makes that condition *measurable on the calibration split alone*, so a
practitioner (and a reviewer) can predict the coverage gain before touching test
data. It reports three diagnostics motivated by the theory:

- **error separation** — the spread of per-group error rates. Per-group
  thresholding is the Neyman--Pearson-optimal response to heterogeneous error
  rates; with zero separation, pooling is already optimal.
- **within-group discriminability** — mean AUROC of confidence vs correctness
  inside each group. Conditioning helps only if, within a group, confidence
  still ranks errors (otherwise the group threshold degenerates).
- **finite-sample penalty** — the smallest group size. Split conformal's +1
  correction means a tiny group cannot certify a tight risk however clean it is.

The headline number, ``predicted_gain``, is an honest cross-validated estimate
of grouped-minus-pooled coverage at the target risk, computed by repeatedly
splitting the calibration set (fit on one half, evaluate coverage on the other).
It never uses the test split, so it is a legitimate a-priori predictor.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from verifydoc.calibration.conformal import ConformalAbstention
from verifydoc.calibration.grouped_conformal import (
    GroupConformalAbstention,
    GroupFn,
    grounded_group,
)
from verifydoc.types import FieldPrediction


@dataclass
class CharacterizationReport:
    """Calibration-split diagnosis of the grouped-vs-pooled coverage gain."""

    predicted_gain: float
    error_separation: float
    within_group_auroc: float
    min_group_size: int
    group_error_rates: dict[str, float]
    recommend: bool

    def summary(self) -> str:
        verdict = "use grouped" if self.recommend else "use pooled"
        return (
            f"predicted coverage gain {self.predicted_gain:+.3f}; "
            f"error separation {self.error_separation:.3f}; "
            f"within-group AUROC {self.within_group_auroc:.3f}; "
            f"min group n={self.min_group_size} -> {verdict}"
        )


def predict_coverage_gain(
    preds: Sequence[FieldPrediction],
    correct: Sequence[int],
    group_of: GroupFn = grounded_group,
    alpha: float = 0.05,
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Cross-validated estimate (on cal data only) of grouped - pooled coverage @ alpha.

    Repeatedly splits the calibration set in half: fits both policies on one
    half and measures the coverage difference on the other. Positive means
    group-conditional conformal is expected to accept more at the same risk.
    """
    items = list(preds)
    y = np.asarray(correct, dtype=float)
    conf = np.array([p.confidence for p in items], dtype=float)
    n = len(items)
    if n != y.size:
        raise ValueError("preds and correct must be the same length")
    rng = np.random.default_rng(seed)
    gains: list[float] = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2 :]
        if a.size < 2 or b.size < 2:
            continue
        grouped = GroupConformalAbstention(alpha=alpha, group_of=group_of).fit(
            [items[i] for i in a], y[a]
        )
        pooled = ConformalAbstention(alpha=alpha).fit(conf[a].tolist(), y[a].tolist())
        g_cov = float(grouped.accept([items[i] for i in b]).mean())
        p_cov = float((conf[b] >= pooled.threshold_).mean())
        gains.append(g_cov - p_cov)
    return float(np.mean(gains)) if gains else 0.0


def characterize(
    preds: Sequence[FieldPrediction],
    correct: Sequence[int],
    group_of: GroupFn = grounded_group,
    alpha: float = 0.05,
    min_gain: float = 0.02,
    seed: int = 0,
) -> CharacterizationReport:
    """Diagnose whether to use grouped conformal, from the calibration split only."""
    items = list(preds)
    y = np.asarray(correct, dtype=float)
    if len(items) != y.size or y.size == 0:
        raise ValueError("preds and correct must be equal-length and non-empty")
    conf = np.array([p.confidence for p in items], dtype=float)
    groups = np.array([group_of(p) for p in items])

    errors: dict[str, float] = {}
    sizes: list[int] = []
    aurocs: list[float] = []
    for g in np.unique(groups):
        mask = groups == g
        sizes.append(int(mask.sum()))
        errors[str(g)] = float(1.0 - y[mask].mean())
        pos = int(y[mask].sum())
        if mask.sum() >= 4 and 0 < pos < int(mask.sum()):
            aurocs.append(float(roc_auc_score(y[mask], conf[mask])))

    separation = (max(errors.values()) - min(errors.values())) if errors else 0.0
    gain = predict_coverage_gain(items, y, group_of, alpha, seed=seed)
    return CharacterizationReport(
        predicted_gain=gain,
        error_separation=float(separation),
        within_group_auroc=float(np.mean(aurocs)) if aurocs else 0.5,
        min_group_size=min(sizes) if sizes else 0,
        group_error_rates=errors,
        recommend=gain >= min_gain,
    )
