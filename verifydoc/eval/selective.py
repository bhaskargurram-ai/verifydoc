"""Selective-prediction / abstention metrics (PROJECT.md §5.C).

Risk-coverage curve, AURC, E-AURC (Geifman & El-Yaniv 2019), Coverage@Risk,
Risk@Coverage, Accuracy@k%, and the error-detection framing (AUROC, AUPR,
FPR@95%TPR with score = 1 - confidence).

Conventions:
- Fields are ranked by confidence descending; ties keep input order (stable
  sort), the standard AURC convention.
- Threshold-style metrics (Coverage@Risk) only cut where the confidence
  strictly drops, so every reported operating point is a real threshold.
- The selective risk of an empty acceptance set is 0.
- Degenerate detection problems (no errors, or no correct fields) make
  AUROC/AUPR/FPR@95%TPR undefined: they return ``nan``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from verifydoc.eval.calibration import _validate


def _sorted_desc(conf: Sequence[float], correct: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    c, y = _validate(conf, correct)
    order = np.argsort(-c, kind="stable")
    return c[order], y[order]


def rc_curve(conf: Sequence[float], correct: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: (coverage_k, selective_risk_k) for k = 1..N."""
    _, y = _sorted_desc(conf, correct)
    k = np.arange(1, y.size + 1)
    coverage = k / y.size
    risk = np.cumsum(1.0 - y) / k
    return coverage, risk


def aurc(conf: Sequence[float], correct: Sequence[int]) -> float:
    """Area under the risk-coverage curve: mean of prefix risks over all k."""
    _, risk = rc_curve(conf, correct)
    return float(risk.mean())


def oracle_aurc(correct: Sequence[int]) -> float:
    """AURC of the oracle ranking (all correct fields before all errors)."""
    y = np.asarray(correct, dtype=float)
    n, n_correct = y.size, y.sum()
    k = np.arange(1, n + 1)
    risk = np.maximum(0.0, k - n_correct) / k
    return float(risk.mean())


def e_aurc(conf: Sequence[float], correct: Sequence[int]) -> float:
    """Excess AURC = AURC - oracle AURC; removes the base-error contribution."""
    return aurc(conf, correct) - oracle_aurc(correct)


def coverage_at_risk(
    conf: Sequence[float], correct: Sequence[int], alpha: float
) -> tuple[float, float]:
    """Max coverage with empirical selective risk <= alpha; returns (coverage, threshold).

    Only threshold-realizable prefixes (cuts where confidence strictly drops)
    are considered. Returns (0.0, inf) when no prefix qualifies.
    """
    c, y = _sorted_desc(conf, correct)
    k = np.arange(1, y.size + 1)
    risk = np.cumsum(1.0 - y) / k
    boundary = np.append(c[:-1] > c[1:], True)
    ok = (risk <= alpha) & boundary
    if not ok.any():
        return 0.0, float("inf")
    i = int(np.flatnonzero(ok)[-1])
    return float((i + 1) / y.size), float(c[i])


def risk_at_coverage(conf: Sequence[float], correct: Sequence[int], coverage: float) -> float:
    """Selective risk when accepting the top ``coverage`` fraction (ceil, ties by order)."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    _, y = _sorted_desc(conf, correct)
    k = max(1, math.ceil(coverage * y.size))
    return float(1.0 - y[:k].mean())


def accuracy_at_k(conf: Sequence[float], correct: Sequence[int], k_frac: float) -> float:
    """Accuracy on the top-k% most confident fields (k_frac in (0, 1])."""
    return 1.0 - risk_at_coverage(conf, correct, k_frac)


# ---------------------------------------------------------------------------
# Error-detection framing: positives = wrong fields, score = 1 - confidence
# ---------------------------------------------------------------------------


def _detection_arrays(
    conf: Sequence[float], correct: Sequence[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    c, y = _validate(conf, correct)
    errors = y == 0.0
    if not errors.any() or errors.all():
        return None
    return 1.0 - c, errors.astype(float)


def auroc(conf: Sequence[float], correct: Sequence[int]) -> float:
    """AUROC of error detection (equals P(score_error > score_correct) + tie credit)."""
    arrays = _detection_arrays(conf, correct)
    if arrays is None:
        return float("nan")
    score, pos = arrays
    order = np.argsort(score, kind="stable")
    ranks = _average_ranks(score[order])[np.argsort(order, kind="stable")]
    n_pos, n_neg = pos.sum(), (1.0 - pos).sum()
    rank_sum = ranks[pos == 1.0].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(sorted_scores: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (input pre-sorted asc)."""
    n = sorted_scores.size
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2 + 1.0
        i = j + 1
    return ranks


def aupr(conf: Sequence[float], correct: Sequence[int]) -> float:
    """Average precision of error detection (step-wise PR interpolation).

    Tie groups are processed atomically so the result is order-invariant.
    """
    arrays = _detection_arrays(conf, correct)
    if arrays is None:
        return float("nan")
    score, pos = arrays
    n_pos = pos.sum()
    ap = 0.0
    tp = fp = 0.0
    for s in np.unique(score)[::-1]:
        group = score == s
        g_tp, g_fp = pos[group].sum(), (1.0 - pos[group]).sum()
        tp, fp = tp + g_tp, fp + g_fp
        if g_tp:
            ap += (g_tp / n_pos) * (tp / (tp + fp))
    return float(ap)


def fpr_at_tpr(conf: Sequence[float], correct: Sequence[int], tpr: float = 0.95) -> float:
    """Min fraction of correct fields flagged while catching >= ``tpr`` of errors."""
    arrays = _detection_arrays(conf, correct)
    if arrays is None:
        return float("nan")
    score, pos = arrays
    n_pos, n_neg = pos.sum(), (1.0 - pos).sum()
    best = 1.0
    for s in np.unique(score):
        flagged = score >= s
        if pos[flagged].sum() / n_pos >= tpr:
            best = min(best, float((1.0 - pos)[flagged].sum() / n_neg))
    return best
