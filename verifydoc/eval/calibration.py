"""Calibration metrics (PROJECT.md §5.B).

Implements ECE (15 equal-width bins by default), Adaptive ECE (equal-mass
bins), MCE, Brier score, NLL, TCE (target calibration error at abstention
operating points), and reliability-diagram data (Guo et al. 2017 conventions).

All functions take a confidence vector ``conf`` in [0, 1] and a 0/1
correctness vector ``correct`` — the (c_i, correct_i) pairs produced by
``eval.extraction.score_fields``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _validate(conf: Sequence[float], correct: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    if c.shape != y.shape or c.ndim != 1:
        raise ValueError("conf and correct must be 1-D and the same length")
    if c.size == 0:
        raise ValueError("empty inputs")
    if ((c < 0) | (c > 1)).any():
        raise ValueError("confidences must lie in [0, 1]")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("correct must be 0/1")
    return c, y


@dataclass
class ReliabilityBin:
    """One bin of a reliability diagram."""

    lo: float
    hi: float
    mean_confidence: float
    accuracy: float
    count: int


def reliability_bins(
    conf: Sequence[float], correct: Sequence[int], n_bins: int = 15
) -> list[ReliabilityBin]:
    """Equal-width reliability-diagram bins over [0, 1] (empty bins skipped).

    Bin edges follow Guo et al.: bin m covers ((m-1)/M, m/M]; conf == 0 falls
    into the first bin.
    """
    c, y = _validate(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for m in range(n_bins):
        mask = idx == m
        if not mask.any():
            continue
        bins.append(
            ReliabilityBin(
                lo=float(edges[m]),
                hi=float(edges[m + 1]),
                mean_confidence=float(c[mask].mean()),
                accuracy=float(y[mask].mean()),
                count=int(mask.sum()),
            )
        )
    return bins


def ece(conf: Sequence[float], correct: Sequence[int], n_bins: int = 15) -> float:
    """Expected Calibration Error: sum_m (|B_m|/N) * |acc(B_m) - conf(B_m)|."""
    n = len(conf)
    return sum(
        (b.count / n) * abs(b.accuracy - b.mean_confidence)
        for b in reliability_bins(conf, correct, n_bins)
    )


def smooth_ece(
    conf: Sequence[float], correct: Sequence[int], bandwidth: float | None = None
) -> float:
    """Kernel-smoothed calibration error (SmoothECE-style; Błasiok & Nakkiran, ICLR 2024).

    Binned ECE is discontinuous and sensitive to bin count; a Gaussian-kernel
    regression of (correct - conf) against conf gives a continuous, bin-free
    estimate: SmoothECE = sum_i w_i |rhat(c_i)| / sum_i w_i over sample points,
    where rhat is the kernel-weighted mean residual. Bandwidth defaults to a
    Silverman-style ``0.15 * n**(-1/5)`` heuristic (their consistent estimator
    selects it automatically; this is a fixed-heuristic variant).
    """
    c, y = _validate(conf, correct)
    n = c.size
    h = bandwidth if bandwidth is not None else max(1e-3, 0.15 * n ** (-0.2))
    resid = y - c
    # weight matrix would be O(n^2); cap for large n by evaluating at the points
    diffs = (c[:, None] - c[None, :]) / h
    w = np.exp(-0.5 * diffs**2)
    rhat = (w * resid[None, :]).sum(axis=1) / w.sum(axis=1)
    return float(np.mean(np.abs(rhat)))


def adaptive_ece(conf: Sequence[float], correct: Sequence[int], n_bins: int = 15) -> float:
    """Adaptive ECE: equal-mass bins (robust when confidences cluster)."""
    c, y = _validate(conf, correct)
    order = np.argsort(c, kind="stable")
    c, y = c[order], y[order]
    splits = np.array_split(np.arange(c.size), n_bins)
    total = 0.0
    for chunk in splits:
        if chunk.size == 0:
            continue
        total += (chunk.size / c.size) * abs(float(y[chunk].mean()) - float(c[chunk].mean()))
    return total


def mce(conf: Sequence[float], correct: Sequence[int], n_bins: int = 15) -> float:
    """Maximum Calibration Error: worst-bin |acc - conf| over non-empty bins."""
    return max(abs(b.accuracy - b.mean_confidence) for b in reliability_bins(conf, correct, n_bins))


def brier(conf: Sequence[float], correct: Sequence[int]) -> float:
    """Brier score: mean squared error between confidence and correctness."""
    c, y = _validate(conf, correct)
    return float(np.mean((c - y) ** 2))


def nll(conf: Sequence[float], correct: Sequence[int], eps: float = 1e-12) -> float:
    """Negative log-likelihood; confidences clipped to [eps, 1-eps]."""
    c, y = _validate(conf, correct)
    c = np.clip(c, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(c) + (1.0 - y) * np.log(1.0 - c)))


def tce(
    conf: Sequence[float],
    correct: Sequence[int],
    alphas: Sequence[float] = (0.01, 0.02, 0.05, 0.10),
    tune_conf: Sequence[float] | None = None,
) -> float:
    """Target Calibration Error: mean over alpha of |achieved risk - alpha|.

    For each target risk ``alpha``, the acceptance threshold is chosen from the
    confidences alone: accept the largest confidence-sorted prefix whose
    *predicted* risk (mean of 1 - c over accepted) is <= alpha, cutting only at
    confidence boundaries so the prefix corresponds to a real threshold.
    The achieved selective risk on (conf, correct) is then compared to alpha.

    # DECISION: the threshold is tuned from predicted risk (no labels), which
    # is exactly how a deployed system must pick its operating point; pass
    # ``tune_conf`` to select the threshold on a calibration split instead.
    # Accepting nothing has selective risk 0 by convention.
    """
    c, y = _validate(conf, correct)
    t = np.asarray(tune_conf, dtype=float) if tune_conf is not None else c
    total = 0.0
    for alpha in alphas:
        threshold = _risk_threshold(t, float(alpha))
        accepted = c >= threshold
        achieved = float(1.0 - y[accepted].mean()) if accepted.any() else 0.0
        total += abs(achieved - alpha)
    return total / len(alphas)


def _risk_threshold(conf: np.ndarray, alpha: float) -> float:
    """Lowest confidence threshold whose predicted selective risk is <= alpha."""
    c = np.sort(conf)[::-1]
    predicted_risk = np.cumsum(1.0 - c) / np.arange(1, c.size + 1)
    # valid cut points: where the next confidence is strictly lower (or the end)
    boundary: np.ndarray = np.append(c[:-1] > c[1:], True)
    ok = (predicted_risk <= alpha) & boundary
    if not ok.any():
        return float("inf")  # accept nothing
    k = int(np.flatnonzero(ok)[-1])
    return float(c[k])
