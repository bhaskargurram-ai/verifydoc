"""Statistical rigor utilities (PROJECT.md §5.H).

Bootstrap 95% confidence intervals (percentile method, field-level resampling)
and paired significance tests (sign-flip permutation and paired bootstrap) for
model-vs-model claims. All routines are seeded and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Statistic = Callable[..., float]


@dataclass
class BootstrapResult:
    point: float
    lo: float
    hi: float
    ci: float
    n_boot: int


def bootstrap_ci(
    statistic: Statistic,
    *arrays: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile bootstrap CI for ``statistic(*arrays)``.

    Arrays are resampled *jointly* (same indices per replicate), so paired
    structures like (confidence, correctness) stay aligned — resampling fields,
    per §5.H.
    """
    cols = [np.asarray(a) for a in arrays]
    n = cols[0].shape[0]
    if any(c.shape[0] != n for c in cols):
        raise ValueError("all arrays must share the first dimension")
    if n == 0:
        raise ValueError("empty inputs")
    rng = np.random.default_rng(seed)
    point = float(statistic(*cols))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(*(c[idx] for c in cols))
    tail = (1.0 - ci) / 2.0
    quantiles: np.ndarray = np.quantile(stats, [tail, 1.0 - tail])
    return BootstrapResult(
        point=point, lo=float(quantiles[0]), hi=float(quantiles[1]), ci=ci, n_boot=n_boot
    )


def paired_permutation_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided sign-flip permutation p-value for mean(a) != mean(b).

    ``scores_a`` and ``scores_b`` are per-field scores of two systems on the
    *same* fields (e.g. 0/1 correctness). Add-one smoothing keeps p > 0.
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("scores must be 1-D, equal-length, non-empty")
    diff = a - b
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((1 + (permuted >= observed - 1e-12).sum()) / (1 + n_perm))


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired-bootstrap p-value: does the sign of mean(a-b) hold up?"""
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("scores must be 1-D, equal-length, non-empty")
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    p_le = (1 + (means <= 0).sum()) / (1 + n_boot)
    p_ge = (1 + (means >= 0).sum()) / (1 + n_boot)
    return float(min(1.0, 2.0 * min(p_le, p_ge)))
