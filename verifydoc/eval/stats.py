"""Statistical rigor utilities (PROJECT.md §5.H).

Bootstrap 95% confidence intervals (percentile method, field-level resampling),
paired significance tests (sign-flip permutation and paired bootstrap) for
model-vs-model claims, and inter-annotator agreement (Cohen's / Fleiss' kappa)
for labeling reliability. All routines are seeded and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

Statistic = Callable[..., float]


def cohens_kappa(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """Cohen's kappa: chance-corrected agreement between two annotators.

    kappa = (p_o - p_e) / (1 - p_e), where p_o is observed agreement and p_e is
    the agreement expected by chance from each annotator's marginal label rates.
    Returns 1.0 for perfect agreement, 0 for chance-level, <0 for worse. If both
    annotators use a single constant label (p_e == 1), returns 1.0 iff they
    agree everywhere.
    """
    a, b = list(labels_a), list(labels_b)
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be equal-length and non-empty")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(rating_counts: Sequence[Sequence[int]]) -> float:
    """Fleiss' kappa for >=2 annotators. ``rating_counts[i][k]`` = number of
    annotators who assigned category k to item i (every row sums to the same
    n_annotators). Chance-corrected agreement across all raters.
    """
    m = np.asarray(rating_counts, dtype=float)
    if m.ndim != 2 or m.shape[0] == 0:
        raise ValueError("rating_counts must be a non-empty items x categories matrix")
    n_raters = m.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]) or n_raters[0] < 2:
        raise ValueError("every item must be rated by the same number (>=2) of annotators")
    n = float(n_raters[0])
    p_i = (np.square(m).sum(axis=1) - n) / (n * (n - 1.0))  # per-item agreement
    p_bar = float(p_i.mean())
    p_j = m.sum(axis=0) / (m.shape[0] * n)  # category marginals
    p_e = float(np.square(p_j).sum())
    if p_e >= 1.0:
        return 1.0 if p_bar >= 1.0 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


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


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm--Bonferroni step-down multiple-testing correction.

    Given ``m`` p-values from a family of comparisons, returns ``reject[i]``
    (aligned to input order) controlling the family-wise error rate at ``alpha``.
    Sort ascending; reject the k-th smallest iff it and all smaller ones satisfy
    ``p <= alpha / (m - rank)``; stop at the first failure (step-down). Uniformly
    more powerful than plain Bonferroni while keeping the same FWER guarantee.
    """
    p = [float(v) for v in pvalues]
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject
