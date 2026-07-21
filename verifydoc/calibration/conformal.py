"""Split conformal abstention / conformal risk control (PROJECT.md §5.G).

Chooses the acceptance threshold on the calibration split so that the
*expected* selective risk of accepted fields is bounded by the target alpha
under exchangeability (conformal risk control with the bounded miscoverage
loss; Angelopoulos et al.; conformal factuality, Mohri & Hashimoto 2024).

Threshold rule: the smallest confidence t such that on the calibration split

    (#errors accepted at t + 1) / (#accepted at t + 1) <= alpha

The +1 terms are the finite-sample correction (the future test point counted
as a worst-case error). Per arXiv:2606.29054, meaningful certification can
force heavy abstention — so this class also reports the abstention it forces
(``abstention_rate_`` on the calibration split), and the benchmark reports
both numbers side by side.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class ConformalAbstention:
    """Distribution-free accept/review policy at a target selective risk."""

    threshold_: float
    abstention_rate_: float
    alpha: float

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._fitted = False

    def fit(self, conf: Sequence[float], correct: Sequence[int]) -> ConformalAbstention:
        """Fit the threshold on the CALIBRATION split (never test)."""
        c = np.asarray(conf, dtype=float)
        y = np.asarray(correct, dtype=float)
        if c.shape != y.shape or c.ndim != 1 or c.size == 0:
            raise ValueError("conf and correct must be 1-D, equal-length, non-empty")

        order = np.argsort(-c, kind="stable")
        c_sorted, y_sorted = c[order], y[order]
        n_accepted = np.arange(1, c.size + 1)
        n_errors = np.cumsum(1.0 - y_sorted)
        bound = (n_errors + 1.0) / (n_accepted + 1.0)
        boundary = np.append(c_sorted[:-1] > c_sorted[1:], True)
        ok = (bound <= self.alpha) & boundary
        if ok.any():
            k = int(np.flatnonzero(ok)[-1])
            self.threshold_ = float(c_sorted[k])
            self.abstention_rate_ = float(1.0 - (k + 1) / c.size)
        else:
            self.threshold_ = float("inf")  # certification impossible: review everything
            self.abstention_rate_ = 1.0
        self._fitted = True
        return self

    def accept(self, conf: Sequence[float]) -> np.ndarray:
        """Boolean accept mask for new confidences (True = auto-accept)."""
        if not self._fitted:
            raise RuntimeError("ConformalAbstention must be fit on the calibration split first")
        return np.asarray(conf, dtype=float) >= self.threshold_

    @property
    def guarantee(self) -> str:
        return (
            f"E[selective risk] <= {self.alpha:.3f} under exchangeability "
            f"(split conformal, finite-sample corrected); forced abstention on "
            f"calibration split: {self.abstention_rate_:.1%}"
        )
