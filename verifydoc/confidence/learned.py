"""Learned signal combiner: logistic regression over the raw signals.

The paper's ablation partner to the transparent weighted mean in
``combined.py``: fit on the CALIBRATION split's (signal-vector, correct)
pairs, predict a fused confidence anywhere else.

# DECISION (feature encoding, pinned by tests): each signal contributes two
# features — its value (0.5 where absent) and a 0/1 missing indicator — so
# "token-probs unavailable" is information, not silently-average data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

SIGNAL_ORDER = ("consensus", "grounding", "token_prob", "verbalized")


def _featurize(signals: Sequence[Mapping[str, float | None]]) -> np.ndarray:
    rows = []
    for sig in signals:
        row: list[float] = []
        for name in SIGNAL_ORDER:
            value = sig.get(name)
            row.append(0.5 if value is None else float(value))
            row.append(1.0 if value is None else 0.0)
        rows.append(row)
    return np.asarray(rows, dtype=float)


class LearnedCombiner:
    """Logistic fusion of confidence signals, fit on the calibration split."""

    def __init__(self, c: float = 1.0) -> None:
        self._model = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
        self._fitted = False
        self._constant: float | None = None

    def fit(
        self,
        signals: Sequence[Mapping[str, float | None]],
        correct: Sequence[int],
    ) -> LearnedCombiner:
        x = _featurize(signals)
        y = np.asarray(correct, dtype=float)
        if x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("signals and correct must be non-empty and aligned")
        if len(np.unique(y)) < 2:
            self._constant = float(y.mean())  # degenerate split: predict base rate
        else:
            self._model.fit(x, y)
        self._fitted = True
        return self

    def predict(self, signals: Sequence[Mapping[str, float | None]]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LearnedCombiner must be fit on the calibration split first")
        if self._constant is not None:
            return np.full(len(signals), self._constant)
        return np.asarray(self._model.predict_proba(_featurize(signals))[:, 1], dtype=float)
