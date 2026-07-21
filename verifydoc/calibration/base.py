"""Calibrator interface: fit on the calibration split, transform anywhere else.

Golden rule #4: calibration parameters are NEVER tuned on test data. Fit takes
the calibration split's (confidence, correctness) pairs; transform maps raw
confidences to calibrated ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

_EPS = 1e-7


class Calibrator(ABC):
    """Post-hoc mapping from raw confidence to calibrated probability."""

    _fitted: bool = False

    def fit(self, conf: Sequence[float], correct: Sequence[int]) -> Calibrator:
        c = np.asarray(conf, dtype=float)
        y = np.asarray(correct, dtype=float)
        if c.shape != y.shape or c.ndim != 1 or c.size == 0:
            raise ValueError("conf and correct must be 1-D, equal-length, non-empty")
        if ((c < 0) | (c > 1)).any():
            raise ValueError("confidences must lie in [0, 1]")
        self._fit(c, y)
        self._fitted = True
        return self

    def transform(self, conf: Sequence[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit on the calibration split first")
        c = np.asarray(conf, dtype=float)
        return np.clip(self._transform(c), 0.0, 1.0)

    @abstractmethod
    def _fit(self, conf: np.ndarray, correct: np.ndarray) -> None: ...

    @abstractmethod
    def _transform(self, conf: np.ndarray) -> np.ndarray: ...


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds with clipping so 0/1 confidences stay finite."""
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
