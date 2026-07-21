"""Isotonic regression calibration (Zadrozny & Elkan 2002).

Non-parametric monotone mapping from confidence to empirical accuracy;
the strongest of the classic calibrators given enough calibration data.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from verifydoc.calibration.base import Calibrator


class IsotonicCalibrator(Calibrator):
    _model: IsotonicRegression

    def _fit(self, conf: np.ndarray, correct: np.ndarray) -> None:
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._model.fit(conf, correct)

    def _transform(self, conf: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(conf), dtype=float)
