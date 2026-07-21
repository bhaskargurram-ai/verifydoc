"""Temperature scaling (Guo et al. 2017) on the confidence's log-odds.

Calibrated p' = sigmoid(logit(p) / T); the single scalar T is chosen to
minimize NLL on the calibration split. T > 1 softens overconfident scores.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from verifydoc.calibration.base import Calibrator, logit, sigmoid


class TemperatureScaling(Calibrator):
    temperature_: float

    def _fit(self, conf: np.ndarray, correct: np.ndarray) -> None:
        z = logit(conf)

        def nll_at(log_t: float) -> float:
            p = sigmoid(z / np.exp(log_t))
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            return float(-np.mean(correct * np.log(p) + (1.0 - correct) * np.log(1.0 - p)))

        result = minimize_scalar(nll_at, bounds=(-4.0, 4.0), method="bounded")
        self.temperature_ = float(np.exp(result.x))

    def _transform(self, conf: np.ndarray) -> np.ndarray:
        return sigmoid(logit(conf) / self.temperature_)
