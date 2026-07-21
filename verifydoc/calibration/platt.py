"""Platt scaling: logistic regression a*logit(p) + b (Platt 1999).

Unlike temperature scaling it can also shift systematically biased scores,
not just soften them.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from verifydoc.calibration.base import Calibrator, logit, sigmoid


class PlattScaling(Calibrator):
    a_: float
    b_: float

    def _fit(self, conf: np.ndarray, correct: np.ndarray) -> None:
        if len(np.unique(correct)) < 2:
            # degenerate split (all right or all wrong): identity mapping
            self.a_, self.b_ = 1.0, 0.0
            return
        model = LogisticRegression(C=1e6, solver="lbfgs")
        model.fit(logit(conf).reshape(-1, 1), correct)
        self.a_ = float(model.coef_[0][0])
        self.b_ = float(model.intercept_[0])

    def _transform(self, conf: np.ndarray) -> np.ndarray:
        return sigmoid(self.a_ * logit(conf) + self.b_)
