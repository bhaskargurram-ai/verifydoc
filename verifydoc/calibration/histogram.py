"""Histogram binning (Zadrozny & Elkan 2001): equal-width bins over [0, 1].

Each bin's calibrated confidence is its empirical accuracy on the calibration
split; empty bins fall back to the global accuracy.
"""

from __future__ import annotations

import numpy as np

from verifydoc.calibration.base import Calibrator


class HistogramBinning(Calibrator):
    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins

    bin_accuracy_: np.ndarray

    def _fit(self, conf: np.ndarray, correct: np.ndarray) -> None:
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, self.n_bins - 1)
        global_acc = float(correct.mean())
        acc = np.full(self.n_bins, global_acc)
        for m in range(self.n_bins):
            mask = idx == m
            if mask.any():
                acc[m] = float(correct[mask].mean())
        self.bin_accuracy_ = acc

    def _transform(self, conf: np.ndarray) -> np.ndarray:
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, self.n_bins - 1)
        return self.bin_accuracy_[idx]
