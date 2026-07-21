"""Split management: the never-tune-on-test rule, enforced in code.

Golden rule #4: calibration and abstention thresholds are fit only on a
dedicated calibration split. These helpers create and verify disjoint splits
by item id; every harness that fits a calibrator must call
``assert_disjoint`` before scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def split_calibration(
    ids: Sequence[T], calibration_fraction: float = 0.5, seed: int = 0
) -> tuple[list[T], list[T]]:
    """Deterministically split ids into (calibration, test); disjoint and complete."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    unique = list(dict.fromkeys(ids))
    if len(unique) < 2:
        raise ValueError("need at least two distinct ids to split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_cal = max(1, min(len(unique) - 1, round(calibration_fraction * len(unique))))
    cal = [unique[i] for i in sorted(order[:n_cal])]
    test = [unique[i] for i in sorted(order[n_cal:])]
    return cal, test


def assert_disjoint(calibration_ids: Sequence[T], test_ids: Sequence[T]) -> None:
    """Raise if any id appears in both splits (tuning-on-test guard)."""
    overlap = set(calibration_ids) & set(test_ids)
    if overlap:
        sample = sorted(str(x) for x in overlap)[:5]
        raise ValueError(
            f"calibration and test splits overlap on {len(overlap)} ids "
            f"(e.g. {sample}); calibration must never see test data"
        )
