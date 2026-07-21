"""Combined confidence: fuse the available signals for one field.

# DECISION: v0.1 uses a transparent weighted mean over whichever signals are
# present (weights renormalized to the available subset). A learned combiner
# (logistic regression on the calibration split) can replace this behind the
# same function without touching other stages; the paper compares both.
"""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_WEIGHTS: dict[str, float] = {
    "consensus": 0.5,
    "grounding": 0.3,
    "token_prob": 0.1,
    "verbalized": 0.1,
}


def combined_confidence(
    signals: Mapping[str, float | None],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Weighted mean of available signals; unknown signal names are rejected."""
    w = dict(weights) if weights is not None else DEFAULT_WEIGHTS
    unknown = set(signals) - set(w)
    if unknown:
        raise ValueError(f"unknown signals: {sorted(unknown)}")
    available = {name: v for name, v in signals.items() if v is not None}
    if not available:
        raise ValueError("no signals available")
    total_weight = sum(w[name] for name in available)
    if total_weight <= 0:
        raise ValueError("available signals have zero total weight")
    value = sum(w[name] * v for name, v in available.items()) / total_weight
    return min(1.0, max(0.0, value))
