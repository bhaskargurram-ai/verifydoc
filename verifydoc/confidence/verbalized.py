"""Verbalized confidence: the extractor rates its own answer (0-1).

Adapters that prompt the model for a per-field self-assessment stash it in
``FieldPrediction.meta["verbalized_confidence"]``. Known failure mode
(paper §5.G): verbalized scores can be inflated by RLHF — always compare
against consensus and report calibration, never trust raw.
"""

from __future__ import annotations

from verifydoc.types import FieldPrediction

META_KEY = "verbalized_confidence"


def verbalized_confidence(pred: FieldPrediction) -> float | None:
    """Clamped self-reported confidence; ``None`` when the adapter gave none."""
    value = pred.meta.get(META_KEY)
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


def apply_verbalized(predictions: list[FieldPrediction]) -> list[FieldPrediction]:
    """Return copies with ``confidence`` set from verbalized scores where available."""
    out = []
    for pred in predictions:
        conf = verbalized_confidence(pred)
        out.append(pred if conf is None else pred.model_copy(update={"confidence": conf}))
    return out
