"""Grounding-based confidence: provenance quality as a trust signal.

The hypothesis (risk-controlled generative OCR, arXiv:2603.19790): values that
cannot be traced to a tight source region are more likely hallucinated. The
signal is the grounding's ``support`` score (e.g. string-match/IoU quality
computed by the grounder); a field with no grounding at all gets
``ungrounded_confidence`` (default 0).
"""

from __future__ import annotations

from verifydoc.types import FieldPrediction


def grounding_confidence(pred: FieldPrediction, ungrounded_confidence: float = 0.0) -> float:
    """Confidence from provenance: the grounding's support, else the floor."""
    if pred.grounding is None:
        return ungrounded_confidence
    return pred.grounding.support


def apply_grounding_confidence(
    predictions: list[FieldPrediction], ungrounded_confidence: float = 0.0
) -> list[FieldPrediction]:
    """Return copies with ``confidence`` set from grounding support."""
    return [
        pred.model_copy(update={"confidence": grounding_confidence(pred, ungrounded_confidence)})
        for pred in predictions
    ]
