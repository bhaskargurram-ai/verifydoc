"""Grounding / provenance metrics (PROJECT.md §5.D).

Box grounding accuracy @ IoU tau, mean IoU, span-grounding F1, and
grounding-conditioned correctness (the "ungrounded values are more likely
wrong" hypothesis test, after risk-controlled generative OCR).

Conventions:
- A field with a gold box but no predicted box scores IoU 0 (a grounding miss).
- Fields without a gold box are excluded from box metrics entirely.
- ``mean_iou`` averages over fields where *both* boxes exist (secondary metric).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Box = tuple[float, float, float, float]
Span = tuple[int, int]


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``(x0, y0, x1, y1)`` boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def pairwise_ious(
    pred_boxes: Sequence[Box | None], gold_boxes: Sequence[Box | None]
) -> list[float | None]:
    """Per-field IoU; ``None`` where there is no gold box (field excluded)."""
    if len(pred_boxes) != len(gold_boxes):
        raise ValueError("pred and gold box lists must be the same length")
    out: list[float | None] = []
    for p, g in zip(pred_boxes, gold_boxes):
        if g is None:
            out.append(None)
        elif p is None:
            out.append(0.0)
        else:
            out.append(iou(p, g))
    return out


def box_grounding_accuracy(
    pred_boxes: Sequence[Box | None], gold_boxes: Sequence[Box | None], tau: float = 0.5
) -> float:
    """Fraction of gold-boxed fields whose predicted box has IoU >= tau."""
    ious = [v for v in pairwise_ious(pred_boxes, gold_boxes) if v is not None]
    if not ious:
        return 0.0
    return sum(v >= tau for v in ious) / len(ious)


def mean_iou(pred_boxes: Sequence[Box | None], gold_boxes: Sequence[Box | None]) -> float:
    """Mean IoU over fields with both a predicted and a gold box."""
    vals = [iou(p, g) for p, g in zip(pred_boxes, gold_boxes) if p is not None and g is not None]
    if len(pred_boxes) != len(gold_boxes):
        raise ValueError("pred and gold box lists must be the same length")
    return sum(vals) / len(vals) if vals else 0.0


def span_f1(pred: Span | None, gold: Span | None) -> float:
    """Character-overlap F1 between two ``[start, end)`` spans."""
    if pred is None or gold is None:
        return 0.0
    overlap = max(0, min(pred[1], gold[1]) - max(pred[0], gold[0]))
    len_p, len_g = pred[1] - pred[0], gold[1] - gold[0]
    if overlap == 0 or len_p == 0 or len_g == 0:
        return 0.0
    precision, recall = overlap / len_p, overlap / len_g
    return 2 * precision * recall / (precision + recall)


def span_grounding_f1(
    pred_spans: Sequence[Span | None], gold_spans: Sequence[Span | None]
) -> float:
    """Macro-averaged span F1 over fields that have a gold span."""
    if len(pred_spans) != len(gold_spans):
        raise ValueError("pred and gold span lists must be the same length")
    scored = [span_f1(p, g) for p, g in zip(pred_spans, gold_spans) if g is not None]
    return sum(scored) / len(scored) if scored else 0.0


@dataclass
class GroundingConditionedReport:
    """Accuracy split by grounding quality (IoU >= tau vs below)."""

    accuracy_grounded: float
    accuracy_ungrounded: float
    n_grounded: int
    n_ungrounded: int

    @property
    def gap(self) -> float:
        """Positive gap = well-grounded fields are more often correct."""
        return self.accuracy_grounded - self.accuracy_ungrounded


def grounding_conditioned_correctness(
    correct: Sequence[int],
    pred_boxes: Sequence[Box | None],
    gold_boxes: Sequence[Box | None],
    tau: float = 0.5,
) -> GroundingConditionedReport:
    """Accuracy of fields whose grounding is correct vs incorrect at IoU tau."""
    if len(correct) != len(pred_boxes):
        raise ValueError("correct and box lists must be the same length")
    ious = pairwise_ious(pred_boxes, gold_boxes)
    grounded = [y for y, v in zip(correct, ious) if v is not None and v >= tau]
    ungrounded = [y for y, v in zip(correct, ious) if v is not None and v < tau]
    return GroundingConditionedReport(
        accuracy_grounded=sum(grounded) / len(grounded) if grounded else 0.0,
        accuracy_ungrounded=sum(ungrounded) / len(ungrounded) if ungrounded else 0.0,
        n_grounded=len(grounded),
        n_ungrounded=len(ungrounded),
    )
