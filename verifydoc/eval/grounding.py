"""Grounding / provenance metrics (PROJECT.md §5.D).

Currently: the IoU primitive. Box-grounding accuracy @ IoU, mean IoU,
span-grounding F1, and grounding-conditioned correctness land with the
grounding stage.
"""

from __future__ import annotations

Box = tuple[float, float, float, float]


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
