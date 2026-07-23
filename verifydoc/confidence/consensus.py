"""Consensus (self-consistency) confidence — the black-box VerifyDoc default.

Run the extractor k times (or run m different extractors) and use agreement as
confidence: no logits needed, works for any adapter.

# DECISION (consensus semantics, pinned by tests):
# - Votes are normalized values (whitespace-collapsed, casefolded); a sample
#   that omits the field (or predicts None) casts an explicit "omit" vote.
# - The consensus value is the modal vote's first-seen raw value; confidence =
#   modal vote count / number of samples. Ties break by first appearance.
# - If the modal vote is "omit", the consensus prediction has value None with
#   confidence = omit fraction (the abstention layer treats it as omitted).
# - Grounding is taken from the first sample that cast the modal vote.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from verifydoc.eval.extraction import normalize_text
from verifydoc.types import FieldPrediction

_OMIT = "<omit>"


def _vote(value: Any) -> str:
    return _OMIT if value is None else normalize_text(value).casefold()


def consensus(samples: list[list[FieldPrediction]]) -> list[FieldPrediction]:
    """Aggregate k sample runs into one prediction per field path."""
    if not samples:
        raise ValueError("need at least one sample run")
    k = len(samples)

    paths: list[str] = []
    seen: set[str] = set()
    for run in samples:
        for sample_pred in run:
            if sample_pred.path not in seen:
                seen.add(sample_pred.path)
                paths.append(sample_pred.path)

    out: list[FieldPrediction] = []
    for path in paths:
        votes: list[str] = []
        first_pred: dict[str, FieldPrediction] = {}
        for run in samples:
            by_path = {p.path: p for p in run}
            pred = by_path.get(path)
            vote = _vote(pred.value) if pred is not None else _OMIT
            votes.append(vote)
            if vote not in first_pred and pred is not None:
                first_pred[vote] = pred
        winner, count = Counter(votes).most_common(1)[0]
        agreement = count / k
        source = first_pred.get(winner)
        if winner == _OMIT or source is None:
            out.append(FieldPrediction(path=path, value=None, confidence=agreement))
        else:
            out.append(
                FieldPrediction(
                    path=path,
                    value=source.value,
                    confidence=agreement,
                    grounding=source.grounding,
                    meta=dict(source.meta),
                )
            )
    return out


def adaptive_consensus(
    sampler: Callable[[], list[FieldPrediction]],
    *,
    threshold: float = 0.5,
    margin: float = 0.25,
    k_min: int = 2,
    k_max: int = 5,
    budget: int | None = None,
) -> tuple[list[FieldPrediction], int]:
    """Budget-aware consensus: draw more samples *only* while the aggregate is
    still ambiguous near the accept/review boundary.

    Draw ``k_min`` samples, then keep drawing (up to ``k_max``, and any hard
    ``budget`` on total draws) while some field's consensus confidence lands
    within ``margin`` of ``threshold``. Returns ``(consensus_predictions,
    n_samples_drawn)`` so callers can account for the cost saved.

    # DECISION (adaptive-k, pinned by tests):
    # - A black-box extractor can only be resampled *whole* — we can't resample
    #   one field — so adaptivity is early-stopping: as soon as every field is a
    #   clear accept/reject, we stop, spending extra draws only on genuinely
    #   ambiguous documents. ``k_min`` defaults to 2 because a single sample
    #   makes every field unanimous (confidence 1.0) and hides disagreement.
    # - Single-pass fast path: ``k_max=1`` (or ``budget=1``) draws exactly once
    #   and returns immediately — no consensus overhead for latency-bound calls.
    # - ``budget`` is a hard ceiling on sampler calls and overrides both bounds;
    #   if it is below ``k_min`` we draw only ``budget`` times.
    """
    if k_min < 1:
        raise ValueError("k_min must be >= 1")
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    hi = k_max if budget is None else min(k_max, budget)
    lo = min(k_min, hi)

    samples: list[list[FieldPrediction]] = [sampler() for _ in range(lo)]
    while len(samples) < hi:
        if not _has_boundary_field(consensus(samples), threshold, margin):
            break
        samples.append(sampler())
    return consensus(samples), len(samples)


def _has_boundary_field(preds: list[FieldPrediction], threshold: float, margin: float) -> bool:
    """True if any field's confidence sits within ``margin`` of ``threshold``."""
    return any(abs(p.confidence - threshold) < margin for p in preds)
