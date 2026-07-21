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
