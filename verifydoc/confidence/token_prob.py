"""Token/sequence-probability confidence (requires logit access).

Adapters that see logits stash per-field token log-probs in
``FieldPrediction.meta["token_logprobs"]``; this signal turns them into a
confidence: geometric-mean token probability (``mean``), worst token (``min``),
or full sequence probability (``prod``).
"""

from __future__ import annotations

import math
from typing import Literal

from verifydoc.types import FieldPrediction

Aggregate = Literal["mean", "min", "prod"]

META_KEY = "token_logprobs"


def token_prob_confidence(pred: FieldPrediction, aggregate: Aggregate = "mean") -> float | None:
    """Confidence from token log-probs; ``None`` when the adapter gave none."""
    logprobs = pred.meta.get(META_KEY)
    if not logprobs:
        return None
    if aggregate == "mean":
        value = math.exp(sum(logprobs) / len(logprobs))
    elif aggregate == "min":
        value = math.exp(min(logprobs))
    else:
        value = math.exp(sum(logprobs))
    return min(1.0, max(0.0, value))


def apply_token_prob(
    predictions: list[FieldPrediction], aggregate: Aggregate = "mean"
) -> list[FieldPrediction]:
    """Return copies with ``confidence`` set from token log-probs where available."""
    out = []
    for pred in predictions:
        conf = token_prob_confidence(pred, aggregate)
        out.append(pred if conf is None else pred.model_copy(update={"confidence": conf}))
    return out
