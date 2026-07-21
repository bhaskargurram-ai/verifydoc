"""The public entrypoint: wire ingest -> adapter -> confidence -> calibration
-> grounding -> policy into ``verify()``.

Stages stay independent (golden rule #2); this module only sequences them.

# DECISION (default confidence): with k > 1 samples the consensus agreement is
# the primary signal (black-box, works for any adapter), fused with grounding
# support and any adapter-provided token-prob/verbalized signals via the
# transparent combined weighting. With k = 1 the consensus term is absent and
# the remaining signals renormalize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.adapters.text_search import TextSearchAdapter
from verifydoc.calibration.base import Calibrator
from verifydoc.confidence import (
    combined_confidence,
    consensus,
    grounding_confidence,
    token_prob_confidence,
    verbalized_confidence,
)
from verifydoc.grounding import ground_predictions
from verifydoc.ingest import ingest_path
from verifydoc.policy import apply_policy
from verifydoc.types import Document, FieldPrediction, Schema, unflatten_json

DEFAULT_THRESHOLD = 0.8


@dataclass
class VerifiedResult:
    """Verified extraction: every field carries the reliability contract."""

    doc_id: str
    fields: list[FieldPrediction] = field(default_factory=list)
    threshold: float = DEFAULT_THRESHOLD

    @property
    def n_accepted(self) -> int:
        return sum(f.decision == "accept" for f in self.fields)

    @property
    def n_review(self) -> int:
        return sum(f.decision == "review" for f in self.fields)

    def to_dict(self) -> dict[str, Any]:
        """Nested JSON where every leaf is {value, confidence, grounding, decision}."""
        flat = {
            f.path: {
                "value": f.value,
                "confidence": round(f.confidence, 4),
                "grounding": f.grounding.model_dump() if f.grounding else None,
                "decision": f.decision,
            }
            for f in self.fields
        }
        nested: dict[str, Any] = unflatten_json(flat) if flat else {}
        return {
            "doc_id": self.doc_id,
            "threshold": self.threshold,
            "n_accepted": self.n_accepted,
            "n_review": self.n_review,
            "fields": nested,
        }

    def values(self) -> Any:
        """Plain nested JSON of just the extracted values."""
        return unflatten_json({f.path: f.value for f in self.fields}) if self.fields else {}


def load_schema(schema: Schema | dict[str, Any] | str | Path) -> Schema:
    if isinstance(schema, Schema):
        return schema
    if isinstance(schema, dict):
        return Schema.from_json_schema(schema)
    path = Path(schema)
    return Schema.from_json_schema(json.loads(path.read_text(encoding="utf-8")), name=path.stem)


def verify(
    source: Document | str | Path,
    schema: Schema | dict[str, Any] | str | Path,
    adapter: ExtractorAdapter | None = None,
    k: int = 1,
    calibrator: Calibrator | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> VerifiedResult:
    """Extract with any adapter and return trust-annotated fields.

    ``k > 1`` enables self-consistency (k adapter runs + consensus voting);
    ``calibrator`` must already be fit on a calibration split; ``threshold``
    is the accept cutoff (use ``policy.threshold_for_target_risk`` to derive
    one from a target error rate).
    """
    doc = source if isinstance(source, Document) else ingest_path(source)
    schema_obj = load_schema(schema)
    adapter = adapter or TextSearchAdapter()

    samples = adapter.extract_samples(doc, schema_obj, k)
    preds = consensus(samples) if k > 1 else samples[0]
    consensus_conf: dict[str, float] | None = None
    if k > 1:
        consensus_conf = {p.path: p.confidence for p in preds}

    preds = ground_predictions(preds, doc)

    scored: list[FieldPrediction] = []
    for pred in preds:
        signals = {
            "consensus": consensus_conf.get(pred.path) if consensus_conf else None,
            "grounding": grounding_confidence(pred) if pred.value is not None else None,
            "token_prob": token_prob_confidence(pred),
            "verbalized": verbalized_confidence(pred),
        }
        try:
            conf = combined_confidence(signals)
        except ValueError:  # no signals at all: keep the adapter's raw score
            conf = pred.confidence
        scored.append(pred.model_copy(update={"confidence": conf}))

    if calibrator is not None:
        calibrated = calibrator.transform([p.confidence for p in scored])
        scored = [p.model_copy(update={"confidence": float(c)}) for p, c in zip(scored, calibrated)]

    scored = apply_policy(scored, threshold)
    return VerifiedResult(doc_id=doc.doc_id, fields=scored, threshold=threshold)
