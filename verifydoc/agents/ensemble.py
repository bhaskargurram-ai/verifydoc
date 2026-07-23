"""Multi-extractor ensemble + adjudication.

Consensus over k samples of *one* extractor can't escape that extractor's blind
spots. Running several **different** extractors (an OCR pipeline, a VLM, an API
model) and adjudicating per field does: where they agree, trust rises; where
they disagree, the best-grounded reading wins and genuine splits stay ``review``.

``adjudicate`` is the judge — a pure function over each extractor's grounded
predictions. Per field path it votes by normalized value, picks the winning
value (ties broken by grounding support), and emits a fused ``FieldPrediction``
whose confidence is the standard ``combined_confidence`` of two signals:
cross-extractor **agreement** and the winner's **grounding support**. The normal
abstention policy then decides accept/review, so the ensemble stays consistent
with the rest of VerifyDoc. ``ensemble_verify`` wraps it around real adapters.

Model-agnostic: extractors are just ``ExtractorAdapter``s; no SDK is imported.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.calibration.base import Calibrator
from verifydoc.confidence.combined import combined_confidence
from verifydoc.eval.extraction import normalize_text
from verifydoc.ingest.loader import ingest_path
from verifydoc.pipeline import DEFAULT_THRESHOLD, VerifiedResult, load_schema, verify
from verifydoc.policy import apply_policy
from verifydoc.types import Document, FieldPrediction, Schema


def _support(pred: FieldPrediction) -> float:
    return pred.grounding.support if pred.grounding is not None else 0.0


def adjudicate(
    field_lists: Sequence[Sequence[FieldPrediction]],
    extractor_names: Sequence[str],
    *,
    n_total: int | None = None,
) -> list[FieldPrediction]:
    """Fuse several extractors' predictions into one prediction per field path.

    For each path: vote by normalized value across extractors; the winning value
    is the modal one (ties broken by best grounding support). ``agreement`` is
    the fraction of the ``n_total`` extractors backing the winner; the fused
    field takes the value + grounding of the best-grounded agreeing extractor and
    a ``combined_confidence`` of {agreement, winner support}. ``meta['ensemble']``
    records the votes and dissenters for auditability.
    """
    total = n_total if n_total is not None else len(field_lists)
    by_path: dict[str, list[FieldPrediction]] = {}
    order: list[str] = []
    for fields in field_lists:
        for f in fields:
            if f.value is None:
                continue
            if f.path not in by_path:
                by_path[f.path] = []
                order.append(f.path)
            by_path[f.path].append(f)

    fused: list[FieldPrediction] = []
    for path in order:
        cands = by_path[path]
        norm = {id(p): normalize_text(p.value).casefold() for p in cands}
        votes = Counter(norm[id(p)] for p in cands)
        top_count = votes.most_common(1)[0][1]
        tied = [v for v, c in votes.items() if c == top_count]
        if len(tied) > 1:
            winner = max(
                tied,
                key=lambda v: max(_support(p) for p in cands if norm[id(p)] == v),
            )
        else:
            winner = tied[0]
        agreeing = [p for p in cands if norm[id(p)] == winner]
        best = max(agreeing, key=_support)
        agreement = len(agreeing) / total if total else 0.0
        grounded = best.grounding is not None
        confidence = combined_confidence(
            {"consensus": agreement, "grounding": _support(best) if grounded else None}
        )
        dissent = sorted({norm[id(p)] for p in cands if norm[id(p)] != winner})
        fused.append(
            FieldPrediction(
                path=path,
                value=best.value,
                confidence=confidence,
                grounding=best.grounding,
                meta={
                    "ensemble": {
                        "agreement": agreement,
                        "n_extractors": total,
                        "n_agree": len(agreeing),
                        "votes": dict(votes),
                        "dissent": dissent,
                    }
                },
            )
        )
    return fused


def ensemble_verify(
    source: Document | str,
    schema: Schema | dict[str, Any] | str,
    adapters: Sequence[ExtractorAdapter],
    *,
    names: Sequence[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    k: int = 1,
    calibrator: Calibrator | None = None,
) -> VerifiedResult:
    """Verify a document with an ensemble of extractors + per-field adjudication.

    Each adapter is run through the standard pipeline (grounding included), the
    grounded predictions are adjudicated, and the fused fields go through the
    normal abstention policy.
    """
    if not adapters:
        raise ValueError("need at least one adapter")
    doc = source if isinstance(source, Document) else ingest_path(source)
    schema_obj = load_schema(schema)
    labels = (
        list(names)
        if names is not None
        else [getattr(a, "name", f"e{i}") for i, a in enumerate(adapters)]
    )

    per = [
        verify(doc, schema_obj, adapter=a, k=k, threshold=threshold, calibrator=calibrator)
        for a in adapters
    ]
    fused = adjudicate([r.fields for r in per], labels, n_total=len(adapters))
    scored = apply_policy(fused, threshold)
    return VerifiedResult(doc_id=doc.doc_id, fields=scored, threshold=threshold)
