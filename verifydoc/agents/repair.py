"""Trust-gated agentic extraction: extract → verify → repair → escalate.

VerifyDoc's abstention policy *flags* the fields you shouldn't trust. This layer
*acts* on that signal: fields the base extractor leaves at ``review`` are retried
through progressively stronger **repair tiers** (a better adapter, more
self-consistency samples, …); a field is adopted only if a tier returns it
``accept``-ed and grounded. Whatever is still ``review`` after the last tier is
**escalated** to a human ``resolver`` — so a human sees only the residue, not
every field.

The point (and the paper thesis): tiers are tried **lazily**, cheapest first, and
only while ``review`` fields remain — so cost scales with document difficulty,
not with the number of tiers. ``AgenticResult`` reports ``n_extract_calls`` so
the accuracy-at-fixed-cost tradeoff is measurable.

Model-agnostic by construction: a tier is just an ``ExtractorAdapter`` (or the
pipeline default) plus a ``k``; no model SDK is imported here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from verifydoc.adapters.base import ExtractorAdapter
from verifydoc.calibration.base import Calibrator
from verifydoc.pipeline import DEFAULT_THRESHOLD, VerifiedResult, verify
from verifydoc.types import Document, FieldPrediction, Schema

# A human (or oracle) escalation step: given the still-uncertain field, return
# its confirmed value.
Resolver = Callable[[FieldPrediction], Any]


@dataclass
class RepairTier:
    """One escalation tier: re-extract with this adapter / self-consistency ``k``."""

    name: str
    adapter: ExtractorAdapter | None = None  # None → pipeline default (text-search)
    k: int = 1


@dataclass
class RepairAttempt:
    """A field that a repair tier moved from ``review`` to ``accept``."""

    path: str
    tier: str
    before: FieldPrediction
    after: FieldPrediction


@dataclass
class AgenticResult:
    """Outcome of an agentic run: the final result + what it took to get there."""

    result: VerifiedResult
    repaired: list[str] = field(default_factory=list)
    escalated: list[str] = field(default_factory=list)
    human_resolved: list[str] = field(default_factory=list)
    attempts: list[RepairAttempt] = field(default_factory=list)
    tiers_used: int = 0
    n_extract_calls: int = 1


def merge_repairs(
    base: VerifiedResult,
    tier_candidates: Iterable[tuple[str, Callable[[], VerifiedResult]]],
    *,
    resolver: Resolver | None = None,
    require_grounded: bool = True,
) -> AgenticResult:
    """Resolve ``review`` fields from lazily-produced candidate results.

    ``tier_candidates`` yields ``(tier_name, make_result)`` pairs; ``make_result``
    is invoked **only while** ``review`` fields remain, so unused tiers cost
    nothing. For each still-``review`` field, the first tier that returns it
    ``accept``-ed (and grounded, when ``require_grounded``) wins. Any residue is
    passed to ``resolver`` (if given) and marked human-confirmed.
    """
    by_path: dict[str, FieldPrediction] = {f.path: f for f in base.fields}
    order = [f.path for f in base.fields]
    review = [p for p in order if by_path[p].decision == "review"]

    attempts: list[RepairAttempt] = []
    repaired: list[str] = []
    tiers_used = 0

    for tier_name, make_result in tier_candidates:
        if not review:
            break
        candidate = make_result()
        tiers_used += 1
        cand_by = {f.path: f for f in candidate.fields}
        for path in list(review):
            cf = cand_by.get(path)
            if cf is None or cf.value is None or cf.decision != "accept":
                continue
            if require_grounded and cf.grounding is None:
                continue
            attempts.append(
                RepairAttempt(path=path, tier=tier_name, before=by_path[path], after=cf)
            )
            by_path[path] = cf
            repaired.append(path)
            review.remove(path)

    human_resolved: list[str] = []
    if resolver is not None:
        for path in list(review):
            value = resolver(by_path[path])
            src = by_path[path]
            by_path[path] = src.model_copy(
                update={
                    "value": value,
                    "decision": "accept",
                    "meta": {**src.meta, "escalated": True, "source": "human"},
                }
            )
            human_resolved.append(path)
        review = []

    final = VerifiedResult(
        doc_id=base.doc_id, fields=[by_path[p] for p in order], threshold=base.threshold
    )
    return AgenticResult(
        result=final,
        repaired=repaired,
        escalated=list(review),
        human_resolved=human_resolved,
        attempts=attempts,
        tiers_used=tiers_used,
    )


def agentic_verify(
    source: Document | str,
    schema: Schema | dict[str, Any] | str,
    *,
    base_adapter: ExtractorAdapter | None = None,
    base_k: int = 1,
    tiers: Iterable[RepairTier] = (),
    threshold: float = DEFAULT_THRESHOLD,
    calibrator: Calibrator | None = None,
    resolver: Resolver | None = None,
) -> AgenticResult:
    """Extract with ``base_adapter``, then repair ``review`` fields through
    ``tiers`` (lazily), then escalate the residue to ``resolver``.

    Returns an :class:`AgenticResult` whose ``n_extract_calls`` counts the base
    run plus every tier that actually ran (each costs its ``k`` samples).
    """
    base = verify(
        source, schema, adapter=base_adapter, k=base_k, threshold=threshold, calibrator=calibrator
    )

    calls = [max(1, base_k)]  # mutable cost accumulator (base run)

    def _tier_thunk(tier: RepairTier) -> Callable[[], VerifiedResult]:
        def run() -> VerifiedResult:
            calls[0] += max(1, tier.k)
            return verify(
                source,
                schema,
                adapter=tier.adapter,
                k=tier.k,
                threshold=threshold,
                calibrator=calibrator,
            )

        return run

    out = merge_repairs(
        base,
        ((tier.name, _tier_thunk(tier)) for tier in tiers),
        resolver=resolver,
    )
    out.n_extract_calls = calls[0]
    return out
