"""Trust-gated auto-repair: extract → verify → repair review fields → escalate.

`verifydoc.agents.agentic_verify` acts on the abstention signal instead of just
reporting it: a weak base extractor leaves a field at `review`; a stronger tier
re-extracts and repairs it; anything still uncertain is escalated to a human.
Tiers run lazily (cheapest first, only while review fields remain), so cost
scales with document difficulty — `n_extract_calls` makes that measurable.

    python examples/agents/auto_repair.py
"""

from __future__ import annotations

from typing import Any

from verifydoc.adapters.canned import CannedAdapter
from verifydoc.agents import RepairTier, agentic_verify
from verifydoc.ingest import document_from_text

DOCUMENT = document_from_text("invoice", ["Vendor: ACME Supplies\nTotal: 1,234.50"])
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}},
}


def main() -> None:
    # Base extractor hallucinates a total absent from the page → ungrounded → review.
    weak = CannedAdapter({"total": "9999.00"})
    # A stronger tier returns the value that IS on the page → grounded → accept.
    strong = CannedAdapter({"total": "1,234.50"})

    out = agentic_verify(
        DOCUMENT,
        SCHEMA,
        base_adapter=weak,
        tiers=[RepairTier("stronger-model", adapter=strong)],
        # a real deployment passes a human here; the demo just confirms the value
        resolver=lambda f: f.value,
    )

    print(f"extract calls : {out.n_extract_calls} (base + {out.tiers_used} repair tier)")
    print(f"repaired      : {out.repaired}")
    print(f"escalated     : {out.escalated}")
    for f in out.result.fields:
        print(f"  {f.path} = {f.value!r}  {f.decision}  (conf {f.confidence:.2f})")


if __name__ == "__main__":
    main()
