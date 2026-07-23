"""Trust-gated agent loop: extract -> grade with VerifyDoc -> accept / escalate / re-extract.

This is the pattern a LangGraph / CrewAI / OpenAI-Agents graph implements to use
documents *safely*: the agent only acts on fields VerifyDoc marks ``accept``
(confident + grounded); ``review`` fields are escalated to a human instead of
being trusted blindly, and optionally re-extracted. It runs here as plain Python
(no framework needed) so the loop is obvious; the escalate branch is where a
LangGraph ``interrupt()`` (human-in-the-loop), an OpenAI-Agents ``needs_approval``
tool, or a VerifyDoc review-queue would sit.

    python examples/agents/agent_review_loop.py
"""

from __future__ import annotations

from typing import Any

from verifydoc import VerifiedResult, verify
from verifydoc.ingest import document_from_text

DOCUMENT = """ACME SUPPLIES INVOICE
Invoice #: INV-2024-0912
Vendor: ACME Supplies Ltd
Date: 2024-03-04
Total: 1,234.50
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "invoice_id": {"type": "string"},
        "vendor": {"type": "string", "x-scoring": "semantic"},
        "total": {"type": "number", "x-numeric-tol": 0.01},
    },
}


def grade(text: str, threshold: float = 0.8) -> VerifiedResult:
    """The agent's tool call: extract + attach confidence/grounding/decision."""
    return verify(document_from_text("doc", [text]), SCHEMA, threshold=threshold)


def agent_step(text: str) -> dict[str, Any]:
    """One trust-gated decision cycle. Returns the routing of every field."""
    result = grade(text)
    accepted = {f.path: f.value for f in result.fields if f.decision == "accept"}
    escalate = [f for f in result.fields if f.decision == "review"]

    # 1) ACCEPT: safe to act on immediately (write to DB, continue the workflow).
    # 2) ESCALATE: hand to a human queue with the source region attached.
    #    In LangGraph:  raise interrupt({"field": f.path, "grounding": f.grounding})
    #    In OpenAI Agents: mark the tool needs_approval and surface the citation.
    # 3) RE-EXTRACT (optional): re-ask the extractor for just the escalated fields
    #    (higher k / a stronger adapter) before bothering a human.
    return {
        "auto_accepted": accepted,
        "needs_human": [
            {
                "field": f.path,
                "value": f.value,
                "confidence": round(f.confidence, 2),
                "page": f.grounding.page if f.grounding else None,
            }
            for f in escalate
        ],
    }


def main() -> None:
    routing = agent_step(DOCUMENT)
    print("Agent auto-accepted (acts on these):")
    for path, value in routing["auto_accepted"].items():
        print(f"  ✅ {path} = {value!r}")
    print("\nAgent escalated to a human (does NOT trust these):")
    for item in routing["needs_human"]:
        print(f"  ⚠️  {item['field']} = {item['value']!r} (conf {item['confidence']}, page {item['page']})")
    if not routing["needs_human"]:
        print("  (none — everything cleared the trust bar)")


if __name__ == "__main__":
    main()
