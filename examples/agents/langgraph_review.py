"""LangGraph trust-gated review: escalate only low-confidence fields.

A LangGraph agent that extracts with any adapter, calls VerifyDoc, **auto-accepts
confident + grounded fields**, and routes only the ``review`` fields to a
human-in-the-loop ``interrupt()`` step — the accept / escalate / finalize loop.

The trust logic lives in plain node functions so it runs (and is tested) with
**zero dependencies**; ``build_langgraph_app()`` wires those same nodes into a
real ``StateGraph`` with ``interrupt()`` when ``langgraph`` is installed::

    # framework-free (works everywhere, e.g. in CI):
    python examples/agents/langgraph_review.py

    # with LangGraph installed, the human step becomes a real interrupt():
    app = build_langgraph_app()
    app.invoke({"document": DOC, "threshold": 0.8}, config={"configurable": {"thread_id": "1"}})
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

from verifydoc import verify
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

# A resolver answers one escalated field → the human-confirmed value.
Resolver = Callable[[dict[str, Any]], Any]


class ReviewState(TypedDict, total=False):
    document: str
    threshold: float
    accepted: dict[str, Any]  # confident + grounded, safe to act on
    pending: list[dict[str, Any]]  # fields awaiting a human
    resolved: dict[str, Any]  # human-confirmed values
    final: dict[str, Any]  # accepted ∪ resolved


# --- nodes: pure functions of state (this is the whole trust policy) ---------


def extract_and_verify(state: ReviewState) -> ReviewState:
    """Extract + attach confidence/grounding/decision, then split accept/review."""
    result = verify(
        document_from_text("doc", [state["document"]]),
        SCHEMA,
        threshold=state.get("threshold", 0.8),
    )
    accepted = {f.path: f.value for f in result.fields if f.decision == "accept"}
    pending = [
        {
            "field": f.path,
            "value": f.value,
            "confidence": round(f.confidence, 3),
            "page": f.grounding.page if f.grounding else None,
        }
        for f in result.fields
        if f.decision == "review"
    ]
    return {**state, "accepted": accepted, "pending": pending, "resolved": {}}


def route_after_verify(state: ReviewState) -> str:
    """Conditional edge: escalate iff any field failed the trust bar."""
    return "human_review" if state.get("pending") else "finalize"


def human_review(state: ReviewState, resolve: Resolver) -> ReviewState:
    """Escalate each pending field to a human (``resolve``) and collect answers.

    In the LangGraph app ``resolve`` is ``interrupt(field)`` — the graph pauses
    and resumes with the human's value. Offline, it is any callback.
    """
    resolved = {item["field"]: resolve(item) for item in state.get("pending", [])}
    return {**state, "resolved": resolved, "pending": []}


def finalize(state: ReviewState) -> ReviewState:
    """Merge auto-accepted + human-confirmed values into the trusted record."""
    return {**state, "final": {**state.get("accepted", {}), **state.get("resolved", {})}}


# --- framework-free driver (used by the CLI demo and the tests) --------------


def run(document: str, *, threshold: float = 0.8, resolve: Resolver | None = None) -> ReviewState:
    """Execute the graph as plain Python: verify → (route) → review? → finalize."""
    resolve = resolve or (lambda item: item["value"])  # default: accept as-is
    state: ReviewState = {"document": document, "threshold": threshold}
    state = extract_and_verify(state)
    if route_after_verify(state) == "human_review":
        state = human_review(state, resolve)
    return finalize(state)


# --- real LangGraph wiring (only if langgraph is installed) ------------------


def build_langgraph_app() -> Any:  # pragma: no cover - requires optional langgraph
    """Wire the nodes into a ``StateGraph`` whose review node is an ``interrupt()``."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    def _review_node(state: ReviewState) -> ReviewState:
        return human_review(state, resolve=lambda item: interrupt(item))

    graph = StateGraph(ReviewState)
    graph.add_node("extract_and_verify", extract_and_verify)
    graph.add_node("human_review", _review_node)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "extract_and_verify")
    graph.add_conditional_edges("extract_and_verify", route_after_verify)
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def main() -> None:
    # A human resolver that just confirms the extractor's value (demo only).
    state = run(DOCUMENT, threshold=0.8, resolve=lambda item: item["value"])
    print("Auto-accepted (agent acts on these):")
    for path, value in state["accepted"].items():
        print(f"  ✅ {path} = {value!r}")
    print("\nEscalated to human interrupt() (not trusted blindly):")
    for item in [{"field": k} for k in state["resolved"]]:
        print(f"  ⚠️  {item['field']} → confirmed {state['resolved'][item['field']]!r}")
    if not state["resolved"]:
        print("  (none — everything cleared the trust bar)")
    print(f"\nFinal trusted record: {state['final']}")


if __name__ == "__main__":
    main()
