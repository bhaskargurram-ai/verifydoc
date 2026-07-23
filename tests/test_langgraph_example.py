"""Offline tests for the LangGraph trust-gated review example (#29).

Loads the example by file path (examples/ is not an installed package) and
exercises both branches of the accept / escalate / finalize loop without
langgraph or any network.
"""

import importlib.util
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agents" / "langgraph_review.py"
_spec = importlib.util.spec_from_file_location("langgraph_review", _EXAMPLE)
assert _spec and _spec.loader
lgr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lgr)


class TestRouting:
    def test_escalates_when_pending(self):
        assert lgr.route_after_verify({"pending": [{"field": "total"}]}) == "human_review"

    def test_finalizes_when_clear(self):
        assert lgr.route_after_verify({"pending": []}) == "finalize"
        assert lgr.route_after_verify({}) == "finalize"


class TestHumanReviewNode:
    def test_resolver_called_per_pending_field(self):
        calls = []

        def resolve(item):
            calls.append(item["field"])
            return f"HUMAN:{item['value']}"

        state = {
            "accepted": {"vendor": "ACME"},
            "pending": [{"field": "total", "value": "999"}],
        }
        out = lgr.human_review(state, resolve)
        assert calls == ["total"]
        assert out["resolved"] == {"total": "HUMAN:999"}
        assert out["pending"] == []  # cleared after review


class TestFinalize:
    def test_merges_accepted_and_resolved(self):
        state = {"accepted": {"vendor": "ACME"}, "resolved": {"total": "1234.50"}}
        assert lgr.finalize(state)["final"] == {"vendor": "ACME", "total": "1234.50"}


class TestRunEndToEnd:
    def test_accept_path_offline(self):
        # text-search grounds vendor + total confidently → auto-accepted, no human
        state = lgr.run(lgr.DOCUMENT, threshold=0.8)
        assert state["accepted"]["vendor"] == "ACME Supplies Ltd"
        assert "total" in state["accepted"]
        assert state["resolved"] == {}
        assert state["final"] == state["accepted"]

    def test_escalate_path_routes_to_resolver(self):
        # inject a pending field to prove the human branch wires end-to-end
        state = lgr.extract_and_verify({"document": lgr.DOCUMENT, "threshold": 0.8})
        state["pending"] = [{"field": "po_number", "value": "PO-1", "confidence": 0.4}]
        state = lgr.human_review(state, resolve=lambda item: "CORRECTED")
        final = lgr.finalize(state)["final"]
        assert final["po_number"] == "CORRECTED"
        assert final["vendor"] == "ACME Supplies Ltd"  # accepted survives the merge
