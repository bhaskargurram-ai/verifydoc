"""Tests for the agentic verify → repair → escalate layer (offline)."""

import pytest

from verifydoc.adapters.canned import CannedAdapter
from verifydoc.agents import RepairTier, agentic_verify, merge_repairs
from verifydoc.ingest import document_from_text
from verifydoc.pipeline import VerifiedResult
from verifydoc.types import FieldPrediction, Grounding


def field(path, value="v", decision="accept", grounded=True, conf=0.9):
    g = Grounding(page=0, char_span=(0, 3), support=0.9) if grounded else None
    return FieldPrediction(path=path, value=value, confidence=conf, decision=decision, grounding=g)


def result(fields, doc_id="doc", threshold=0.8):
    return VerifiedResult(doc_id=doc_id, fields=fields, threshold=threshold)


def tier(name, res):
    """(name, thunk) where the thunk records that it was called."""
    calls = []
    return name, (lambda: (calls.append(1), res)[1]), calls


class TestMergeRepairs:
    def test_tier_repairs_reviewed_field(self):
        base = result([field("id"), field("total", value="?", decision="review", conf=0.3)])
        cand = result([field("total", value="100", decision="accept", conf=0.95)])
        name, thunk, _ = tier("stronger", cand)
        out = merge_repairs(base, [(name, thunk)])
        assert out.repaired == ["total"] and out.escalated == []
        assert out.tiers_used == 1 and len(out.attempts) == 1
        by = {f.path: f for f in out.result.fields}
        assert by["total"].decision == "accept" and by["total"].value == "100"
        assert [f.path for f in out.result.fields] == ["id", "total"]  # order preserved

    def test_no_reviews_means_no_tier_is_run(self):
        base = result([field("id"), field("total", value="100")])
        called = []
        thunk = lambda: (called.append(1), result([]))[1]  # noqa: E731
        out = merge_repairs(base, [("stronger", thunk)])
        assert called == [] and out.tiers_used == 0 and out.repaired == []

    def test_ungrounded_accept_is_not_adopted(self):
        base = result([field("total", value="?", decision="review", conf=0.3)])
        cand = result([field("total", value="100", decision="accept", grounded=False)])
        out = merge_repairs(base, [("weak", (lambda: cand))])
        assert out.repaired == [] and out.escalated == ["total"]

    def test_require_grounded_false_adopts_ungrounded(self):
        base = result([field("total", value="?", decision="review", conf=0.3)])
        cand = result([field("total", value="100", decision="accept", grounded=False)])
        out = merge_repairs(base, [("weak", (lambda: cand))], require_grounded=False)
        assert out.repaired == ["total"]

    def test_first_tier_wins_and_later_tier_skipped(self):
        base = result([field("total", value="?", decision="review", conf=0.3)])
        cand1 = result([field("total", value="100", decision="accept")])
        n1, t1, _ = tier("t1", cand1)
        n2, t2, t2_calls = tier("t2", result([]))
        out = merge_repairs(base, [(n1, t1), (n2, t2)])
        assert out.repaired == ["total"] and out.tiers_used == 1
        assert t2_calls == []  # second tier never consulted — reviews already cleared

    def test_escalate_to_resolver(self):
        base = result([field("total", value="?", decision="review", conf=0.3)])
        seen = []
        out = merge_repairs(base, [], resolver=lambda f: (seen.append(f.path), "HUMAN-42")[1])
        assert seen == ["total"] and out.escalated == [] and out.human_resolved == ["total"]
        by = {f.path: f for f in out.result.fields}
        assert by["total"].value == "HUMAN-42" and by["total"].decision == "accept"
        assert by["total"].meta.get("escalated") is True

    def test_residue_without_resolver_is_escalated(self):
        base = result([field("total", value="?", decision="review", conf=0.3)])
        out = merge_repairs(base, [])
        assert out.escalated == ["total"] and out.human_resolved == []


class TestAgenticVerify:
    DOC = document_from_text("doc", ["Vendor: ACME\nTotal: 100"])
    SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}

    def test_repairs_via_a_stronger_tier_and_counts_cost(self):
        # base extractor hallucinates a value absent from the doc → ungrounded → review;
        # the repair tier returns the value that IS in the doc → grounded → accept.
        base = CannedAdapter({"total": "999"})
        strong = CannedAdapter({"total": "100"})
        out = agentic_verify(
            self.DOC,
            self.SCHEMA,
            base_adapter=base,
            tiers=[RepairTier("strong", adapter=strong)],
            threshold=0.8,
        )
        assert out.repaired == ["total"]
        assert out.n_extract_calls == 2  # base + one tier
        by = {f.path: f for f in out.result.fields}
        assert by["total"].decision == "accept" and str(by["total"].value) == "100"

    def test_no_tier_needed_costs_one_call(self):
        good = CannedAdapter({"total": "100"})
        out = agentic_verify(
            self.DOC, self.SCHEMA, base_adapter=good, tiers=[RepairTier("unused")], threshold=0.8
        )
        assert out.repaired == [] and out.tiers_used == 0 and out.n_extract_calls == 1

    def test_escalates_when_no_tier_resolves(self):
        base = CannedAdapter({"total": "999"})
        seen = []
        out = agentic_verify(
            self.DOC,
            self.SCHEMA,
            base_adapter=base,
            resolver=lambda f: (seen.append(f.path), 100)[1],
            threshold=0.8,
        )
        assert seen == ["total"] and out.human_resolved == ["total"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
