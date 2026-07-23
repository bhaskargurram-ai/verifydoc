"""Tests for the multi-extractor ensemble + adjudication (offline)."""

import pytest

from verifydoc.adapters.canned import CannedAdapter
from verifydoc.agents import adjudicate, ensemble_verify
from verifydoc.ingest import document_from_text
from verifydoc.types import FieldPrediction, Grounding


def pred(path, value, support=None, conf=0.8):
    g = Grounding(page=0, char_span=(0, 3), support=support) if support is not None else None
    return FieldPrediction(path=path, value=value, confidence=conf, grounding=g)


class TestAdjudicate:
    def test_full_agreement_is_confident_and_grounded(self):
        lists = [
            [pred("total", "100", 0.9)],
            [pred("total", "100", 0.8)],
            [pred("total", "100", 0.95)],
        ]
        (f,) = adjudicate(lists, ["a", "b", "c"])
        assert f.value == "100"
        assert f.meta["ensemble"]["agreement"] == pytest.approx(1.0)
        assert f.grounding is not None and f.grounding.support == pytest.approx(
            0.95
        )  # best-grounded
        assert f.meta["ensemble"]["dissent"] == []

    def test_majority_wins_and_best_grounded_supplies_value(self):
        lists = [
            [pred("total", "100", 0.6)],  # agrees, weakly grounded
            [pred("total", "100", 0.92)],  # agrees, best grounded → supplies value/grounding
            [pred("total", "999")],  # dissent, ungrounded
        ]
        (f,) = adjudicate(lists, ["a", "b", "c"])
        assert f.value == "100"
        assert f.meta["ensemble"]["agreement"] == pytest.approx(2 / 3)
        assert f.grounding.support == pytest.approx(0.92)
        assert f.meta["ensemble"]["dissent"] == ["999"]

    def test_tie_broken_by_grounding_support(self):
        lists = [[pred("total", "100", 0.9)], [pred("total", "200")]]  # 1 vs 1
        (f,) = adjudicate(lists, ["a", "b"])
        assert f.value == "100"  # grounded group wins the tie
        assert f.meta["ensemble"]["agreement"] == pytest.approx(0.5)

    def test_none_values_are_ignored(self):
        lists = [[pred("x", None)], [pred("x", "v", 0.9)]]
        (f,) = adjudicate(lists, ["a", "b"])
        assert f.value == "v"
        # only one extractor asserted → agreement 1/2 (abstention counts against)
        assert f.meta["ensemble"]["agreement"] == pytest.approx(0.5)

    def test_normalization_groups_votes(self):
        lists = [[pred("v", " ACME ", 0.9)], [pred("v", "acme", 0.5)], [pred("v", "Globex")]]
        (f,) = adjudicate(lists, ["a", "b", "c"])
        assert f.value == " ACME "  # raw value of the best-grounded agreeing extractor
        assert f.meta["ensemble"]["agreement"] == pytest.approx(2 / 3)


class TestEnsembleVerify:
    DOC = document_from_text("doc", ["Vendor: ACME\nTotal: 100"])
    SCHEMA = {"type": "object", "properties": {"total": {"type": "number", "x-numeric-tol": 0.01}}}

    def test_agreement_accepts(self):
        res = ensemble_verify(
            self.DOC,
            self.SCHEMA,
            [CannedAdapter({"total": "100"}), CannedAdapter({"total": "100"})],
            names=["ocr", "vlm"],
            threshold=0.8,
        )
        by = {f.path: f for f in res.fields}
        assert by["total"].decision == "accept"
        assert by["total"].meta["ensemble"]["agreement"] == pytest.approx(1.0)

    def test_disagreement_prefers_grounded_reading(self):
        # one extractor reads the on-page value (grounded), the other hallucinates
        res = ensemble_verify(
            self.DOC,
            self.SCHEMA,
            [CannedAdapter({"total": "100"}), CannedAdapter({"total": "999"})],
            names=["good", "bad"],
            threshold=0.8,
        )
        by = {f.path: f for f in res.fields}
        assert str(by["total"].value) == "100"
        assert "999" in by["total"].meta["ensemble"]["dissent"]

    def test_requires_at_least_one_adapter(self):
        with pytest.raises(ValueError):
            ensemble_verify(self.DOC, self.SCHEMA, [])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
