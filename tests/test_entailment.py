"""Tests for entailment-based grounding verification (#16, grounding-sweep P4)."""

import math

import pytest

from verifydoc.confidence import (
    CrossEncoderEntailmentScorer,
    EntailmentScorer,
    LexicalEntailmentScorer,
    apply_entailment_grounding,
    entailment_support,
)
from verifydoc.confidence.entailment import _hypothesis, _premise
from verifydoc.types import FieldPrediction, Grounding


def grounded(path, value, support, char_span=(0, 10), page=0):
    g = Grounding(page=page, char_span=char_span, support=support)
    return FieldPrediction(path=path, value=value, confidence=0.5, grounding=g)


class FakeNLI:
    """An oracle NLI scorer: entails iff the hypothesis appears as a contiguous
    *token* run in the premise — so 'total is 100' is NOT entailed by
    'subtotal is 100' (semantic, not just substring/lexical presence)."""

    def score(self, premise: str, hypothesis: str) -> float:
        pt = premise.casefold().split()
        ht = hypothesis.casefold().split()
        if not ht:
            return 0.0
        return float(any(pt[i : i + len(ht)] == ht for i in range(len(pt) - len(ht) + 1)))


class TestLexicalScorer:
    def test_token_overlap_fraction(self):
        s = LexicalEntailmentScorer()
        assert s.score("total is 100 dollars", "total is 100") == pytest.approx(1.0)
        assert s.score("total is 100", "vendor is acme") == pytest.approx(1 / 3)  # only "is"

    def test_empty_hypothesis(self):
        assert LexicalEntailmentScorer().score("anything", "") == 0.0

    def test_satisfies_protocol(self):
        assert isinstance(LexicalEntailmentScorer(), EntailmentScorer)


class TestHypothesisAndPremise:
    def test_hypothesis_uses_leaf_field_name(self):
        p = grounded("invoice.total_due", 100, 0.9)
        assert _hypothesis(p, "{field} is {value}") == "total due is 100"

    def test_premise_slices_char_span(self):
        p = grounded("total", 100, 0.9, char_span=(6, 17))
        assert _premise(p, "Line: Subtotal: 100 more") == "Subtotal: 1"

    def test_premise_none_when_ungrounded(self):
        p = FieldPrediction(path="x", value="v", confidence=0.5)
        assert _premise(p, "text") is None

    def test_premise_whole_text_when_no_span(self):
        g = Grounding(page=0, support=0.8)  # grounded, no char_span
        p = FieldPrediction(path="x", value="v", confidence=0.5, grounding=g)
        assert _premise(p, "the whole thing") == "the whole thing"


class TestEntailmentSupport:
    def test_none_for_ungrounded(self):
        p = FieldPrediction(path="x", value="v", confidence=0.5)
        assert entailment_support(p, "text", FakeNLI()) is None

    def test_oracle_scores(self):
        p = grounded("total", "100", 0.9, char_span=(0, 14))
        assert entailment_support(p, "total is 100", FakeNLI()) == 1.0
        assert entailment_support(p, "subtotal is 90", FakeNLI()) == 0.0


class TestApplyEntailmentGrounding:
    def test_catches_lexically_present_but_wrong(self):
        # value 100 is lexically in the span, but the span asserts *subtotal* 100.
        # min-combine drops support to 0 → the field will route to review.
        p = grounded("total", "100", support=0.95, char_span=(0, 15))
        (out,) = apply_entailment_grounding([p], "subtotal is 100", FakeNLI(), combine="min")
        assert out.grounding.support == pytest.approx(0.0)

    def test_keeps_support_when_entailed(self):
        p = grounded("total", "100", support=0.8, char_span=(0, 12))
        (out,) = apply_entailment_grounding([p], "total is 100", FakeNLI(), combine="min")
        assert out.grounding.support == pytest.approx(0.8)  # min(0.8, 1.0)

    def test_product_combine(self):
        p = grounded("total", "100", support=0.6, char_span=(0, 11))

        class Half:
            def score(self, premise, hypothesis):
                return 0.5

        (out,) = apply_entailment_grounding([p], "total is 100", Half(), combine="product")
        assert out.grounding.support == pytest.approx(0.3)

    def test_ungrounded_passthrough(self):
        p = FieldPrediction(path="x", value="v", confidence=0.5)
        (out,) = apply_entailment_grounding([p], "text", FakeNLI())
        assert out is p  # returned unchanged

    def test_default_scorer_is_lexical(self):
        p = grounded("total", "100", support=0.9, char_span=(0, 12))
        (out,) = apply_entailment_grounding([p], "total is 100", combine="replace")
        assert out.grounding.support == pytest.approx(1.0)  # lexical overlap = 1

    def test_unknown_combine_raises(self):
        with pytest.raises(ValueError):
            apply_entailment_grounding([], "t", FakeNLI(), combine="nope")


class TestCrossEncoderScorer:
    def test_softmax_over_injected_model(self):
        class FakeModel:  # stands in for a sentence_transformers CrossEncoder
            def predict(self, pairs):
                return [[0.0, math.log(4.0), 0.0]]  # entailment logit at index 1

        s = CrossEncoderEntailmentScorer(model=FakeModel(), entail_index=1)
        # softmax([0, ln4, 0]) = [1, 4, 1]/6 → entailment prob = 4/6
        assert s.score("p", "h") == pytest.approx(4 / 6)
