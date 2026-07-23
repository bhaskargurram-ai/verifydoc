"""Tests for the confidence signals (token-prob, verbalized, consensus, grounding, combined)."""

import math

import pytest

from verifydoc.confidence import (
    adaptive_consensus,
    apply_token_prob,
    apply_verbalized,
    combined_confidence,
    consensus,
    grounding_confidence,
    token_prob_confidence,
    verbalized_confidence,
)
from verifydoc.types import FieldPrediction, Grounding


def pred(path="f", value="v", conf=0.5, **meta):
    return FieldPrediction(path=path, value=value, confidence=conf, meta=meta)


class TestTokenProb:
    def test_geometric_mean(self):
        p = pred(token_logprobs=[math.log(0.9), math.log(0.4)])
        assert token_prob_confidence(p) == pytest.approx(math.sqrt(0.36))

    def test_min_aggregate(self):
        p = pred(token_logprobs=[math.log(0.9), math.log(0.4)])
        assert token_prob_confidence(p, "min") == pytest.approx(0.4)

    def test_prod_aggregate(self):
        p = pred(token_logprobs=[math.log(0.5), math.log(0.5)])
        assert token_prob_confidence(p, "prod") == pytest.approx(0.25)

    def test_missing_meta(self):
        assert token_prob_confidence(pred()) is None

    def test_apply_preserves_missing(self):
        preds = [pred(token_logprobs=[math.log(0.8)]), pred(conf=0.3)]
        out = apply_token_prob(preds)
        assert out[0].confidence == pytest.approx(0.8)
        assert out[1].confidence == 0.3  # untouched
        assert preds[0].confidence == 0.5  # copies, not mutation


class TestVerbalized:
    def test_reads_and_clamps(self):
        assert verbalized_confidence(pred(verbalized_confidence=0.7)) == 0.7
        assert verbalized_confidence(pred(verbalized_confidence=1.7)) == 1.0
        assert verbalized_confidence(pred()) is None

    def test_apply(self):
        out = apply_verbalized([pred(verbalized_confidence=0.25)])
        assert out[0].confidence == 0.25


class TestConsensus:
    def test_unanimous(self):
        runs = [[pred(value="42.50")] for _ in range(3)]
        (result,) = consensus(runs)
        assert result.value == "42.50"
        assert result.confidence == pytest.approx(1.0)

    def test_majority_with_normalization(self):
        runs = [
            [pred(value=" ACME Corp ")],  # normalizes to same vote
            [pred(value="acme corp")],
            [pred(value="Globex")],
        ]
        (result,) = consensus(runs)
        assert result.value == " ACME Corp "  # first-seen raw form of modal vote
        assert result.confidence == pytest.approx(2 / 3)

    def test_missing_field_counts_as_omit_vote(self):
        runs = [[pred(value="x")], [], [pred(value="x")], []]
        (result,) = consensus(runs)
        assert result.value == "x"
        assert result.confidence == pytest.approx(0.5)

    def test_majority_omit_returns_none(self):
        runs = [[pred(value=None)], [], [pred(value="x")]]
        (result,) = consensus(runs)
        assert result.value is None
        assert result.confidence == pytest.approx(2 / 3)

    def test_grounding_from_modal_sample(self):
        g = Grounding(page=0, bbox=(0.1, 0.1, 0.2, 0.2), support=0.9)
        runs = [
            [FieldPrediction(path="f", value="v", grounding=g)],
            [pred(value="v")],
        ]
        (result,) = consensus(runs)
        assert result.grounding == g

    def test_multiple_paths_ordered(self):
        runs = [
            [pred(path="a", value="1"), pred(path="b", value="2")],
            [pred(path="b", value="2")],
        ]
        results = consensus(runs)
        assert [r.path for r in results] == ["a", "b"]
        by_path = {r.path: r for r in results}
        assert by_path["a"].confidence == pytest.approx(0.5)
        assert by_path["b"].confidence == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            consensus([])


class TestAdaptiveConsensus:
    """#15 — draw extra samples only for ambiguous, near-threshold documents."""

    @staticmethod
    def _counting_sampler(runs):
        """Return (sampler, calls) where sampler yields runs in order, cycling,
        and `calls` is a mutable [count] the test can read afterwards."""
        calls = [0]
        seq = list(runs)

        def sampler():
            run = seq[calls[0] % len(seq)]
            calls[0] += 1
            return [FieldPrediction(**p) for p in run]

        return sampler, calls

    def test_single_pass_fast_path(self):
        sampler, calls = self._counting_sampler([[{"path": "f", "value": "x"}]])
        preds, n = adaptive_consensus(sampler, k_max=1)
        assert n == 1 and calls[0] == 1
        assert preds[0].confidence == pytest.approx(1.0)  # one sample → unanimous

    def test_unanimous_stops_at_k_min(self):
        # every draw agrees → confidence 1.0 (far from 0.5) → stop at the floor
        sampler, calls = self._counting_sampler([[{"path": "f", "value": "x"}]])
        _preds, n = adaptive_consensus(sampler, k_min=2, k_max=6)
        assert n == 2 and calls[0] == 2

    def test_ambiguous_field_draws_up_to_k_max(self):
        # alternating votes keep confidence near 0.5 → keep drawing to the cap
        runs = [[{"path": "f", "value": "a"}], [{"path": "f", "value": "b"}]]
        sampler, calls = self._counting_sampler(runs)
        _preds, n = adaptive_consensus(sampler, k_min=2, k_max=5)
        assert n == 5 and calls[0] == 5

    def test_budget_caps_total_draws(self):
        runs = [[{"path": "f", "value": "a"}], [{"path": "f", "value": "b"}]]
        sampler, calls = self._counting_sampler(runs)
        _preds, n = adaptive_consensus(sampler, k_min=2, k_max=8, budget=3)
        assert n == 3 and calls[0] == 3

    def test_budget_below_k_min_clamps(self):
        sampler, calls = self._counting_sampler([[{"path": "f", "value": "x"}]])
        _preds, n = adaptive_consensus(sampler, k_min=4, k_max=6, budget=1)
        assert n == 1 and calls[0] == 1

    def test_invalid_bounds_raise(self):
        sampler, _ = self._counting_sampler([[{"path": "f", "value": "x"}]])
        with pytest.raises(ValueError):
            adaptive_consensus(sampler, k_min=0)


class TestGroundingBased:
    def test_support_as_confidence(self):
        p = FieldPrediction(path="f", value="v", grounding=Grounding(page=0, support=0.85))
        assert grounding_confidence(p) == 0.85

    def test_ungrounded_floor(self):
        assert grounding_confidence(pred()) == 0.0
        assert grounding_confidence(pred(), ungrounded_confidence=0.2) == 0.2


class TestCombined:
    def test_weighted_mean(self):
        val = combined_confidence(
            {"consensus": 1.0, "grounding": 0.5},
            weights={"consensus": 0.5, "grounding": 0.5},
        )
        assert val == pytest.approx(0.75)

    def test_renormalizes_over_available(self):
        # only consensus present: full weight goes to it
        assert combined_confidence({"consensus": 0.6, "token_prob": None}) == pytest.approx(0.6)

    def test_default_weights(self):
        val = combined_confidence(
            {"consensus": 1.0, "grounding": 0.0, "token_prob": None, "verbalized": None}
        )
        assert val == pytest.approx(0.5 / 0.8)

    def test_errors(self):
        with pytest.raises(ValueError):
            combined_confidence({"nope": 0.5})
        with pytest.raises(ValueError):
            combined_confidence({"consensus": None})
