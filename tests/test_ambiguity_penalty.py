"""Ablation of the ambiguity-penalty forms for grounding support (paper §5).

The penalty discounts a match score when a value is located at ``m`` equally-good
places. ``uniform`` (1/m) is the calibrated default -- the posterior P(true
source) under a uniform prior over the m matches; ``sqrt``/``log`` are softer;
``softmax`` (score-weighted competition against m-1 baseline candidates) and
``entropy`` (Shannon entropy in bits of a uniform prior over m) are two more.
These are hand-computed regression values pinning each mode's behavior.
"""

import math

import pytest

from verifydoc.grounding.attach import ambiguity_penalty


class TestAmbiguityPenalty:
    def test_single_match_is_never_penalized(self):
        for mode in ("uniform", "sqrt", "log", "softmax", "entropy", "none"):
            assert ambiguity_penalty(0.9, 1, mode) == 0.9

    def test_none_leaves_score(self):
        assert ambiguity_penalty(0.8, 5, "none") == 0.8

    def test_uniform_is_one_over_m(self):
        assert ambiguity_penalty(0.8, 4, "uniform") == pytest.approx(0.2)

    def test_sqrt_is_one_over_root_m(self):
        assert ambiguity_penalty(0.8, 4, "sqrt") == pytest.approx(0.4)

    def test_log_form(self):
        assert ambiguity_penalty(0.8, 4, "log") == pytest.approx(0.8 / (1.0 + math.log(4)))

    def test_softmax_form(self):
        # winner's logit is its raw score, the other m-1 tied candidates sit
        # at baseline logit 0 -> discount is the softmax posterior on the winner
        s, m = 0.8, 4
        w = math.exp(s)
        expected = s * w / (w + (m - 1))
        assert ambiguity_penalty(s, m, "softmax") == pytest.approx(expected)

    def test_softmax_depends_on_raw_score(self):
        # unlike uniform/sqrt/log, softmax discounts a stronger match less
        # harshly at the same m
        weak = ambiguity_penalty(0.3, 4, "softmax")
        strong = ambiguity_penalty(0.95, 4, "softmax")
        assert (weak / 0.3) < (strong / 0.95)

    def test_entropy_form(self):
        assert ambiguity_penalty(0.8, 4, "entropy") == pytest.approx(0.8 / (1.0 + math.log2(4)))

    def test_penalty_strength_ordering(self):
        # at m=4, s=0.8: uniform penalizes hardest, then entropy, then log,
        # then softmax, then sqrt, then none (unpenalized)
        s = 0.8
        u = ambiguity_penalty(s, 4, "uniform")
        en = ambiguity_penalty(s, 4, "entropy")
        lg = ambiguity_penalty(s, 4, "log")
        sm = ambiguity_penalty(s, 4, "softmax")
        sq = ambiguity_penalty(s, 4, "sqrt")
        no = ambiguity_penalty(s, 4, "none")
        assert u < en < lg < sm < sq < no

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            ambiguity_penalty(0.8, 4, "banana")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])