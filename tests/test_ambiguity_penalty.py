"""Ablation of the ambiguity-penalty forms for grounding support (paper §5).

The penalty discounts a match score when a value is located at ``m`` equally-good
places. ``uniform`` (1/m) is the calibrated default -- the posterior P(true
source) under a uniform prior over the m matches; ``sqrt``/``log`` are softer.
These are hand-computed regression values pinning each mode's behavior.
"""

import math

import pytest

from verifydoc.grounding.attach import ambiguity_penalty


class TestAmbiguityPenalty:
    def test_single_match_is_never_penalized(self):
        for mode in ("uniform", "sqrt", "log", "none"):
            assert ambiguity_penalty(0.9, 1, mode) == 0.9

    def test_none_leaves_score(self):
        assert ambiguity_penalty(0.8, 5, "none") == 0.8

    def test_uniform_is_one_over_m(self):
        assert ambiguity_penalty(0.8, 4, "uniform") == pytest.approx(0.2)

    def test_sqrt_is_one_over_root_m(self):
        assert ambiguity_penalty(0.8, 4, "sqrt") == pytest.approx(0.4)

    def test_log_form(self):
        assert ambiguity_penalty(0.8, 4, "log") == pytest.approx(0.8 / (1.0 + math.log(4)))

    def test_penalty_strength_ordering(self):
        # at m=4, uniform penalizes hardest, then log, then sqrt, then none
        s = 0.8
        u = ambiguity_penalty(s, 4, "uniform")
        lg = ambiguity_penalty(s, 4, "log")
        sq = ambiguity_penalty(s, 4, "sqrt")
        no = ambiguity_penalty(s, 4, "none")
        assert u < lg < sq < no

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            ambiguity_penalty(0.8, 4, "softmax")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
