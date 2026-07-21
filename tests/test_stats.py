"""Tests for eval/stats.py: determinism, coverage behavior, sensitivity."""

import numpy as np
import pytest

from verifydoc.eval.stats import bootstrap_ci, paired_bootstrap_test, paired_permutation_test


class TestBootstrapCI:
    def test_constant_data_degenerate_ci(self):
        res = bootstrap_ci(np.mean, [1.0] * 20)
        assert res.point == res.lo == res.hi == 1.0

    def test_deterministic_given_seed(self):
        data = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1] * 5
        a = bootstrap_ci(np.mean, data, seed=7)
        b = bootstrap_ci(np.mean, data, seed=7)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_ci_contains_point(self):
        rng = np.random.default_rng(0)
        data = rng.random(200)
        res = bootstrap_ci(np.mean, data)
        assert res.lo <= res.point <= res.hi
        assert res.hi - res.lo < 0.2

    def test_joint_resampling_keeps_pairs_aligned(self):
        conf = np.linspace(0.1, 0.9, 50)
        correct = (conf > 0.5).astype(float)

        def frac_agree(c, y):
            return float(np.mean((c > 0.5) == (y == 1.0)))

        res = bootstrap_ci(frac_agree, conf, correct)
        assert res.point == res.lo == res.hi == 1.0  # alignment preserved

    def test_validation(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.mean, [1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            bootstrap_ci(np.mean, [])


class TestPairedTests:
    def test_identical_systems_not_significant(self):
        scores = [1, 0, 1, 1, 0, 1, 0, 1] * 4
        assert paired_permutation_test(scores, scores) == pytest.approx(1.0)

    def test_clear_difference_significant(self):
        a = [1] * 40
        b = [0] * 30 + [1] * 10
        assert paired_permutation_test(a, b) < 0.01
        assert paired_bootstrap_test(a, b) < 0.01

    def test_deterministic(self):
        a = [1, 0, 1, 1, 1, 0, 1, 1, 0, 1] * 3
        b = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1] * 3
        assert paired_permutation_test(a, b, seed=3) == paired_permutation_test(a, b, seed=3)

    def test_validation(self):
        with pytest.raises(ValueError):
            paired_permutation_test([1], [1, 0])
        with pytest.raises(ValueError):
            paired_bootstrap_test([], [])
