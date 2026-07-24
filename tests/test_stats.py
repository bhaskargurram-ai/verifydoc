"""Tests for eval/stats.py: determinism, coverage behavior, sensitivity."""

import numpy as np
import pytest

from verifydoc.eval.stats import (
    bootstrap_ci,
    cluster_bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap_test,
    paired_permutation_test,
)


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


class TestClusterBootstrap:
    def test_point_matches_and_deterministic(self):
        clusters = [i // 5 for i in range(50)]  # 10 clusters of 5
        data = np.linspace(0, 1, 50)
        a = cluster_bootstrap_ci(np.mean, clusters, data, seed=3)
        b = cluster_bootstrap_ci(np.mean, clusters, data, seed=3)
        assert a.point == pytest.approx(float(np.mean(data)))
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_clustered_ci_wider_under_within_cluster_correlation(self):
        # 20 documents, 10 fields each, all fields in a doc share the SAME value
        # (maximal within-doc correlation): the effective N is 20 docs, not 200
        # fields, so the document-clustered CI must be wider than the field CI.
        rng = np.random.default_rng(0)
        clusters, vals = [], []
        for d in range(20):
            v = float(rng.integers(0, 2))
            clusters += [d] * 10
            vals += [v] * 10
        vals = np.array(vals)
        field = bootstrap_ci(np.mean, vals, seed=1)
        clust = cluster_bootstrap_ci(np.mean, clusters, vals, seed=1)
        assert (clust.hi - clust.lo) > (field.hi - field.lo)

    def test_validation(self):
        with pytest.raises(ValueError):
            cluster_bootstrap_ci(np.mean, [0, 1], [1.0])  # length mismatch


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


class TestHolmBonferroni:
    def test_empty(self):
        assert holm_bonferroni([]) == []

    def test_all_reject_when_tiny(self):
        # three highly significant p-values -> all rejected at 0.05
        assert holm_bonferroni([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]

    def test_step_down_stops(self):
        # m=3: sorted [0.01, 0.04, 0.5]; thresholds 0.05/3, 0.05/2, 0.05/1
        # 0.01<=0.0167 reject; 0.04>0.025 stop -> only smallest rejected
        assert holm_bonferroni([0.01, 0.04, 0.5], alpha=0.05) == [True, False, False]

    def test_order_preserved(self):
        # reject flags align to INPUT order, not sorted order
        assert holm_bonferroni([0.5, 0.001], alpha=0.05) == [False, True]

    def test_more_conservative_than_uncorrected(self):
        # a p just under alpha alone is not rejected once it's one of many
        assert holm_bonferroni([0.04, 0.04, 0.04], alpha=0.05) == [False, False, False]


class TestInterAnnotatorAgreement:
    def test_cohens_kappa_perfect(self):
        from verifydoc.eval.stats import cohens_kappa

        assert cohens_kappa([1, 0, 1, 1], [1, 0, 1, 1]) == pytest.approx(1.0)

    def test_cohens_kappa_hand_computed(self):
        from verifydoc.eval.stats import cohens_kappa

        # a=[1,1,0,0], b=[1,0,0,0]: p_o=3/4; marginals a:2/4,2/4 b:1/4,3/4
        # p_e = .5*.25 + .5*.75 = .5; kappa=(.75-.5)/(1-.5)=.5
        assert cohens_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)

    def test_cohens_kappa_chance(self):
        from verifydoc.eval.stats import cohens_kappa

        # independent alternating labels -> near 0
        assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
        assert cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)

    def test_cohens_kappa_constant_labels(self):
        from verifydoc.eval.stats import cohens_kappa

        assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0
        assert cohens_kappa([1, 1, 1], [1, 1, 0]) == pytest.approx(0.0)

    def test_cohens_kappa_validation(self):
        from verifydoc.eval.stats import cohens_kappa

        with pytest.raises(ValueError):
            cohens_kappa([1], [1, 0])
        with pytest.raises(ValueError):
            cohens_kappa([], [])

    def test_fleiss_kappa_perfect(self):
        from verifydoc.eval.stats import fleiss_kappa

        # 3 items, 3 raters, all agree -> kappa 1
        assert fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == pytest.approx(1.0)

    def test_fleiss_kappa_range(self):
        from verifydoc.eval.stats import fleiss_kappa

        # mixed ratings give kappa in [-1, 1]
        k = fleiss_kappa([[2, 1], [1, 2], [3, 0], [0, 3]])
        assert -1.0 <= k <= 1.0

    def test_fleiss_kappa_validation(self):
        from verifydoc.eval.stats import fleiss_kappa

        with pytest.raises(ValueError):
            fleiss_kappa([[3, 0], [2, 0]])  # unequal rater counts
        with pytest.raises(ValueError):
            fleiss_kappa([[1, 0]])  # <2 raters
