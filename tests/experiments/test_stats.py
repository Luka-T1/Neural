import math

import numpy as np
import pytest
from experiments.dimension_escape.stats import (bootstrap_spearman_ci, cohen_dz, fisher_exact_2x2,
                                                holm, hodges_lehmann_paired, mann_whitney_u,
                                                spearman, theil_sen_slope, wilcoxon_signed_rank,
                                                wilson_interval)


def test_wilson():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = wilson_interval(10, 20)
    assert abs((lo + hi) / 2 - 0.5) < 1e-9
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_fisher_known_values():
    assert abs(fisher_exact_2x2(3, 1, 1, 3) - 0.4857142857) < 1e-8
    assert abs(fisher_exact_2x2(10, 0, 0, 10) - 2 / math.comb(20, 10)) < 1e-12
    assert abs(fisher_exact_2x2(5, 5, 5, 5) - 1.0) < 1e-9
    assert abs(fisher_exact_2x2(0, 20, 0, 20) - 1.0) < 1e-9


def test_mwu_exact_and_ties():
    u, p = mann_whitney_u([1, 2, 3], [4, 5, 6])
    assert u == 0.0 and abs(p - 0.1) < 1e-12
    u, p = mann_whitney_u([5, 6, 7], [1, 2, 3])
    assert u == 9.0 and abs(p - 0.1) < 1e-12
    u, p = mann_whitney_u([21] * 10, [21] * 10)
    assert p == 1.0
    u, p = mann_whitney_u([1, 1, 2, 2] * 5, [3, 3, 4, 4] * 5)
    assert 0 <= p < 1e-3
    for x, y in (([1], [2]), ([1, 2], [1, 2])):
        assert 0.0 <= mann_whitney_u(x, y)[1] <= 1.0


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm([0.5, 0.9, 1.0]) == [1.0, 1.0, 1.0]


def test_spearman():
    assert abs(spearman([1, 2, 3, 4], [2, 4, 6, 9]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


# ---------------------------------------------------------------- uparena statistika (Deo 2)

def test_wilcoxon_exact_all_positive_small_n():
    # d = 1..5: W+ = 15, W- = 0; P(W+ <= 0) = 1/32; p = 2/32
    stat, p = wilcoxon_signed_rank([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert stat == 0.0 and abs(p - 0.0625) < 1e-12


def test_wilcoxon_exact_mixed_signs():
    # |d| = 1..5, znaci + - + - +: W+ = 9, W- = 6, stat = 6; P(W <= 6) = 13/32
    stat, p = wilcoxon_signed_rank([1, -2, 3, -4, 5], [0, 0, 0, 0, 0])
    assert stat == 6.0 and abs(p - 2 * 13 / 32) < 1e-12


def test_wilcoxon_zeros_are_dropped_and_empty_gives_one():
    assert wilcoxon_signed_rank([1, 2, 3], [1, 2, 3]) == (0.0, 1.0)
    stat, p = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 6])
    assert stat == 0.0 and abs(p - 0.0625) < 1e-12  # n_eff = 5 (nula odbačena)
    assert wilcoxon_signed_rank([3.0], [1.0]) == (0.0, 1.0)  # n_eff = 1: p = 2 * 1/2 = 1


def test_wilcoxon_ties_use_normal_approx_and_p_in_unit_interval():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [0, 0, 0, 0, 0, 0, 0, 0]
    stat, p = wilcoxon_signed_rank(x, y)
    assert 0.0 <= p <= 1.0 and isinstance(stat, float) and isinstance(p, float)
    d = [2, 2, -2, 5, 5, 5, -5, 1]
    _, p_ties = wilcoxon_signed_rank(d, [0] * len(d))
    assert 0.0 <= p_ties <= 1.0


def test_wilcoxon_large_n_normal_no_continuity_and_bounds():
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = rng.normal(size=80)
    stat, p = wilcoxon_signed_rank(x, y)
    assert 0.0 <= p <= 1.0 and stat >= 0.0
    assert wilcoxon_signed_rank(list(x), list(y)) == (stat, p)  # liste i ndarray daju isto


def test_wilcoxon_length_mismatch_raises():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1, 2], [1])


def test_hodges_lehmann_known_median_and_exact_ci():
    # d = 1..5: Walsh proseci imaju medijanu 3; T raspodela za n=5 nema validan k, pa k=1
    est, lo, hi = hodges_lehmann_paired([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert est == 3.0 and lo == 1.0 and hi == 5.0
    assert all(isinstance(v, float) for v in (est, lo, hi))


def test_hodges_lehmann_ci_from_walsh_order_statistics():
    d = np.arange(1.0, 8.0)  # n = 7: P(T<=1)=2/128, P(T<=2)=3/128 <= 0.025 < P(T<=3)=5/128 => k = 3
    walsh = np.sort([(d[i] + d[j]) / 2 for i in range(7) for j in range(i, 7)])
    est, lo, hi = hodges_lehmann_paired(d, np.zeros(7))
    assert est == float(np.median(walsh))
    assert lo == walsh[2] and hi == walsh[-3]


def test_hodges_lehmann_empty_and_single():
    assert all(np.isnan(v) for v in hodges_lehmann_paired([], []))
    assert hodges_lehmann_paired([2.0], [0.5]) == (1.5, 1.5, 1.5)


def test_cohen_dz_known_and_degenerate():
    # d = 1, 3, 5: mean 3, sd (ddof=1) = 2, dz = 1.5
    assert abs(cohen_dz([2, 4, 6], [1, 1, 1]) - 1.5) < 1e-12
    assert np.isnan(cohen_dz([2, 3], [1, 2]))  # sd = 0
    assert np.isnan(cohen_dz([2], [1]))        # n < 2


def test_theil_sen_known_and_degenerate():
    assert theil_sen_slope([0, 1, 2, 3], [1, 3, 5, 7]) == 2.0
    # valjani parovi: (0,2)=-2, (0,3)=0, (1,2)=-2, (1,3)=0, (2,3)=2 -> medijana 0
    assert theil_sen_slope([0, 0, 1, 2], [5, 5, 3, 5]) == 0.0
    assert np.isnan(theil_sen_slope([1, 1, 1], [1, 2, 3]))


def test_bootstrap_spearman_monotone_and_deterministic():
    x = np.arange(10.0)
    groups = np.arange(10)
    rho, lo, hi = bootstrap_spearman_ci(x, x, groups, n_boot=200, seed=0)
    assert rho == pytest.approx(1.0) and lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    y = np.array([0.0, 2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 9.0])
    a = bootstrap_spearman_ci(x, y, groups, n_boot=300, seed=7)
    b = bootstrap_spearman_ci(x, y, groups, n_boot=300, seed=7)
    assert a == b and all(isinstance(v, float) for v in a)
    assert a[1] <= a[0] <= a[2]


def test_bootstrap_spearman_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_spearman_ci([1, 2, 3], [1, 2, 3], [0, 1], n_boot=5)
