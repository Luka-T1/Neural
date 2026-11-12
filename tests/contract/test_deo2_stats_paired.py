"""Ugovor: uparena statistika Dela 2 (stats.py) naspram scipy referenci i poznatih brojeva.

scipy se koristi samo ovde, kao referenca; implementacija u stats.py ne uvozi scipy.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.dimension_escape.stats import (cohen_dz, spearman, theil_sen_slope,
                                                wilcoxon_signed_rank)

scipy_stats = pytest.importorskip("scipy.stats")


def _rng_pairs(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    return x, y


@pytest.mark.parametrize("seed,n", [(0, 6), (1, 12), (2, 25), (3, 50)])
def test_wilcoxon_exact_matches_scipy(seed, n):
    x, y = _rng_pairs(seed, n)
    _, p_ours = wilcoxon_signed_rank(x, y)
    res = scipy_stats.wilcoxon(x, y, method="exact")
    assert math.isclose(p_ours, float(res.pvalue), rel_tol=1e-9, abs_tol=1e-12)


def test_wilcoxon_stat_is_min_of_w_plus_w_minus():
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    stat, _ = wilcoxon_signed_rank(x, np.zeros(5))
    res = scipy_stats.wilcoxon(x, np.zeros(5), method="exact")
    assert stat == float(res.statistic) == 6.0


@pytest.mark.parametrize("seed,n", [(4, 51), (5, 60), (6, 90)])  # n > 50: normalna aproksimacija
def test_wilcoxon_approx_matches_scipy_no_continuity(seed, n):
    x, y = _rng_pairs(seed, n)
    _, p_ours = wilcoxon_signed_rank(x, y)
    res = scipy_stats.wilcoxon(x, y, method="approx", correction=False)
    assert math.isclose(p_ours, float(res.pvalue), rel_tol=1e-7, abs_tol=1e-10)


def test_wilcoxon_ties_match_scipy_approx():
    x = np.array([2.0, 2.0, -2.0, 5.0, 5.0, 5.0, -5.0, 1.0, 3.0, 3.0])
    y = np.zeros_like(x)
    _, p_ours = wilcoxon_signed_rank(x, y)
    res = scipy_stats.wilcoxon(x, y, method="approx", correction=False)
    assert math.isclose(p_ours, float(res.pvalue), rel_tol=1e-7, abs_tol=1e-10)


def test_wilcoxon_known_value_all_positive():
    _, p = wilcoxon_signed_rank([1, 2, 3, 4, 5], [0] * 5)
    assert abs(p - 0.0625) < 1e-12


def test_cohen_dz_matches_manual_formula():
    x, y = _rng_pairs(7, 40)
    d = x - y
    assert math.isclose(cohen_dz(x, y), float(d.mean() / d.std(ddof=1)), rel_tol=1e-12)


def test_theil_sen_matches_scipy_median_slope():
    rng = np.random.default_rng(8)
    t = np.arange(20.0)
    y = 0.3 * t + rng.normal(scale=0.5, size=20)
    ours = theil_sen_slope(t, y)
    ref = scipy_stats.theilslopes(y, t)[0]
    assert math.isclose(ours, float(ref), rel_tol=1e-12, abs_tol=1e-12)


def test_spearman_matches_scipy_rho():
    x, y = _rng_pairs(9, 30)
    assert math.isclose(spearman(x, y), float(scipy_stats.spearmanr(x, y)[0]), rel_tol=1e-9)


def test_outputs_are_plain_floats_for_json():
    import json

    x, y = _rng_pairs(10, 8)
    out = [wilcoxon_signed_rank(x, y), cohen_dz(x, y), theil_sen_slope(np.arange(8.0), y)]
    flat = [v for item in out for v in (item if isinstance(item, tuple) else (item,))]
    assert all(type(v) is float for v in flat)
    json.dumps(flat)
