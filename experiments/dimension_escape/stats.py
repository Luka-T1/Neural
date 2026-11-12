"""Statistika bez obaveznog scipy-ja: Wilson, Fisher (2x2), Mann-Whitney U, Holm, Spearman.

Uparena statistika (faza 2, dopuna Dela 2): `wilcoxon_signed_rank` (d = x − y, nule se odbacuju,
prosečni rangovi |d|, stat = min(W+, W−); egzaktna dvostrana p za n_eff ≤ 50 bez veza, inače normalna
aproksimacija sa korekcijom za veze i bez korekcije za kontinuitet), `hodges_lehmann_paired` (medijana
Walsh proseka (d_i + d_j)/2, i ≤ j, uz nule; CI iz egzaktne T raspodele za n ≤ 50, normalna aproksimacija
inače; ako nema validnog k, k = 1), `cohen_dz` (mean(d)/sd(d, ddof=1)), `theil_sen_slope` (medijana nagiba
svih parova sa t_i ≠ t_j) i `bootstrap_spearman_ci` (bootstrap po grupama, percentili 2.5 i 97.5).
Ulazi su liste ili ndarray; izlazi su obični float (bezbedno za json)."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Dvostrani egzaktni test za [[a,b],[c,d]] (hipergeometrijski; zbir verovatnoća <= p_obs)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    if n == 0:
        return 1.0
    lo, hi = max(0, c1 - r2), min(r1, c1)

    def logp(x):
        return _log_comb(r1, x) + _log_comb(r2, c1 - x) - _log_comb(n, c1)

    lp_obs = logp(a)
    tot = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= lp_obs + 1e-9:
            tot += math.exp(lp)
    return float(min(1.0, max(0.0, tot)))


@lru_cache(maxsize=None)
def _u_counts(m: int, n: int) -> tuple:
    """Broj rasporeda za svaku vrednost U (egzaktna raspodela bez veza)."""
    if m == 0 or n == 0:
        return (1,)
    a = _u_counts(m - 1, n)
    b = _u_counts(m, n - 1)
    out = [0] * (m * n + 1)
    for u, v in enumerate(a):
        out[u + n] += v
    for u, v in enumerate(b):
        out[u] += v
    return tuple(out)


def _rank_average(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def mann_whitney_u(x, y) -> tuple[float, float]:
    """(U_x, p dvostrani). Egzaktno za male uzorke bez veza, inače normalna aproksimacija
    sa korekcijom za veze i za kontinuitet."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return (float("nan"), 1.0)
    allv = np.concatenate([x, y])
    ranks = _rank_average(allv)
    u1 = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    has_ties = len(np.unique(allv)) < len(allv)
    if not has_ties and n1 * n2 <= 400 and n1 <= 20 and n2 <= 20:
        counts = np.array(_u_counts(n1, n2), float)
        probs = counts / counts.sum()
        u_obs = int(round(u1))
        lower = probs[: u_obs + 1].sum()
        upper = probs[u_obs:].sum()
        return (u1, float(min(1.0, 2 * min(lower, upper))))
    n = n1 + n2
    _, tc = np.unique(allv, return_counts=True)
    tie = float((tc ** 3 - tc).sum())
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return (u1, 1.0)
    mu = n1 * n2 / 2.0
    diff = abs(u1 - mu)
    z = max(0.0, diff - 0.5) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2.0))
    return (u1, float(min(1.0, max(0.0, p))))


def holm(pvals) -> list[float]:
    p = list(map(float, pvals))
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx, ry = _rank_average(x), _rank_average(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


# ------------------------------------------------------------ uparena statistika (Deo 2)
_EXACT_MAX_N = 50


@lru_cache(maxsize=None)
def _signed_rank_counts(n: int) -> tuple:
    """counts[w] = broj podskupova {1..n} čiji je zbir w (egzaktna raspodela W+ bez veza)."""
    m = n * (n + 1) // 2
    counts = [0] * (m + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(m, r - 1, -1):
            counts[s] += counts[s - r]
    return tuple(counts)


def _cdf_exact(n: int, w: int) -> float:
    """P(W+ <= w) za n ranga bez veza."""
    counts = _signed_rank_counts(n)
    if w < 0:
        return 0.0
    if w >= len(counts) - 1:
        return 1.0
    return float(sum(counts[: w + 1]) / 2 ** n)


def _z_two_sided(alpha: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilcoxon_signed_rank(x, y) -> tuple[float, float]:
    """Wilcoxon test sa znakom ranga za uparene uzorke. Vraća (stat, p), stat = min(W+, W−)."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError("x i y moraju imati istu dužinu")
    d = x - y
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return (0.0, 1.0)
    absd = np.abs(d)
    ranks = _rank_average(absd)
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    stat = min(w_plus, w_minus)
    _, tc = np.unique(absd, return_counts=True)
    has_ties = bool(np.any(tc > 1))
    if n <= _EXACT_MAX_N and not has_ties:
        p = min(1.0, 2.0 * _cdf_exact(n, int(round(stat))))
        return (stat, float(max(0.0, p)))
    tie = float((tc.astype(float) ** 3 - tc).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0.0:
        return (stat, 1.0)
    z = (stat - n * (n + 1) / 4.0) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return (stat, float(min(1.0, max(0.0, p))))


def hodges_lehmann_paired(x, y, alpha: float = 0.05) -> tuple[float, float, float]:
    """Hodges–Lehmann procena pomaka za uparene uzorke: (est, lo, hi). Nule se NE odbacuju."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.shape != y.shape:
        raise ValueError("x i y moraju imati istu dužinu")
    d = x - y
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    m = len(walsh)
    est = float(np.median(walsh))
    if n <= _EXACT_MAX_N:
        k = 1
        for cand in range(1, m + 1):
            if _cdf_exact(n, cand - 1) <= alpha / 2.0:
                k = cand
            else:
                break
    else:
        mean = n * (n + 1) / 4.0
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        k = int(math.floor(mean - _z_two_sided(alpha) * sd + 0.5))
        k = min(max(k, 1), m)
    lo = float(walsh[k - 1])
    hi = float(walsh[m - k])
    return (est, lo, hi)


def cohen_dz(x, y) -> float:
    """Cohenov dz za uparene uzorke: mean(d) / sd(d, ddof=1). nan ako je n < 2 ili sd = 0."""
    d = np.asarray(x, float).ravel() - np.asarray(y, float).ravel()
    if len(d) < 2:
        return float("nan")
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(d.mean() / sd)


def theil_sen_slope(t, y) -> float:
    """Theil–Sen nagib: medijana nagiba svih parova sa t_i ≠ t_j. nan ako ima manje od 2 različita t."""
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    if t.shape != y.shape:
        raise ValueError("t i y moraju imati istu dužinu")
    if len(np.unique(t)) < 2:
        return float("nan")
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    keep = dt != 0.0
    slopes = (y[j] - y[i])[keep] / dt[keep]
    return float(np.median(slopes))


def bootstrap_spearman_ci(x, y, groups, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(rho, lo, hi): rho na svim podacima; bootstrap uzorkuje grupe sa vraćanjem; percentili 2.5 i 97.5.
    nan vrednosti bootstrap replika se ignorišu."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    g = np.asarray(groups).ravel()
    if not (len(x) == len(y) == len(g)):
        raise ValueError("x, y i groups moraju imati istu dužinu")
    rho = spearman(x, y)
    ug = np.unique(g)
    members = [np.flatnonzero(g == u) for u in ug]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick])
        vals.append(spearman(x[idx], y[idx]))
    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float(rho), float("nan"), float("nan"))
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return (float(rho), float(lo), float(hi))
