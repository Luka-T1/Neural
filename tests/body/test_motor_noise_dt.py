"""Motorni šum se skalira sa sqrt(REF_DT_MS/dt); pri dt=0.1 ms korak je bit-identičan starom."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neural.body.motor import NOISE_NA, REF_DT_MS, MotorNeurons, noise_scale


def test_noise_scale_reference_and_value():
    assert noise_scale(0.1) == 1.0
    assert noise_scale(REF_DT_MS) == 1.0
    assert noise_scale(0.025) == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf"), -float("inf")])
def test_noise_scale_rejects_invalid_dt(bad):
    with pytest.raises(ValueError):
        noise_scale(bad)


class _OldMotorNeurons:
    """Referentna kopija starog koraka (pre skaliranja šuma), ista logika ostatka."""

    def __init__(self, rng, n_pools, n):
        self._rng = rng
        self._n_pools = n_pools
        self._n = n
        self._v = np.full((n_pools, n), -65.0)
        self._refr = np.zeros((n_pools, n))

    def step(self, I, dt_ms):
        I = np.asarray(I, dtype=np.float64)
        noisy = I[:, None] + NOISE_NA * self._rng.standard_normal((self._n_pools, self._n))
        active = self._refr <= 0.0
        dv = dt_ms * (-50.0 * (self._v - (-65.0)) * 1e-3 + noisy) / 1.0
        v = np.where(active, self._v + dv, self._v)
        spiked = active & (v >= -50.0)
        v = np.where(spiked, -65.0, v)
        self._refr = np.where(spiked, 2.0, np.maximum(self._refr - dt_ms, 0.0))
        self._v = v
        return spiked.sum(axis=1) / self._n


def test_bit_identical_to_old_step_at_reference_dt():
    seed = 42
    new = MotorNeurons(np.random.default_rng(seed), 4, 10)
    old = _OldMotorNeurons(np.random.default_rng(seed), 4, 10)
    cur = np.random.default_rng(7)
    for k in range(3000):
        I = 0.5 + 1.3 * cur.random(4)
        a = new.step(I, 0.1)
        b = old.step(I, 0.1)
        assert np.array_equal(a, b), f"spajkovi se razlikuju u koraku {k}"
    assert np.array_equal(new.v, old._v)


def test_variance_ratio_across_dt_at_zero_current():
    def stationary_var(dt):
        mn = MotorNeurons(np.random.default_rng(3), 4, 10)
        steps_per_ms = int(round(1.0 / dt))
        total_steps = int(round(2200.0 / dt))
        burn = int(round(200.0 / dt))
        samples = []
        for k in range(total_steps):
            mn.step(np.zeros(4), dt)
            if k >= burn and (k + 1) % steps_per_ms == 0:
                samples.append(mn.v.ravel())
        return float(np.var(np.concatenate(samples)))

    v_fine = stationary_var(0.1)
    v_fine2 = stationary_var(0.025)
    ratio = v_fine2 / v_fine
    print(f"\nvarijansa v pri I=0: dt=0.1 -> {v_fine}, dt=0.025 -> {v_fine2}, odnos={ratio}")
    assert 0.8 < ratio < 1.25
    assert math.isfinite(v_fine) and v_fine > 0
