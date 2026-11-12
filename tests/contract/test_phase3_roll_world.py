"""Ugovor faze 3 za svet: roll_gain ne dira aksijalnu dinamiku, φ i n vidljivi samo kroz probe().snapshot()."""
import math

import numpy as np

from neural.environment.channel_world import ChannelWorld, WorldConfig


def _run(g, steps=1500):
    w = ChannelWorld(WorldConfig(dims=3, roll_gain=g), 7)
    w.reset()
    p = w.make_probe()
    n0 = p.snapshot()["n"].copy()
    rows, xs, snaps = [], [], []
    for k in range(steps):
        a = np.array([1.0, 1.0, 0.0, 0.0]) if k < steps // 2 else np.array([0.6, 0.4, 0.5, 0.5])
        o = w.step(a)
        rows.append((o.touch[0:2].copy(), o.proprio.copy(), o.chem_gradient.copy()))
        xs.append(w.probe().x[0])
        snaps.append(p.snapshot())
    return n0, rows, xs, snaps


def test_roll_zero_has_no_phi_or_drift():
    n0, _, _, snaps = _run(0.0, 300)
    for s in snaps:
        assert s["phi"] == 0.0
        np.testing.assert_array_equal(s["n"], n0)


def test_roll_keeps_axial_and_sensors_bit_identical():
    _, r0, x0, _ = _run(0.0)
    _, r1, x1, s1 = _run(0.05)
    assert x0 == x1
    for (t0, p0, c0), (t1, p1, c1) in zip(r0, r1):
        np.testing.assert_array_equal(t0, t1)
        np.testing.assert_array_equal(p0, p1)
        np.testing.assert_array_equal(c0, c1)
    assert any(s["phi"] != 0.0 for s in s1)


def test_n_is_rotation_of_initial_n():
    n0, _, _, snaps = _run(0.05)
    for s in snaps:
        c, sn = math.cos(s["phi"]), math.sin(s["phi"])
        np.testing.assert_allclose(s["n"], [c * n0[0] - sn * n0[1], sn * n0[0] + c * n0[1]], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(s["n"]), 1.0, atol=1e-12)
