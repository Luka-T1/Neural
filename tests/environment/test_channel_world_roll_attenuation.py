"""Roll se pokreće samo asimetrijom ko-kontrakcije i prigušuje poprečni pomak (dims=3)."""
import math

import numpy as np

from neural.environment.channel_world import ChannelWorld, WorldConfig

DT = 0.1


def _run(roll_gain, drive, n_steps, seed=3):
    w = ChannelWorld(WorldConfig(dims=3, roll_gain=roll_gain), seed)
    w.reset()
    p = w.make_probe()
    obs, phi, perp = [], [], []
    for k in range(n_steps):
        o = w.step(drive(k * DT), DT)
        s = p.snapshot()
        obs.append(np.array(o.as_array(), dtype=np.float64))
        phi.append(s["phi"])
        perp.append(s["x_perp_norm"])
    return np.array(obs), np.array(phi), np.array(perp)


def _balanced(t_ms):
    om = 2 * math.pi * 1.0 / 1000.0
    s = 0.4 * max(math.sin(om * t_ms), 0.0)
    c = 0.4 * max(math.cos(om * t_ms), 0.0)
    return np.array([0.5 + s, 0.5, 0.5 + c, 0.5])


def _asym(t_ms):
    om = 2 * math.pi * 2.0 / 1000.0
    s, c = math.sin(om * t_ms), math.cos(om * t_ms)
    return np.array([0.75 + 0.25 * s, 0.75 - 0.25 * s, 0.5 + 0.25 * c, 0.5 - 0.25 * c])


def test_balanced_cocontraction_is_bitwise_roll_invariant():
    o0, phi0, p0 = _run(0.0, _balanced, 10000)
    o1, phi1, p1 = _run(0.05, _balanced, 10000)
    assert np.all(phi1 == 0.0)
    assert np.array_equal(o0, o1)
    assert np.array_equal(p0, p1)


def test_cocontraction_asymmetry_attenuates_perp_displacement():
    _, _, p0 = _run(0.0, _asym, 20000)
    _, phi1, p1 = _run(0.05, _asym, 20000)
    assert abs(phi1[-1]) > 1.0
    m0, m1 = p0[-5000:].mean(), p1[-5000:].mean()
    assert m1 < 0.5 * m0, f"odnos {m1 / m0:.3f}"
