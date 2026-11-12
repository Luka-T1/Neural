"""ChannelWorld: jednostrani zid (escape_sign != 0) — neutralnost dok je p ≥ −side_wall, dodirni kanal, dims=3."""
from __future__ import annotations

import numpy as np

from neural.environment.channel_world import ChannelWorld, WorldConfig


def _rev_quad(k: int) -> np.ndarray:
    # qA = sin, qB = -cos -> x_perp ide ka +n (d_area > 0)
    ph = 2 * np.pi * k / 1000.0
    return np.array([0.5 + 0.5 * np.sin(ph), 0.5 - 0.5 * np.sin(ph), 0.5 - 0.5 * np.cos(ph), 0.5 + 0.5 * np.cos(ph)])


def _quad(k: int) -> np.ndarray:
    ph = 2 * np.pi * k / 1000.0
    return np.array([0.5 + 0.5 * np.sin(ph), 0.5 - 0.5 * np.sin(ph), 0.5 + 0.5 * np.cos(ph), 0.5 - 0.5 * np.cos(ph)])


def _run(cfg: WorldConfig, seed: int, cmd, steps: int):
    w = ChannelWorld(cfg, seed)
    w.reset()
    p = w.make_probe()
    xs, touch, p_proj = [], [], []
    sign = float(cfg.escape_sign) if cfg.escape_sign != 0 else 1.0
    for k in range(steps):
        o = w.step(cmd(k), 1.0)
        s = p.snapshot()
        xs.append(s["x"].copy())
        touch.append(np.array(o.as_array(), dtype=np.float64))
        p_proj.append(sign * float(np.dot(s["x"][1:], s["n"])))
    return np.array(xs), np.array(touch), np.array(p_proj), w


def test_plus_sign_is_bitwise_neutral_while_projection_above_minus_wall():
    # Kvadratura ka +n: p ≥ 0 ceo put, zid se nikad ne aktivira -> trajektorija i opservacija su isti kao sa 0.
    steps = 3000
    x0, t0, _, _ = _run(WorldConfig(dims=2, obs_noise=0.0), 5, _rev_quad, steps)
    x1, t1, p1, _ = _run(WorldConfig(dims=2, escape_sign=1, obs_noise=0.0), 5, _rev_quad, steps)
    assert np.min(p1) >= -0.5
    assert np.array_equal(x0, x1)
    assert np.array_equal(t0, t1)


def test_touch_ventral_is_larger_with_active_wall():
    # Kvadratura ka −n uz escape_sign=+1: zid je aktivan; dodirni kanal 2 mora biti veći od golog exp(-(perp/groove)^2).
    cfg = WorldConfig(dims=2, escape_sign=1, obs_noise=0.0)
    w = ChannelWorld(cfg, 5)
    w.reset()
    p = w.make_probe()
    active = 0
    for k in range(3000):
        o = w.step(_quad(k), 1.0)
        s = p.snapshot()
        perp = float(np.linalg.norm(s["x"][1:]))
        base = np.exp(-((perp / cfg.groove_width_mm) ** 2))
        proj = s["escape_sign"] * s["x_perp_n"]
        if proj < -cfg.side_wall_mm:
            active += 1
            assert o.touch[2] > base
        else:
            assert o.touch[2] == base
    assert active > 0


def test_dims3_negative_sign_runs_and_n_stays_unit():
    w = ChannelWorld(WorldConfig(dims=3, escape_sign=-1), 11)
    w.reset()
    p = w.make_probe()
    for k in range(2000):
        w.step(_rev_quad(k), 1.0)
        s = p.snapshot()
        assert abs(np.linalg.norm(s["n"]) - 1.0) < 1e-12
        assert np.all(np.isfinite(s["x"]))
    assert s["escape_sign"] == -1


import pytest  # noqa: E402


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("d", [0.05, 0.1, 0.5, 2.0])
def test_touch_stays_in_unit_interval_inside_wall(dims, sign, d):
    cfg = WorldConfig(dims=dims, escape_sign=sign, obs_noise=0.0, sigma_perp_mm_per_sqrt_ms=0.0)
    w = ChannelWorld(cfg, 3)
    w.reset()
    w._x_perp = -sign * (cfg.side_wall_mm + d) * w._n
    o = w.step(np.full(4, 0.5), 0.1)
    assert w._wall_depth > 0.0
    t = np.array(o.touch)
    assert np.all(t >= 0.0) and np.all(t <= 1.0)


def test_touch_unchanged_without_wall():
    # escape_sign=0: bez zida, touch[2] je samo exp(-(perp/groove)^2) (grana sa zidom se ne izvršava)
    cfg = WorldConfig(dims=2, obs_noise=0.0)
    w = ChannelWorld(cfg, 3)
    w.reset()
    o = w.step(np.full(4, 0.5), 0.1)
    perp = float(np.linalg.norm(w._x_perp))
    assert o.touch[2] == pytest.approx(np.exp(-((perp / cfg.groove_width_mm) ** 2)), abs=0)
