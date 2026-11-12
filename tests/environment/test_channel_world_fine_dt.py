"""ChannelWorld pri finom koraku (dt=0.1 ms): oblik sonde, zaključano bekstvo, reset, bez petlje nema bekstva."""

import numpy as np
import pytest

from neural.environment import ChannelWorld, WorldConfig


@pytest.mark.parametrize("dims", [2, 3])
def test_probe_shape_after_reset(dims):
    w = ChannelWorld(WorldConfig(dims=dims), seed=1)
    w.reset()
    assert w.probe().x.shape == (dims,)


@pytest.mark.parametrize("dims", [2, 3])
def test_escaped_is_latched(dims):
    cfg = WorldConfig(dims=dims, escape_threshold_mm=1e-6, escape_hold_ms=1.0)
    w = ChannelWorld(cfg, seed=3)
    w.reset()
    rng = np.random.default_rng(0)
    latched = None
    for _ in range(300):
        w.step(rng.uniform(0, 1, 4), 0.1)
        p = w.probe()
        if latched is None:
            if p.escaped:
                latched = p.t_escape_ms
        else:
            assert p.escaped is True
            assert p.t_escape_ms == latched
    assert latched is not None, "bekstvo nije izazvano"


def test_step_without_reset_raises():
    w = ChannelWorld(WorldConfig(dims=2), seed=1)
    with pytest.raises(RuntimeError):
        w.step(np.zeros(4), 0.1)


@pytest.mark.parametrize("dims", [2, 3])
def test_constant_commands_no_escape(dims):
    w = ChannelWorld(WorldConfig(dims=dims), seed=5)
    w.reset()
    cmd = np.array([1.0, 0.0, 0.0, 0.0])
    peak = 0.0
    for _ in range(200_000):  # 20 s pri dt=0.1 ms
        w.step(cmd, 0.1)
        peak = max(peak, w.probe().x_perp_norm)
    assert peak < 0.5
    assert not w.probe().escaped
