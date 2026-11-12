"""Body pri dt=0.1 ms (konfiguracija harnessa): oblici, determinizam, dt provera, mera šuma motora."""

import numpy as np
import pytest

from neural.body import Body, BodyConfig
from neural.core.seed import derive_seed, subsystem_rng
from neural.environment import ChannelWorld, WorldConfig


def _make(dims, seed, dt=0.1):
    world = ChannelWorld(WorldConfig(dims=dims), seed=derive_seed(seed, "world"))
    return Body(world, BodyConfig(dt_ms=dt), rng=subsystem_rng(seed, "body"))


@pytest.mark.parametrize("dims", [2, 3])
def test_shapes_finite_and_commands(dims):
    b = _make(dims, 7)
    out = b.reset()
    assert out.shape == (9,) and np.all(np.isfinite(out))
    for _ in range(2000):
        out = b.step(np.full(4, 1.0), 0.1)
        assert out.shape == (9,) and np.all(np.isfinite(out))
        assert b.last_commands.shape == (4,)
        assert np.all(b.last_commands >= 0.0)


@pytest.mark.parametrize("dims", [2, 3])
def test_bit_identical_determinism(dims):
    def run():
        b = _make(dims, 11)
        outs = [b.reset()]
        for _ in range(500):
            outs.append(b.step(np.full(4, 1.0), 0.1))
        return np.array(outs)

    assert np.array_equal(run(), run())


def test_wrong_dt_raises():
    b = _make(2, 1)
    b.reset()
    with pytest.raises(ValueError):
        b.step(np.zeros(4), 1.0)

# Mera šuma motora pri dt (varijansa napona, I=0) je u tests/body/test_motor_noise_dt.py.
