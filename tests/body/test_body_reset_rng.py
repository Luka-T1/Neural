"""Body.reset() vraća motorni RNG na početno stanje: epizode sa istim seedom su bit-identične."""
from __future__ import annotations

import numpy as np
import pytest

from neural.body import Body, BodyConfig
from neural.core.seed import subsystem_rng
from neural.environment import ChannelWorld, WorldConfig


def _episode(body: Body, steps: int = 3000):
    aff = [body.reset().copy()]
    cmds = []
    pm = np.full(4, 0.8)
    for _ in range(steps):
        aff.append(body.step(pm, 0.1).copy())
        cmds.append(body.last_commands.copy())
    return np.array(cmds), np.array(aff)


def _make(**kw):
    world = ChannelWorld(WorldConfig(dims=2), seed=1)
    return Body(world, BodyConfig(dt_ms=0.1), **kw)


@pytest.mark.parametrize("kind", ["seed", "rng"])
def test_two_episodes_bit_identical(kind):
    kw = {"seed": 7} if kind == "seed" else {"rng": subsystem_rng(5, "body")}
    b = _make(**kw)
    c1, a1 = _episode(b)
    c2, a2 = _episode(b)
    assert np.array_equal(c1, c2) and np.array_equal(a1, a2)


def test_motor_noise_is_used():
    c1, _ = _episode(_make(seed=1), 1500)
    c2, _ = _episode(_make(seed=2), 1500)
    assert not np.array_equal(c1, c2)
