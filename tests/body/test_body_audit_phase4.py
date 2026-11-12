"""Audit tela (faza 4): komande u [0,1] pre aktivacija, receptori bez znaka x_perp, reset reproducibilan."""
from __future__ import annotations

import numpy as np

from neural.body import Body, BodyConfig
from neural.environment import ChannelWorld, WorldConfig


class _SpyWorld:
    """Omotač oko ChannelWorld koji beleži komande koje telo prosleđuje svetu."""

    def __init__(self, inner: ChannelWorld) -> None:
        self._inner = inner
        self.received: list[np.ndarray] = []

    def reset(self, *a, **k):
        return self._inner.reset(*a, **k)

    def step(self, commands, dt_ms):
        self.received.append(np.array(commands, dtype=np.float64, copy=True))
        return self._inner.step(commands, dt_ms)


def test_commands_clipped_to_unit_interval_before_world():
    spy = _SpyWorld(ChannelWorld(WorldConfig(dims=2), 1))
    body = Body(spy, seed=1)
    body.reset()
    rng = np.random.default_rng(2)
    for _ in range(300):
        body.step(rng.uniform(-1000.0, 1000.0, 4), 1.0)
    cmds = np.stack(spy.received)
    assert np.all(cmds >= 0.0) and np.all(cmds <= 1.0)
    assert np.all(body.last_commands >= 0.0) and np.all(body.last_commands <= 1.0)


def test_receptor_inputs_do_not_carry_sign_of_x_perp():
    # Dodir (ventralni kanal), propriocepcija i hemija zavise samo od ‖x_perp‖, ne od znaka.
    w = ChannelWorld(WorldConfig(dims=3, obs_noise=0.0), 4)
    w.reset()
    v = np.array([0.3, -0.2])
    w._x_perp = v.copy()
    o_pos = w._observe(np.zeros(9))
    w._x_perp = -v
    o_neg = w._observe(np.zeros(9))
    assert np.array_equal(o_pos.touch, o_neg.touch)
    assert np.array_equal(o_pos.proprio, o_neg.proprio)
    assert np.array_equal(o_pos.chem_gradient, o_neg.chem_gradient)


def test_reset_returns_same_vector_after_episode():
    body = Body(ChannelWorld(WorldConfig(dims=2), 3), seed=3)
    first = body.reset().copy()
    rng = np.random.default_rng(0)
    for _ in range(200):
        body.step(rng.uniform(0.0, 6.0, 4), 1.0)
    again = body.reset()
    assert np.array_equal(first, again)
    assert again.dtype == np.float64 and again.shape == (9,)
