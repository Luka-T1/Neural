"""Fixture `fake_world`: lažni svet i telo samo za brz smoke test harnessa.
Ne validira fiziku sveta (stvarni svet pokriva test_integration_real.py)."""
import types

import numpy as np
import pytest


@pytest.fixture
def fake_world(monkeypatch):
    class WorldConfig:
        def __init__(self, dims=2, **kw):
            self.dims = dims
            self.tau_act_ms = 50.0

    class BodyConfig:
        def __init__(self, dt_ms=1.0, **kw):
            self.dt_ms = dt_ms

    class ChannelWorld:
        def __init__(self, cfg, seed=0):
            self.cfg = cfg
            self.x = np.zeros(cfg.dims)
            self.t = 0.0

        def make_probe(self):
            return types.SimpleNamespace(snapshot=lambda: {
                "x": self.x.copy(), "escaped": False, "t_escape_ms": None})

    class Body:
        def __init__(self, world, cfg=None, rng=None):
            self.w = world
            self.last_commands = np.zeros(4)

        def reset(self):
            self.last_commands = np.zeros(4)
            return np.zeros(9)

        def step(self, cmd, dt):
            self.last_commands = np.asarray(cmd, float).copy()
            self.w.x[0] += 1e-3 * dt * (cmd[0] - cmd[1])
            return np.concatenate([self.w.x[:1], np.abs(self.w.x[1:]), np.zeros(9 - len(self.w.x))])

    import experiments.dimension_escape.harness as h
    for n, o in dict(ChannelWorld=ChannelWorld, WorldConfig=WorldConfig, Body=Body, BodyConfig=BodyConfig).items():
        monkeypatch.setattr(h, n, o)
