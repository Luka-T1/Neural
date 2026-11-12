"""Ugovor: organizam vidi samo Observation; koordinate i dims samo kroz probe."""
import dataclasses
import os

import numpy as np
import pytest

from neural.core.interfaces import Observation
from neural.environment.channel_world import ChannelWorld, WorldConfig


PHASE3 = pytest.mark.skipif(os.environ.get('NEURAL_REQUIRE_PHASE3') != '1', reason='faza 3')
ROLL = [0.0, pytest.param(0.05, marks=PHASE3)]


def _obs(dims, roll_gain=0.0):
    cfg = WorldConfig(dims=dims) if roll_gain == 0.0 else WorldConfig(dims=dims, roll_gain=roll_gain)
    w = ChannelWorld(cfg, seed=3)
    w.reset()
    return w, w.step(np.full(4, 0.5))


def test_observation_fields_only():
    assert [f.name for f in dataclasses.fields(Observation)] == ["touch", "proprio", "chem_gradient"]
    _, o = _obs(2)
    public = {n for n in dir(o) if not n.startswith("_")}
    assert public == {"touch", "proprio", "chem_gradient", "as_array"}


@pytest.mark.parametrize('roll_gain', ROLL)
def test_observation_read_only(roll_gain):
    _, o = _obs(3, roll_gain)
    for f in ("touch", "proprio", "chem_gradient"):
        a = getattr(o, f)
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.touch = np.zeros(3)


def test_as_array_shape_independent_of_dims():
    shapes = {d: _obs(d)[1].as_array().shape for d in (2, 3, 4)}
    assert len(set(shapes.values())) == 1
    assert shapes[2] == (9,)


@pytest.mark.parametrize('roll_gain', ROLL)
def test_observation_has_no_coordinates_or_dims(roll_gain):
    _, o = _obs(3, roll_gain)
    for n in ("x", "dims", "escaped", "t_escape_ms", "x_perp", "phi", "roll", "n", "axis"):
        assert not hasattr(o, n)


@pytest.mark.parametrize('roll_gain', ROLL)
def test_world_state_only_via_probe_snapshot(roll_gain):
    w, _ = _obs(3, roll_gain)
    snap = w.make_probe().snapshot()
    for k in ("x", "dims", "escaped", "t_escape_ms"):
        assert k in snap
    assert snap["dims"] == 3 and snap["x"].shape == (3,)
