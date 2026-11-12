"""Ugovor: komponente sveta i tela zadovoljavaju Protocol-e iz neural.core.interfaces."""
import numpy as np

from neural.body import types as body_types
from neural.body.body import Body, BodyConfig
from neural.body.motor import MotorUnit
from neural.body.receptors import ChemoReceptor, ProprioReceptor, TouchReceptor
from neural.core import interfaces as I
from neural.environment import types as env_types
from neural.environment.channel_world import ChannelWorld, WorldConfig


def test_world_is_environment_and_supports_probe():
    w = ChannelWorld(WorldConfig(dims=2), seed=1)
    assert isinstance(w, I.Environment)
    assert isinstance(w, I.SupportsProbe)
    assert w.n_actuators == 4
    assert w.n_sensors == 9
    assert isinstance(w.make_probe(), I.Probe)


def test_probe_snapshot_is_dict():
    w = ChannelWorld(WorldConfig(dims=3), seed=1)
    w.reset(1)
    assert isinstance(w.make_probe().snapshot(), dict)


def test_receptors_are_sensors():
    for r in (TouchReceptor(3, 1.0), ProprioReceptor(4, 1.0), ChemoReceptor(2, 1.0)):
        assert isinstance(r, I.Sensor)
        assert isinstance(r.name, str) and r.n_out >= 1


def test_motor_unit_is_motor():
    m = MotorUnit()
    assert isinstance(m, I.Motor)
    assert m.name == "motor_unit"
    assert m.n_in == 4


def test_body_has_reset_step_and_does_not_expose_world_state():
    b = Body(ChannelWorld(WorldConfig(dims=2), seed=1), BodyConfig(), seed=1)
    assert callable(b.reset) and callable(b.step)
    out = b.reset()
    assert isinstance(out, np.ndarray) and out.shape == (9,)


def test_types_modules_are_reexports():
    for n in ("Observation", "Probe", "Environment", "SupportsProbe"):
        assert getattr(env_types, n) is getattr(I, n)
    for n in ("Sensor", "Motor", "Observation"):
        assert getattr(body_types, n) is getattr(I, n)
