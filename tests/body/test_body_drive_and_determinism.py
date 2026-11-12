"""Telo: oblici, nenegativnost, determinizam, izolacija povratnog tipa i pogon motornih pula."""

import numpy as np
import pytest

from neural.body import Body, BodyConfig
from neural.body.motor import MotorNeurons
from neural.environment import ChannelWorld, NDWorld, WorldConfig


def _channel_body(dims=2, seed=0):
    world = ChannelWorld(WorldConfig(dims=dims), seed)
    return Body(world, seed=seed)


@pytest.mark.parametrize("dims", [2, 3])
def test_shape_and_nonnegative_current(dims):
    body = _channel_body(dims)
    out0 = body.reset()
    assert out0.shape == (9,)
    assert np.all(out0 >= 0.0)
    rng = np.random.default_rng(1)
    for _ in range(200):
        out = body.step(rng.uniform(0.0, 6.0, 4), 1.0)
        assert out.shape == (9,)
        assert np.all(out >= 0.0)


def test_ndworld_body_shape():
    world = NDWorld(n_dims=3, n_touch=3, n_chemo=2, n_actuators=4)
    body = Body(world, seed=3)
    out = body.reset(3)
    assert out.shape == (9,)
    assert np.all(body.step(np.full(4, 2.0), 1.0) >= 0.0)


@pytest.mark.parametrize("dims", [2, 3])
def test_determinism_same_seed(dims):
    def trace(seed):
        body = _channel_body(dims, seed)
        out = [body.reset()]
        for _ in range(300):
            out.append(body.step(np.array([2.0, 0.0, 3.0, 1.0]), 1.0))
        return np.stack(out)

    assert np.array_equal(trace(7), trace(7))
    assert not np.array_equal(trace(7), trace(8))


def test_five_nA_pool0_increases_x0_and_pool1_decreases_it():
    body = _channel_body(2, 0)
    body.reset()
    x0_start = body._world.probe().x[0]
    for _ in range(1000):
        body.step(np.array([5.0, 0.0, 0.0, 0.0]), 1.0)
    x0_pool0 = body._world.probe().x[0]
    assert x0_pool0 - x0_start > 0.1

    body2 = _channel_body(2, 0)
    body2.reset()
    for _ in range(1000):
        body2.step(np.array([0.0, 5.0, 0.0, 0.0]), 1.0)
    x0_pool1 = body2._world.probe().x[0]
    assert x0_start - x0_pool1 > 0.1


def test_rheobase_and_five_nA_firing():
    rng = np.random.default_rng(0)
    neurons = MotorNeurons(rng, n_pools=4, neurons_per_pool=10)
    counts_05 = np.zeros(4)
    counts_5 = np.zeros(4)
    for _ in range(1000):
        counts_05 += neurons.step(np.array([0.5, 0.0, 0.0, 0.0]), 1.0)
    neurons.reset()
    for _ in range(1000):
        counts_5 += neurons.step(np.array([5.0, 0.0, 0.0, 0.0]), 1.0)
    assert counts_05[0] == 0.0
    assert counts_5[0] > 100.0  # prosečno > 100 spajkova po neuronu u 1000 ms


def test_step_rejects_wrong_dt_and_shape():
    body = _channel_body(2, 0)
    body.reset()
    with pytest.raises(ValueError):
        body.step(np.zeros(4), 0.5)
    with pytest.raises(ValueError):
        body.step(np.zeros(3), 1.0)


def test_isolation_return_type_is_plain_ndarray():
    body = _channel_body(2, 0)
    out = body.reset()
    assert type(out) is np.ndarray
    out2 = body.step(np.full(4, 3.0), 1.0)
    assert type(out2) is np.ndarray
    assert not hasattr(out2, "touch")
    assert not hasattr(out2, "probe")


def test_body_public_attributes_have_no_coordinates_or_dims():
    from neural.body import receptors, motor

    body = _channel_body(3, 0)
    objects = [body, body._touch, body._proprio, body._chemo, body._motor_unit]
    for obj in objects:
        for name in dir(obj):
            if name.startswith("_") or name in {"reset", "step", "transduce", "actuate"}:
                continue
            lowered = name.lower()
            assert not any(tok in lowered for tok in ("pos", "coord", "dim", "world", "probe")), (type(obj).__name__, name)
    assert receptors.G_NA == 2.0
    assert motor.C_NF == 1.0
    assert body.last_commands.shape == (4,)


def test_sensor_transduce_takes_only_observation():
    import inspect

    from neural.body.receptors import ChemoReceptor, ProprioReceptor, TouchReceptor

    for cls in (TouchReceptor, ProprioReceptor, ChemoReceptor):
        params = list(inspect.signature(cls.transduce).parameters)
        assert params == ["self", "obs"], (cls.__name__, params)


def test_sensor_works_with_fake_observation_only():
    from neural.body.receptors import TouchReceptor
    from neural.environment.types import Observation

    rec = TouchReceptor(3, dt_ms=1.0)
    obs = Observation(touch=np.array([0.2, 0.0, 0.9]), proprio=np.zeros(4), chem_gradient=np.zeros(2))
    out = rec.transduce(obs)
    assert out.shape == (3,)
    assert np.all(out >= 0.0)


def test_body_config_pool_mismatch_raises():
    with pytest.raises(ValueError):
        BodyConfig(n_pools=3, n_actuators=4)


def test_simulator_integration_is_not_available_yet():
    # (g) attach_sensor / attach_motor / attach_environment zahtevaju Simulator (neural.core),
    # koji u ovom worktree-u ne postoji. Test se preskače dok ne bude dostupan.
    pytest.importorskip("neural.core.simulator", reason="Simulator (neural.core) nije dostupan u ovom worktree-u")
