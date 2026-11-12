"""Izolacija i determinizam NDWorld-a (uputstvo arhitekture, stavke a–f)."""

import inspect

import numpy as np
import pytest

from neural.environment import NDWorld, Observation
from neural.environment.types import Environment, SupportsProbe

FORBIDDEN = ("pos", "coord", "dim", "probe", "world", "perp", "escape")


def _make(n_dims, seed_world=None):
    return NDWorld(n_dims=n_dims, n_touch=3, n_chemo=2, n_actuators=4)


def _actions(n, seed=123):
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=(n, 4))


def _trace(world, seed, actions):
    obs = [world.reset(seed).as_array()]
    for a in actions:
        obs.append(world.step(a).as_array())
    return np.stack(obs)


# (a) isti seed -> bit-identične opservacije; drugi seed -> različite
def test_same_seed_bit_identical_100_steps():
    acts = _actions(100)
    a = _trace(_make(2), 7, acts)
    b = _trace(_make(2), 7, acts)
    assert np.array_equal(a, b)


def test_different_seed_differs():
    acts = _actions(100)
    a = _trace(_make(2), 7, acts)
    b = _trace(_make(2), 8, acts)
    assert not np.array_equal(a, b)


# (b) oblici polja nezavisni od n_dims
@pytest.mark.parametrize("n_dims", [2, 3, 5])
def test_shapes_independent_of_n_dims(n_dims):
    w = _make(n_dims)
    obs = w.reset(1)
    assert isinstance(obs, Observation)
    assert obs.touch.shape == (3,)
    assert obs.proprio.shape == (4,)
    assert obs.chem_gradient.shape == (2,)
    assert obs.as_array().shape == (9,)
    nxt = w.step(np.full(4, 0.5))
    assert nxt.as_array().shape == (9,)


# (c) javni atributi bez koordinata / dimenzija / sveta
def test_public_attributes_are_only_counts():
    w = _make(3)
    public = {n for n in dir(w) if not n.startswith("_")}
    assert {"n_actuators", "n_sensors"} <= public
    for name in public:
        if name in {"reset", "step", "make_probe", "CONFIDENCE", "PARAMS_META", "n_actuators", "n_sensors"}:
            continue
        if name.startswith("__"):
            continue
        assert not any(tok in name.lower() for tok in ("pos", "coord", "dim", "world")), name
    assert w.n_actuators == 4
    assert w.n_sensors == 9


def test_observation_has_no_forbidden_public_names():
    names = {f for f in Observation.__dataclass_fields__} | {n for n in dir(Observation) if not n.startswith("_")}
    for name in names:
        assert not any(tok in name.lower() for tok in FORBIDDEN), name


# (e) isinstance(env, Environment) i tačan n_dims u probe-u
@pytest.mark.parametrize("n_dims", [2, 3, 5])
def test_is_environment_and_probe_reports_n_dims(n_dims):
    w = _make(n_dims)
    assert isinstance(w, Environment)
    assert isinstance(w, SupportsProbe)
    w.reset(4)
    snap = w.make_probe().snapshot()
    assert snap["n_dims"] == n_dims
    assert snap["position"].shape == (n_dims,)
    assert set(snap) >= {"position", "n_dims", "walls", "escaped"}


# (d) organizam ne dobija probe; step prima samo akciju (oblik proveren)
def test_step_rejects_wrong_action_shape():
    w = _make(2)
    w.reset(0)
    with pytest.raises(ValueError):
        w.step(np.zeros(3))
    with pytest.raises(ValueError):
        w.step(np.zeros((2, 2)))


def test_constructor_validation():
    with pytest.raises(ValueError):
        NDWorld(n_dims=1, n_touch=3, n_chemo=2, n_actuators=4)
    with pytest.raises(ValueError):
        NDWorld(n_dims=2, n_touch=3, n_chemo=2, n_actuators=3)
    with pytest.raises(ValueError):
        NDWorld(n_dims=2, n_touch=3, n_chemo=2, n_actuators=4, not_a_param=1.0)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        _make(2).step(np.zeros(4))


# (f) zid blokira kretanje
def test_wall_blocks_movement():
    w = NDWorld(n_dims=2, n_touch=3, n_chemo=2, n_actuators=4, k_drive_mm_per_ms=0.05)
    w.reset(11)
    snap = w.make_probe().snapshot
    walls = snap()["walls"]
    visible = [i for i in range(2) if np.isfinite(walls[i]).all()]
    assert len(visible) == 1
    ax = visible[0]
    lo, hi = walls[ax]
    reached_hi = reached_lo = False
    touched = False
    for direction in (1.0, -1.0):
        # q_A = a0 - a1 = direction; drugi par neutralan
        act = np.array([1.0, 0.0, 0.5, 0.5]) if direction > 0 else np.array([0.0, 1.0, 0.5, 0.5])
        for _ in range(3000):
            obs = w.step(act)
            pos = snap()["position"][ax]
            assert lo - 1e-12 <= pos <= hi + 1e-12
            reached_hi |= np.isclose(pos, hi)
            reached_lo |= np.isclose(pos, lo)
            touched |= obs.touch.max() > 0.0
    assert reached_hi or reached_lo
    assert touched


def test_chem_gradient_is_scalar_per_receptor_not_world_vector():
    w = _make(5)
    obs = w.reset(2)
    assert obs.chem_gradient.ndim == 1 and obs.chem_gradient.shape == (2,)


def test_hidden_axis_is_not_in_observation_shape():
    # promena n_dims ne menja dužine kanala (nezavisnost od n_dims)
    shapes = set()
    for d in (2, 3, 5):
        obs = _make(d).reset(9)
        shapes.add((obs.touch.shape, obs.proprio.shape, obs.chem_gradient.shape))
    assert len(shapes) == 1
