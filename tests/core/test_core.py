import warnings
from typing import Any

import numpy as np
import pytest

from neural.core.config import DEFAULT_CONFIG_PATH, Config, load_config
from neural.core.events import EventQueue
from neural.core.integrator import detect_threshold_crossings, exp_euler_step, rk4_step
from neural.core.interfaces import ContractError, Observation, SpikeEvent
from neural.core.registry import MechanismRegistry, validate_mechanism
from neural.core.seed import derive_seed, subsystem_rng
from neural.core.simulator import MultirateWarning, Simulator
from neural.core.state import SimState


class Decay:
    timescale_ms = 10.0
    confidence = "established"
    params_meta = {"tau": {"value": 10.0, "unit": "ms", "source": "ASSUMPTION: test"}}

    def __init__(self, name="d", tau=10.0, x0=1.0):
        self.name, self.tau, self.x0 = name, tau, x0

    def init_state(self, n, rng):
        return {"x": np.full(n, self.x0)}

    def derivatives(self, t, s, inp):
        return {"x": -s["x"] / self.tau + inp.get("u", 0.0)}

    def linear_coeffs(self, t, s, inp):
        return {"x": (inp.get("u", np.zeros_like(s["x"])), np.full_like(s["x"], -1 / self.tau))}

    def outputs(self, s):
        return {"x": s["x"]}


class Spiker:
    name, timescale_ms, confidence = "sp", 100.0, "hypothesis"
    params_meta: dict[str, dict[str, Any]] = {}
    spike_var, spike_threshold_mV = "v", 0.0

    def init_state(self, n, rng):
        return {"v": np.full(n, -1.0)}

    def derivatives(self, t, s, inp):
        return {"v": np.full_like(s["v"], 1.0)}  # 1 mV/ms ramp


class Recv:
    name, timescale_ms, confidence = "rx", 100.0, "hypothesis"
    params_meta: dict[str, dict[str, Any]] = {}

    def __init__(self):
        self.log = []

    def init_state(self, n, rng):
        return {"c": np.zeros(n)}

    def step(self, t, dt, s, inp):
        return s, []

    def on_spikes(self, t_ms, s, ev):
        self.log.append((t_ms, [e.index for e in ev]))
        return {"c": s["c"] + len(ev)}


def sim(cfg):
    return Simulator(cfg)


def test_rk4_exact_for_linear_and_pure(rng):
    y = {"x": np.array([1.0])}
    out = rk4_step(lambda t, y: {"x": -y["x"]}, 0.0, y, 0.1)
    assert out["x"][0] == pytest.approx(np.exp(-0.1), rel=1e-6)
    assert y["x"][0] == 1.0


@pytest.mark.parametrize("ratio", [0.1, 1, 10, 100])
def test_exp_euler_stable_monotone(ratio):
    tau = 1.0
    x = {"x": np.array([1.0])}
    c = {"x": (np.array([0.0]), np.array([-1 / tau]))}
    prev = 1.0
    for _ in range(5):
        x = exp_euler_step(c, 0, x, ratio * tau)
        assert 0 <= x["x"][0] <= prev
        prev = x["x"][0]


def test_exp_euler_B_zero_and_missing():
    out = exp_euler_step({"x": (np.array([2.0]), np.array([0.0]))}, 0, {"x": np.array([1.0])}, 0.5)
    assert out["x"][0] == 2.0
    with pytest.raises(ContractError):
        exp_euler_step({}, 0, {"x": np.array([1.0])}, 0.5)


def test_threshold_crossings():
    idx, t = detect_threshold_crossings([-1, 0, 1, -1], [1, 1, 2, 1], 0.0, 10.0, 2.0)
    assert list(idx) == [0, 3] and np.allclose(t, [11.0, 11.0])


def test_state_t_digest():
    s = SimState(40000000, 0.025, {"a": {"x": np.arange(3.0)}})
    assert s.t_ms == 40000000 * 0.025
    c = s.copy()
    assert c.digest() == s.digest()
    c.mech["a"]["x"][0] = 9
    assert c.digest() != s.digest() and s.mech["a"]["x"][0] == 0


def test_event_queue_order():
    q = EventQueue()
    for src, idx, tgt in [("b", 1, "z"), ("a", 2, "z"), ("a", 1, "z"), ("a", 1, "y")]:
        q.schedule(SpikeEvent(0.0, src, idx), 1.0, tgt)
    q.schedule(SpikeEvent(0.0, "a", 1), 1.0, "y")
    assert q.peek_time() == 1.0 and len(q) == 5
    got = [(d.target, d.event.source_id, d.event.index) for d in q.pop_due(1.0000001)]
    assert got == [("y", "a", 1), ("y", "a", 1), ("z", "a", 1), ("z", "a", 2), ("z", "b", 1)]
    with pytest.raises(ValueError):
        q.schedule(SpikeEvent(0.0, "a", 1), -1, "z")
    q.schedule(SpikeEvent(0.0, "a", 1), 1.0, "z")
    assert q.pop_due(1.0) == []  # strictly less than


def test_registry_validation():
    r = MechanismRegistry()

    @r.register("d")
    class D(Decay):
        pass

    with pytest.raises(ValueError):
        r.register("d")(D)
    assert r.names() == ["d"] and isinstance(r.create("d"), D)
    with pytest.raises(KeyError):
        r.get("zz")
    bad = Decay()
    bad.confidence = "sure"
    with pytest.raises(ContractError):
        validate_mechanism(bad)
    bad = Decay()
    bad.params_meta = {"p": {"value": 1}}
    with pytest.raises(ContractError):
        validate_mechanism(bad)


def test_config(default_config_path):
    c = load_config()
    assert c.seed == 12345 and c.dt_ms == 0.025 and DEFAULT_CONFIG_PATH == default_config_path
    assert load_config(overrides={"seed": 1, "logging": {"a": 1}}).logging == {"a": 1}
    assert Config.from_dict(c.to_dict()) == c
    for bad in ({"seed": 0, "foo": 1}, {"seed": 0, "dt_ms": 0}, {"seed": 0, "control_period_ms": 1.01}):
        with pytest.raises(ValueError):
            Config.from_dict(bad)


def test_seed_streams():
    a = subsystem_rng(1, "x").random(3)
    assert np.array_equal(a, subsystem_rng(1, "x").random(3))
    assert not np.array_equal(a, subsystem_rng(1, "y").random(3))
    assert derive_seed(1, "e") == derive_seed(1, "e") and 0 <= derive_seed(1, "e") < 2**32


def test_observation():
    o = Observation([1, 2], np.zeros(1), [3])
    assert o.touch.dtype == np.float64
    with pytest.raises(ValueError):
        Observation(np.zeros((2, 2)), [0], [0])


def test_simulator_decay_and_digest_stable(base_config):
    def run():
        s = sim(base_config)
        s.add_mechanism(Decay(), 2)
        return s.run(10.0)

    a, b = run(), run()
    assert a.digest() == b.digest()
    assert a.mech["d"]["x"][0] == pytest.approx(np.exp(-1.0), rel=1e-6)
    assert a.t_ms == pytest.approx(10.0)


def test_every_substeps_ode(base_config):
    s = sim(base_config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.add_mechanism(Decay(), 1, every=4, substeps=2)
    st = s.run(10.0)
    assert st.mech["d"]["x"][0] == pytest.approx(np.exp(-1.0), rel=1e-5)


def test_multirate_warning(base_config):
    s = sim(base_config)
    with pytest.warns(MultirateWarning):
        s.add_mechanism(Decay(tau=1.0), 1, every=100)
    with pytest.raises(ValueError):
        s.add_mechanism(Decay(tau=1.0), 1)  # duplicate


def test_exp_euler_method_in_sim(base_config):
    s = sim(base_config)
    s.add_mechanism(Decay(), 1, method="exp_euler")
    assert s.run(10.0).mech["d"]["x"][0] == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_connect_matrix(base_config):
    s = sim(base_config)
    s.add_mechanism(Decay("a", tau=1e9, x0=1.0), 2)
    s.add_mechanism(Decay("b", tau=1e9, x0=0.0), 3)
    M = np.array([[1.0, 0], [0, 2.0], [1.0, 1.0]])
    s.connect("a", "x", "b", "u", matrix=M)
    s.connect("a", "x", "b", "u", matrix=M)  # contributions add
    s.run(1.0)
    assert np.allclose(s.state.mech["b"]["x"], 2 * M @ np.ones(2) * 1.0, rtol=1e-6)
    with pytest.raises(ValueError):
        s.connect("a", "x", "b", "u")  # identity with 3 != 2
    with pytest.raises(ValueError):
        s.connect("a", "x", "b", "u", matrix=np.ones((2, 2)))


def test_spike_projection_and_order(base_config):
    s = sim(base_config)
    rx = Recv()
    s.add_mechanism(Spiker(), 1)
    s.add_mechanism(rx, 1)
    with pytest.raises(ValueError):
        s.project("sp", "rx", 0.001)
    with pytest.raises(ContractError):
        s.project("rx", "sp", 1.0)
    s.project("sp", "rx", 1.0)
    got = []
    s.run(5.0, recorder=lambda st, ev: got.extend(ev))
    assert len(got) == 1 and got[0].t_ms == pytest.approx(1.0, abs=1e-9)  # -1 mV, 1 mV/ms
    assert len(rx.log) == 1 and rx.log[0][0] == pytest.approx(2.0, abs=0.026)
    with pytest.raises(ValueError):
        s.run(0.0123)


class Env:
    n_actuators, n_sensors = 2, 1

    def __init__(self):
        self.actions, self.seed = [], None

    def reset(self, seed):
        self.seed = seed
        return Observation([1.0], [0.0], [0.0])

    def step(self, a):
        self.actions.append(a.copy())
        return Observation([float(len(self.actions))], [0.0], [0.0])


class Sens:
    name, n_out = "s", 1

    def transduce(self, obs):
        return obs.touch * 5


class Mot:
    name, n_in = "m", 1

    def actuate(self, counts, window):
        return np.array([counts.sum(), window])


def test_environment_loop(base_config):
    s = sim(base_config)
    env = Env()
    s.add_mechanism(Decay(tau=1e9, x0=0.0), 1)
    s.add_mechanism(Spiker(), 1)
    s.attach_sensor(Sens(), "d", "u")
    s.attach_motor(Mot(), "sp")
    s.attach_environment(env)
    s.run(3.0)
    assert env.seed == derive_seed(0, "environment") and len(env.actions) == 3
    assert env.actions[1][1] == 1.0
    # ZOH: u=5 in first ms, 5*1=... then touch=1,2,3 -> x ~ 5+10 after 2 ms
    s2 = sim(base_config)
    s2.add_mechanism(Decay(tau=1e9, x0=0.0), 1)
    s2.add_mechanism(Spiker(), 1)
    s2.attach_sensor(Sens(), "d", "u")
    s2.attach_motor(Mot(), "sp")
    e2 = Env()
    s2.attach_environment(e2)
    s2.run(2.0)
    assert s2.state.mech["d"]["x"][0] == pytest.approx(5.0 * 1 + 5.0 * 1, rel=1e-4)  # touch 1 then 1 (see Env)
    bad = sim(base_config)
    bad.add_mechanism(Spiker(), 1)
    bad.attach_environment(Env())
    with pytest.raises(ContractError):
        bad.run(1.0)
