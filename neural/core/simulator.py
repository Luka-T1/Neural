"""Fixed-step multirate simulator (phase 1).

Base step k (t_k = k*dt):
 0. due spike deliveries (arrival < t_{k+1}) -> on_spikes(t_k, ...)
 1. at control-period start: sensors transduce the observation (ZOH)
 2. bus snapshot; every mechanism with (k+1) % every == 0 advances
    t_{k+1-every} -> t_{k+1} in `substeps` substeps
 3. new spikes are scheduled on every projection
 4. at control-period end: motors -> concatenated action
 5. env.step(action) -> new observation
Units: mV, ms, nA (injected), nS, pA.
"""
from __future__ import annotations

import warnings
from typing import Any
from dataclasses import dataclass

import numpy as np

from .config import Config
from .events import EventQueue
from .integrator import detect_threshold_crossings, exp_euler_step, rk4_step
from .interfaces import ContractError, SimMechanism, SpikeEvent
from .registry import MechanismRegistry, validate_mechanism
from .seed import derive_seed, subsystem_rng
from .state import SimState


class MultirateWarning(UserWarning):
    pass


@dataclass
class _Entry:
    mech: SimMechanism
    n: int
    every: int
    substeps: int
    method: str


class Simulator:
    def __init__(self, config: Config, registry: MechanismRegistry | None = None):
        self.config = config
        self.registry = registry
        self.dt = config.dt_ms
        self._spc = config.steps_per_control
        self._entries: dict[str, _Entry] = {}
        self._conns: list[tuple] = []
        self._projs: list[tuple] = []
        self._sensors: list[tuple] = []
        self._motors: list[tuple] = []
        self._env = None
        self._state: SimState | None = None
        self._queue = EventQueue()
        self._obs = None
        self._bus_sensor: list[tuple] = []
        self._counts: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ construction
    def add_mechanism(self, mech: Any, n: int, every: int = 1, substeps: int = 1, method: str | None = None):
        validate_mechanism(mech)
        if mech.name in self._entries:
            raise ValueError(f"duplicate mechanism name '{mech.name}'")
        if every < 1 or substeps < 1:
            raise ValueError("every and substeps must be >= 1")
        method = method or self.config.default_method
        if method not in ("rk4", "exp_euler"):
            raise ValueError(f"unknown method '{method}'")
        if callable(getattr(mech, "derivatives", None)) and method == "exp_euler" \
                and not callable(getattr(mech, "linear_coeffs", None)):
            raise ContractError(f"{mech.name}: exp_euler needs linear_coeffs")
        if every * self.dt > mech.timescale_ms / 10:
            warnings.warn(f"{mech.name}: every*dt = {every * self.dt} ms exceeds timescale/10 "
                          f"= {mech.timescale_ms / 10} ms", MultirateWarning, stacklevel=2)
        self._entries[mech.name] = _Entry(mech, int(n), int(every), int(substeps), method)
        return mech

    def _need(self, name: str) -> _Entry:
        if name not in self._entries:
            raise ValueError(f"unknown mechanism '{name}'")
        return self._entries[name]

    @staticmethod
    def _matrix(matrix, n_dst: int, n_src: int):
        if matrix is None:
            if n_dst != n_src:
                raise ValueError(f"identity needs n_dst == n_src ({n_dst} != {n_src})")
            return None
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (n_dst, n_src):
            raise ValueError(f"matrix shape {m.shape} != {(n_dst, n_src)}")
        return m

    def connect(self, src: str, src_key: str, dst: str, dst_key: str, matrix=None):
        s, d = self._need(src), self._need(dst)
        self._conns.append((src, src_key, dst, dst_key, self._matrix(matrix, d.n, s.n)))

    def project(self, src: str, dst: str, delay_ms: float):
        self._need(src)
        d = self._need(dst)
        if delay_ms < self.dt * (1 - 1e-12):
            raise ValueError(f"delay_ms {delay_ms} < dt {self.dt}")
        if not callable(getattr(d.mech, "on_spikes", None)):
            raise ContractError(f"{dst} has no on_spikes")
        self._projs.append((src, dst, float(delay_ms)))

    def attach_sensor(self, sensor, dst: str, dst_key: str = "I_ext_nA", matrix=None):
        d = self._need(dst)
        self._sensors.append((sensor, dst, dst_key, self._matrix(matrix, d.n, sensor.n_out)))

    def attach_motor(self, motor, src: str):
        self._need(src)
        self._motors.append((motor, src))

    def attach_environment(self, env):
        self._env = env

    # ------------------------------------------------------------------- state
    @property
    def state(self) -> SimState:
        if self._state is None:
            self.reset()
        assert self._state is not None
        return self._state

    def reset(self) -> SimState:
        seed = self.config.seed
        mech = {}
        for name, e in self._entries.items():
            mech[name] = e.mech.init_state(e.n, subsystem_rng(seed, name))
        self._state = SimState(0, self.dt, mech)
        self._queue = EventQueue()
        self._counts = {name: np.zeros(e.n, dtype=np.int64) for name, e in self._entries.items()}
        self._bus_sensor = []
        self._obs = self._env.reset(derive_seed(seed, "environment")) if self._env is not None else None
        assert self._state is not None
        return self._state

    # -------------------------------------------------------------------- step
    def _inputs(self, snap: dict, extra: list) -> dict[str, dict[str, np.ndarray]]:
        inputs: dict[str, dict] = {name: {} for name in self._entries}

        def add(dst, key, vec):
            cur = inputs[dst].get(key)
            inputs[dst][key] = vec if cur is None else cur + vec

        for src, src_key, dst, dst_key, m in self._conns:
            mech = self._entries[src].mech
            vals = mech.outputs(snap[src]) if callable(getattr(mech, "outputs", None)) else snap[src]
            v = np.asarray(vals[src_key] if isinstance(vals, dict) else vals, dtype=np.float64)
            add(dst, dst_key, v if m is None else m @ v)
        for dst, key, vec in extra:
            add(dst, key, vec)
        return inputs

    def _advance(self, name: str, e: _Entry, t0: float, state, inputs: dict):
        mech, events = e.mech, []
        dt_sub = e.every * self.dt / e.substeps
        for s in range(e.substeps):
            t = t0 + s * dt_sub
            if callable(getattr(mech, "step", None)) and not callable(getattr(mech, "derivatives", None)):
                state, ev = mech.step(t, dt_sub, state, inputs)
                events.extend(ev)
                continue
            if e.method == "rk4":
                new = rk4_step(lambda tt, y: mech.derivatives(tt, y, inputs), t, state, dt_sub)
            else:
                new = exp_euler_step(mech.linear_coeffs(t, state, inputs), t, state, dt_sub)
            sv = getattr(mech, "spike_var", None)
            if sv:
                idx, tc = detect_threshold_crossings(state[sv], new[sv], mech.spike_threshold_mV, t, dt_sub)
                events.extend(SpikeEvent(float(tt), name, int(i)) for i, tt in zip(idx, tc))
            state = new
        return state, events

    def step(self) -> list[SpikeEvent]:
        st = self.state
        k, dt = st.step, self.dt
        t_k, t_next = k * dt, (k + 1) * dt

        # 0) deliveries
        by_target: dict[str, list] = {}
        for dl in self._queue.pop_due(t_next):
            by_target.setdefault(dl.target, []).append(dl.event)
        for target in sorted(by_target):
            mech = self._entries[target].mech
            st.mech[target] = mech.on_spikes(t_k, st.mech[target], by_target[target])

        # 1) sensors (ZOH)
        if self._env is not None and k % self._spc == 0:
            self._bus_sensor = []
            for sensor, dst, key, m in self._sensors:
                v = np.asarray(sensor.transduce(self._obs), dtype=np.float64)
                if v.shape != (sensor.n_out,):
                    raise ContractError(f"sensor {sensor.name}: expected shape ({sensor.n_out},), got {v.shape}")
                self._bus_sensor.append((dst, key, v if m is None else m @ v))

        # 2) advance
        snap = dict(st.mech)
        inputs = self._inputs(snap, self._bus_sensor)
        new_events: list[SpikeEvent] = []
        for name, e in self._entries.items():
            if (k + 1) % e.every != 0:
                continue
            t0 = (k + 1 - e.every) * dt
            st.mech[name], ev = self._advance(name, e, t0, snap[name], inputs[name])
            new_events.extend(ev)

        # 3) schedule spikes
        for ev in new_events:
            c = self._counts[ev.source_id]
            if 0 <= ev.index < c.size:
                c[ev.index] += 1
            for src, dst, delay in self._projs:
                if src == ev.source_id:
                    self._queue.schedule(ev, delay, dst)

        st.step = k + 1

        # 4-5) control
        if self._env is not None and (k + 1) % self._spc == 0:
            parts = [np.asarray(m.actuate(self._counts[src], self.config.control_period_ms),
                                dtype=np.float64).ravel() for m, src in self._motors]
            action = np.concatenate(parts) if parts else np.zeros(0)
            if action.size != self._env.n_actuators:
                raise ContractError(f"action length {action.size} != n_actuators {self._env.n_actuators}")
            for c in self._counts.values():
                c[:] = 0
            self._obs = self._env.step(action)
        return new_events

    def run(self, duration_ms: float, recorder=None) -> SimState:
        r = duration_ms / self.dt
        if abs(r - round(r)) > 1e-9 * max(1.0, abs(r)):
            raise ValueError("duration_ms must be an integer multiple of dt_ms")
        _ = self.state
        for _i in range(int(round(r))):
            ev = self.step()
            if recorder is not None:
                recorder(self._state, ev)
        assert self._state is not None
        return self._state
