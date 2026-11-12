"""Locked contract of the simulator core (phase 1).

Units: potential mV, time ms, injected current nA, synaptic conductance nS,
synaptic current pA, with I = g*(V - E) (positive = outward).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable

import numpy as np

CONFIDENCE_LEVELS = ("established", "strongly_supported", "incomplete", "hypothesis", "speculative")
Confidence = Literal["established", "strongly_supported", "incomplete", "hypothesis", "speculative"]


class ContractError(ValueError):
    """Raised when a component violates the interface contract."""


class _ParamMetaBase(TypedDict):
    value: object
    unit: str
    source: str


class ParamMeta(_ParamMetaBase, total=False):
    doi_status: Literal["verified", "needs-verify", "TBD-verify"]


State = dict[str, np.ndarray]


@dataclass(frozen=True)
class SpikeEvent:
    t_ms: float
    source_id: str
    index: int


@dataclass(frozen=True)
class Observation:
    touch: np.ndarray
    proprio: np.ndarray
    chem_gradient: np.ndarray

    def __post_init__(self) -> None:
        for f in ("touch", "proprio", "chem_gradient"):
            a = np.array(getattr(self, f), dtype=np.float64, copy=True)
            if a.ndim != 1:
                raise ValueError(f"Observation.{f} must be 1-D, got ndim={a.ndim}")
            a.setflags(write=False)
            object.__setattr__(self, f, a)

    def as_array(self) -> np.ndarray:
        """Order: touch, proprio, chem_gradient. Returns a new float64 array."""
        return np.concatenate([self.touch, self.proprio, self.chem_gradient])


@runtime_checkable
class Mechanism(Protocol):
    name: str
    timescale_ms: float
    confidence: str
    params_meta: dict

    def init_state(self, n: int, rng: np.random.Generator) -> State: ...


@runtime_checkable
class ODEMechanism(Mechanism, Protocol):
    """Optional: linear_coeffs(t, state, inputs) -> {key: (A, B)} with dx/dt = A + B*x
    (exp-Euler); spike_var: str | None; spike_threshold_mV: float."""

    def derivatives(self, t: float, state: State, inputs: dict) -> State: ...


@runtime_checkable
class StepMechanism(Mechanism, Protocol):
    def step(self, t: float, dt: float, state: State, inputs: dict) -> tuple[State, list[SpikeEvent]]: ...


class SimMechanism(Mechanism, Protocol):
    """Static-typing view used by Simulator: union of the optional capabilities.

    Which of them are really present is decided at runtime with callable(getattr(...))."""

    spike_var: str | None
    spike_threshold_mV: float

    def derivatives(self, t: float, state: State, inputs: dict) -> State: ...

    def linear_coeffs(self, t: float, state: State, inputs: dict) -> dict: ...

    def step(self, t: float, dt: float, state: State, inputs: dict) -> tuple[State, list[SpikeEvent]]: ...

    def on_spikes(self, t_ms: float, state: State, events: list[SpikeEvent]) -> State: ...

    def outputs(self, state: State) -> dict: ...


@runtime_checkable
class SpikeReceiver(Protocol):
    def on_spikes(self, t_ms: float, state: State, events: list[SpikeEvent]) -> State: ...


@runtime_checkable
class Sensor(Protocol):
    name: str
    n_out: int

    def transduce(self, obs: Observation) -> np.ndarray: ...


@runtime_checkable
class Motor(Protocol):
    name: str
    n_in: int

    def actuate(self, spike_counts: np.ndarray, window_ms: float) -> np.ndarray: ...


@runtime_checkable
class Environment(Protocol):
    n_actuators: int
    n_sensors: int

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: np.ndarray) -> Observation: ...


@runtime_checkable
class Probe(Protocol):
    def snapshot(self) -> dict: ...


@runtime_checkable
class SupportsProbe(Protocol):
    """Never used by the Simulator."""

    def make_probe(self) -> Probe: ...
