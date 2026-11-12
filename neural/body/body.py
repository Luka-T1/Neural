"""Telo: transdukuje opservaciju u aferentne struje i motorne spajkove u komande.

Telo samo transdukuje: nema logike tipa „ako zid onda…“. Dobija Observation od
sveta i vraća goli np.ndarray (9,) aferentnih struja u nA. Probe nikad ne vidi.
Telo ne uvozi svet ni koordinate; svet je prosleđen kao objekat sa
reset(...) i step(commands, dt_ms).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from neural.body.motor import MotorNeurons, MotorUnit
from neural.body.receptors import ChemoReceptor, ProprioReceptor, TouchReceptor
from neural.environment.types import Observation

CONFIDENCE = "incomplete"


@dataclass(frozen=True)
class BodyConfig:
    n_touch: int = 3
    n_actuators: int = 4
    n_chemo: int = 2
    n_pools: int = 4
    neurons_per_pool: int = 10
    dt_ms: float = 1.0

    def __post_init__(self) -> None:
        if self.n_pools != self.n_actuators:
            raise ValueError("n_pools mora biti jednako n_actuators (jedna pula po aktuatoru)")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms mora biti > 0")


class Body:
    """Body(world, config, seed). Ako rng nije prosleđen, pravi se iz seed-a.

    Napomena: specifikacija traži da RNG dolazi spolja (subsystem_rng(seed, 'body')).
    Dok neural.core.seed ne postoji, fallback je np.random.default_rng(seed).

    reset() vraća motorni RNG na početno stanje (sačuvano u konstruktoru), pa su epizode sa
    istim seedom ponovljive. Kad se rng prosledi spolja, menja se i stanje tog spoljnog generatora.
    """

    def __init__(self, world, config: BodyConfig | None = None, seed: int = 0, rng: np.random.Generator | None = None) -> None:
        self._world = world
        self._cfg = config or BodyConfig()
        self._rng = rng if rng is not None else np.random.default_rng(int(seed))
        self._rng_state0 = copy.deepcopy(self._rng.bit_generator.state)
        self._motor_neurons = MotorNeurons(self._rng, self._cfg.n_pools, self._cfg.neurons_per_pool)
        self._motor_unit = MotorUnit(self._cfg.n_pools)
        dt = self._cfg.dt_ms
        self._touch = TouchReceptor(self._cfg.n_touch, dt)
        self._proprio = ProprioReceptor(self._cfg.n_actuators, dt)
        self._chemo = ChemoReceptor(self._cfg.n_chemo, dt)
        self.last_commands = np.zeros(self._cfg.n_actuators)

    def reset(self, *args, **kwargs) -> np.ndarray:
        obs = self._world.reset(*args, **kwargs)
        self._check(obs)
        self._rng.bit_generator.state = copy.deepcopy(self._rng_state0)
        self._motor_neurons.reset()
        self._motor_unit.reset()
        self._touch.reset()
        self._proprio.reset()
        self._chemo.reset()
        self.last_commands = np.zeros(self._cfg.n_actuators)
        return self._transduce(obs)

    def step(self, premotor_current_nA: np.ndarray, dt_ms: float) -> np.ndarray:
        if abs(float(dt_ms) - self._cfg.dt_ms) > 1e-12:
            raise ValueError(f"dt_ms mora biti {self._cfg.dt_ms} (konfiguracija tela), dobijeno {dt_ms}")
        premotor = np.asarray(premotor_current_nA, dtype=np.float64)
        if premotor.shape != (self._cfg.n_pools,):
            raise ValueError(f"premotorna struja mora imati oblik ({self._cfg.n_pools},), dobijeno {premotor.shape}")
        counts = self._motor_neurons.step(premotor, dt_ms)
        commands = self._motor_unit.actuate(counts, dt_ms)
        self.last_commands = commands.copy()
        obs = self._world.step(commands, dt_ms)
        self._check(obs)
        return self._transduce(obs)

    # --- interno ----------------------------------------------------------

    def _check(self, obs: Observation) -> None:
        if not isinstance(obs, Observation):
            raise TypeError("svet mora vratiti Observation")
        expect = (
            (obs.touch.shape, (self._cfg.n_touch,)),
            (obs.proprio.shape, (self._cfg.n_actuators,)),
            (obs.chem_gradient.shape, (self._cfg.n_chemo,)),
        )
        for got, want in expect:
            if got != want:
                raise ValueError(f"oblik opservacije {got} ne odgovara konfiguraciji tela {want}")

    def _transduce(self, obs: Observation) -> np.ndarray:
        out = np.concatenate(
            [self._touch.transduce(obs), self._proprio.transduce(obs), self._chemo.transduce(obs)]
        )
        return np.asarray(out, dtype=np.float64)
