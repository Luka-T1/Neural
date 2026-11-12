"""Simulation state container."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .interfaces import State


@dataclass
class SimState:
    step: int
    dt_ms: float
    mech: dict[str, State]

    @property
    def t_ms(self) -> float:
        return self.step * self.dt_ms

    def copy(self) -> "SimState":
        return SimState(self.step, self.dt_ms,
                        {m: {k: np.array(v, copy=True) for k, v in s.items()} for m, s in self.mech.items()})

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(str(int(self.step)).encode())
        for m in sorted(self.mech):
            for k in sorted(self.mech[m]):
                a = np.ascontiguousarray(self.mech[m][k])
                h.update(m.encode() + b"\0" + k.encode() + b"\0" + str(a.dtype).encode()
                         + b"\0" + str(a.shape).encode() + b"\0" + a.tobytes())
        return h.hexdigest()
