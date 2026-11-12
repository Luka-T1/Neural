"""Receptori: transdukcija opservacije u afferentne struje (nA).

Za svaki kanal: s_slow <- s_slow + dt/200 * (s - s_slow),
I = G * (0.5 * s + 0.5 * max(0, s - s_slow)), G = 2.0 nA. Uvek I >= 0.
Receptor dobija SAMO Observation i nema pristup svetu ni koordinatama.
"""

from __future__ import annotations

import numpy as np

from neural.environment.types import Observation

CONFIDENCE = "incomplete"
G_NA = 2.0
TAU_SLOW_MS = 200.0

PARAMS_META = {
    "G": {
        "value": G_NA,
        "unit": "nA",
        "source": "ASSUMPTION: pojačanje transdukcije (dizajn eksperimenta), nije izmereno",
        "confidence": "speculative",
    },
    "tau_slow": {
        "value": TAU_SLOW_MS,
        "unit": "ms",
        "source": "ASSUMPTION: spora adaptacija receptora (dizajn eksperimenta)",
        "confidence": "speculative",
    },
}


class _Receptor:
    def __init__(self, n_out: int, dt_ms: float) -> None:
        if int(n_out) < 1:
            raise ValueError("n_out mora biti >= 1")
        if dt_ms <= 0:
            raise ValueError("dt_ms mora biti > 0")
        self.n_out = int(n_out)
        self._dt = float(dt_ms)
        self._s_slow: np.ndarray | None = None

    def reset(self) -> None:
        self._s_slow = None

    def _transduce_values(self, s: np.ndarray) -> np.ndarray:
        s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
        if s.shape != (self.n_out,):
            raise ValueError(f"očekivan oblik ({self.n_out},), dobijeno {s.shape}")
        if self._s_slow is None:
            self._s_slow = s.copy()
        self._s_slow = self._s_slow + (self._dt / TAU_SLOW_MS) * (s - self._s_slow)
        current = G_NA * (0.5 * s + 0.5 * np.maximum(0.0, s - self._s_slow))
        return np.maximum(current, 0.0)


class TouchReceptor(_Receptor):
    name = "touch"

    def transduce(self, obs: Observation) -> np.ndarray:
        return self._transduce_values(obs.touch)


class ProprioReceptor(_Receptor):
    name = "proprio"

    def transduce(self, obs: Observation) -> np.ndarray:
        return self._transduce_values(obs.proprio)


class ChemoReceptor(_Receptor):
    name = "chemo"

    def transduce(self, obs: Observation) -> np.ndarray:
        return self._transduce_values(obs.chem_gradient)
