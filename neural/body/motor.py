"""Motorni neuroni i aktuatori.

MotorNeurons: n_pools x neurons_per_pool LIF neurona.
  C = 1 nF, g_L = 50 nS, E_L = -65 mV, prag -50 mV, reset -65 mV,
  refrakcija 2 ms, Gausov šum struje 0.1 nA pri referentnom dt=0.1 ms,
  skalira se sa sqrt(0.1/dt) (noise_scale), tako da stacionarna varijansa
  membrane ne zavisi od dt. Jedinice: mV/ms/nA/nS.
  Rheobaza ~0.75 nA (<= 1 nA), 5 nA pouzdano okida.

MotorUnit (aktuator): r je niskopropusni filtar brzine (tau = 20 ms),
  brzina = counts / window_ms * 1000 (Hz), u = clip(r / r_max, 0, 1), r_max = 100 Hz.

PRIVREMENO: LIF je lokalna implementacija jer u ovom worktree-u nema
mehanizama za uvoz (neural.cellular._adapter). Zameniti mehanizmom kad postane dostupan.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

CONFIDENCE = "incomplete"

C_NF = 1.0
G_L_NS = 50.0
E_L_MV = -65.0
V_TH_MV = -50.0
V_RESET_MV = -65.0
T_REF_MS = 2.0
NOISE_NA = 0.1
REF_DT_MS = 0.1
R_MAX_HZ = 100.0
TAU_OUT_MS = 20.0

PARAMS_META = {
    "C": {"value": C_NF, "unit": "nF", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "g_L": {"value": G_L_NS, "unit": "nS", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "E_L": {"value": E_L_MV, "unit": "mV", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "V_th": {"value": V_TH_MV, "unit": "mV", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "V_reset": {"value": V_RESET_MV, "unit": "mV", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "t_ref": {"value": T_REF_MS, "unit": "ms", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno)", "confidence": "speculative"},
    "noise": {"value": NOISE_NA, "unit": "nA", "source": "ASSUMPTION: LIF parametri za motorne neurone (privremeno); šum 0.1 nA je pretpostavljen pri referentnom dt=REF_DT_MS=0.1 ms i skalira se sa sqrt(REF_DT_MS/dt) (ASSUMPTION o referentnom dt)", "confidence": "speculative"},
    "r_max": {"value": R_MAX_HZ, "unit": "Hz", "source": "ASSUMPTION: saturacija aktuatora (dizajn eksperimenta)", "confidence": "speculative"},
    "tau_out": {"value": TAU_OUT_MS, "unit": "ms", "source": "ASSUMPTION: filtar brzine aktuatora (dizajn eksperimenta)", "confidence": "speculative"},
}


def noise_scale(dt_ms: float) -> float:
    """Skala motornog šuma: sqrt(REF_DT_MS / dt_ms). Lokalna kopija (body ne zavisi od organizma)."""
    dt = float(dt_ms)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_ms mora biti konačan i > 0, dobijeno {dt_ms}")
    return math.sqrt(REF_DT_MS / dt)


class MotorNeurons:
    """Pule LIF neurona. Šum dolazi iz prosleđenog rng-a (nema sopstvenog RNG-a)."""

    def __init__(self, rng: np.random.Generator, n_pools: int = 4, neurons_per_pool: int = 10) -> None:
        if n_pools < 1 or neurons_per_pool < 1:
            raise ValueError("n_pools i neurons_per_pool moraju biti >= 1")
        self._rng = rng
        self._n_pools = int(n_pools)
        self._n = int(neurons_per_pool)
        self._scale_dt: Optional[float] = None
        self._scale = 1.0
        self.reset()

    @property
    def v(self) -> np.ndarray:
        """Kopija napona membrane, shape (n_pools, neurons_per_pool)."""
        return self._v.copy()

    def reset(self) -> None:
        self._v = np.full((self._n_pools, self._n), V_RESET_MV)
        self._refr = np.zeros((self._n_pools, self._n))

    def step(self, premotor_current_nA: np.ndarray, dt_ms: float) -> np.ndarray:
        """Vraća prosečan broj spajkova po neuronu u pulu za ovaj korak, shape (n_pools,)."""
        I = np.asarray(premotor_current_nA, dtype=np.float64)
        if I.shape != (self._n_pools,):
            raise ValueError(f"premotorna struja mora imati oblik ({self._n_pools},), dobijeno {I.shape}")
        if dt_ms != self._scale_dt:
            self._scale = noise_scale(dt_ms)
            self._scale_dt = dt_ms
        noisy = I[:, None] + NOISE_NA * self._rng.standard_normal((self._n_pools, self._n)) * self._scale
        active = self._refr <= 0.0
        dv = dt_ms * (-G_L_NS * (self._v - E_L_MV) * 1e-3 + noisy) / C_NF
        v = np.where(active, self._v + dv, self._v)
        spiked = active & (v >= V_TH_MV)
        v = np.where(spiked, V_RESET_MV, v)
        self._refr = np.where(spiked, T_REF_MS, np.maximum(self._refr - dt_ms, 0.0))
        self._v = v
        return spiked.sum(axis=1) / self._n


class MotorUnit:
    """Aktuator: filtrira brzinu pule i saturiše je u komandu u in [0, 1]."""

    def __init__(self, n_pools: int = 4, r_max_hz: float = R_MAX_HZ, tau_ms: float = TAU_OUT_MS) -> None:
        if r_max_hz <= 0 or tau_ms <= 0:
            raise ValueError("r_max_hz i tau_ms moraju biti > 0")
        self.name = "motor_unit"
        self._n_pools = int(n_pools)
        self.n_in = self._n_pools
        self._r_max = float(r_max_hz)
        self._tau = float(tau_ms)
        self.reset()

    def reset(self) -> None:
        self._r = np.zeros(self._n_pools)

    def actuate(self, spike_counts: np.ndarray, window_ms: float) -> np.ndarray:
        counts = np.asarray(spike_counts, dtype=np.float64)
        if counts.shape != (self._n_pools,):
            raise ValueError(f"spike_counts mora imati oblik ({self._n_pools},), dobijeno {counts.shape}")
        if window_ms <= 0:
            raise ValueError("window_ms mora biti > 0")
        rate_hz = counts / window_ms * 1000.0
        ratio = min(1.0, window_ms / self._tau)
        self._r = self._r + ratio * (rate_hz - self._r)
        return np.clip(self._r / self._r_max, 0.0, 1.0)
