"""AMPA / NMDA / GABA_A double-exponential synapses with Mg2+ block.

g(t) = g_max * w * f * (exp(-t/tau_d) - exp(-t/tau_r)), peak = g_max * w.
B(V) = 1 / (1 + [Mg]/3.57 * exp(-0.062 V)) (Jahr & Stevens 1990).
Units: ms, mV, mM, nS (conductance), pA (current, outward positive).
Default tau and E values are the specification defaults (phenomenological,
cell-type dependent), not universal experimental constants.
"""
import math

import numpy as np

from neural.cellular._adapter import Mechanism
from neural.cellular import references as R


def mg_block(V_mV, mg_mM=1.0):
    V = np.asarray(V_mV, dtype=float)
    b = 1.0 / (1.0 + mg_mM / 3.57 * np.exp(-0.062 * V))
    return float(b) if b.ndim == 0 else b


def peak_time(tau_r, tau_d):
    return tau_r * tau_d / (tau_d - tau_r) * math.log(tau_d / tau_r)


class DoubleExpSynapse(Mechanism):
    name = 'double_exp_synapse'
    status = 'implemented'
    reliability_class = 'strongly_supported'
    references = [R.JAHR_STEVENS1990, R.DAYAN_ABBOTT2001]

    def __init__(self, tau_rise_ms, tau_decay_ms, E_rev_mV, g_max_nS=1.0,
                 mg_block_enabled=False, mg_mM=1.0):
        if not 0.0 < tau_rise_ms < tau_decay_ms:
            raise ValueError('need 0 < tau_rise < tau_decay')
        self.tau_rise_ms = float(tau_rise_ms)
        self.tau_decay_ms = float(tau_decay_ms)
        self.E_rev_mV = float(E_rev_mV)
        self.g_max_nS = float(g_max_nS)
        self.mg_block_enabled = bool(mg_block_enabled)
        self.mg_mM = float(mg_mM)
        tp = peak_time(self.tau_rise_ms, self.tau_decay_ms)
        self._norm = 1.0 / (math.exp(-tp / self.tau_decay_ms) - math.exp(-tp / self.tau_rise_ms))
        self.reset()

    def reset(self):
        self._a = 0.0
        self._b = 0.0

    def state(self):
        return {'a': self._a, 'b': self._b}

    @property
    def g(self):
        return self.g_max_nS * (self._b - self._a)

    def on_spike(self, weight=1.0):
        self._a += self._norm * weight
        self._b += self._norm * weight

    def step(self, dt):
        self._a *= math.exp(-dt / self.tau_rise_ms)
        self._b *= math.exp(-dt / self.tau_decay_ms)
        return self.g

    def current(self, V_mV):
        B = mg_block(V_mV, self.mg_mM) if self.mg_block_enabled else 1.0
        return self.g * B * (V_mV - self.E_rev_mV)

    def ca_current(self, V_mV, fraction=0.10):
        """Ca2+ share of the current. fraction ~0.1 is an unverified placeholder (TBD-verify)."""
        return fraction * self.current(V_mV)


def ampa(g_max_nS=1.0):
    return DoubleExpSynapse(0.2, 2.0, 0.0, g_max_nS)


def nmda(g_max_nS=1.0, mg_mM=1.0):
    return DoubleExpSynapse(2.0, 100.0, 0.0, g_max_nS, mg_block_enabled=True, mg_mM=mg_mM)


def gaba_a(g_max_nS=1.0):
    return DoubleExpSynapse(0.5, 5.0, -70.0, g_max_nS)
