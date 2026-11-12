"""Firing-rate homeostasis: rate estimator and multiplicative synaptic scaling.

Biological purpose: keep a neuron's mean firing rate near a set point by
scaling all its excitatory synapses up/down together.

Equations:
    RateEstimator:  r' = -r/tau (exact decay per step); on spike r += 1000/tau  [Hz]
    SynapticScaling: w_i <- w_i (1 + dt (nu_target - nu)/(tau_s nu_target)), w >= w_min
Weight ratios of one neuron are preserved (multiplicative; unless clipped).

Approximation loses: the molecular sensor (calcium / BDNF / TNF-alpha), the
distinction between scaling, intrinsic plasticity and inhibitory changes.
The phenomenon is strongly supported; the mechanism is incomplete.
Biological tau_s is hours to days; here it is accelerated (ASSUMPTION).

References: Turrigiano1998, Turrigiano2008, DayanAbbott2001.
"""
from __future__ import annotations

import numpy as np

from neural.core.registry import REGISTRY

CONFIDENCE = "strongly_supported"  # phenomenon; mechanism is 'incomplete'
MECHANISM_CONFIDENCE = "incomplete"

PARAMS_META = {
    "tau_rate_ms": {"value": 1000.0, "unit": "ms", "source": "ASSUMPTION: estimator window",
                    "confidence": "speculative"},
    "nu_target_Hz": {"value": 5.0, "unit": "Hz", "source": "ASSUMPTION: set point",
                     "confidence": "speculative"},
    "tau_s_ms": {"value": 1.0e4, "unit": "ms", "source": "ASSUMPTION: vremensko ubrzanje; Turrigiano1998; Turrigiano2008",
                 "confidence": "incomplete"},
    "w_min": {"value": 0.0, "unit": "1", "source": "ASSUMPTION: lower bound", "confidence": "speculative"},
}
# tau_s_ms: biological timescale is hours-days; accelerated here (time compression).


class RateEstimator:
    def __init__(self, n=1, tau_ms=1000.0):
        self.tau, self.r = tau_ms, np.zeros(n)

    def step(self, dt, spikes):
        self.r = self.r * np.exp(-dt / self.tau) + (1000.0 / self.tau) * np.asarray(spikes, dtype=np.float64)
        return self.r


class SynapticScaling:
    def __init__(self, nu_target=5.0, tau_s=1.0e4, w_min=0.0):
        self.nu_target, self.tau_s, self.w_min = nu_target, tau_s, w_min

    def step(self, dt, w, rate):
        """w (n_post, n_pre), rate (n_post,) in Hz."""
        f = 1.0 + dt * (self.nu_target - np.asarray(rate)) / (self.tau_s * self.nu_target)
        return np.maximum(w * f[:, None], self.w_min)


@REGISTRY.register("homeostasis")
class Homeostasis:
    """StepMechanism. Input 'spikes' (n bool) = postsynaptic spikes. State: r (Hz), w (n, n_pre)."""

    name = "homeostasis"
    timescale_ms = 1000.0
    confidence = CONFIDENCE
    params_meta = PARAMS_META

    def __init__(self, n_pre=1, w0=1.0, name="homeostasis", tau_rate_ms=1000.0, nu_target=5.0,
                 tau_s=1.0e4, w_min=0.0):
        self.name, self.n_pre, self.w0 = name, n_pre, w0
        self.est_tau = tau_rate_ms
        self.scal = SynapticScaling(nu_target, tau_s, w_min)

    def init_state(self, n, rng):
        return {"r": np.zeros(n), "w": np.full((n, self.n_pre), float(self.w0))}

    def step(self, t, dt, s, inputs):
        sp = np.asarray(inputs.get("spikes", np.zeros_like(s["r"])), dtype=np.float64)
        r = s["r"] * np.exp(-dt / self.est_tau) + (1000.0 / self.est_tau) * sp
        return {"r": r, "w": self.scal.step(dt, s["w"], r)}, []
