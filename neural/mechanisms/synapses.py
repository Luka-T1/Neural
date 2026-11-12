"""Conductance-based synapses (AMPA, NMDA, GABA_A), double-exponential kinetics.

Biological purpose: fast chemical transmission; receptor-gated conductance
following transmitter release.

Equations (nS, mV, ms, pA; I = g (V - E), positive = outward):
    g(t) = g_max * w * f * (b(t) - a(t)),   a' = -a/tau_r, b' = -b/tau_d
    on spike: a += 1, b += 1 (scaled by weight w)
    f = 1 / (exp(-t_p/tau_d) - exp(-t_p/tau_r)),
    t_p = tau_r tau_d/(tau_d - tau_r) ln(tau_d/tau_r)   -> peak g = g_max * w
    NMDA: I = g B(V) (V - E),  B(V) = 1/(1 + exp(-0.062 V) [Mg]/3.57), [Mg]=1 mM
Decay is the exact exponential over a step.

Approximation loses: no transmitter diffusion / vesicle release statistics,
no receptor desensitisation or saturation, no short-term plasticity, kinetics
are typical values independent of cell type and temperature, B(V) is the
Jahr-Stevens fit only.

References: Destexhe1998, JahrStevens1990, Destexhe1994, DayanAbbott2001.
"""
from __future__ import annotations

import numpy as np

from neural.core.interfaces import SpikeEvent  # noqa: F401  (re-export for typing)
from neural.core.registry import REGISTRY

CONFIDENCE = "strongly_supported"

_TAU_SRC = "Destexhe1998; ASSUMPTION"
SYNAPSE_TYPES = {
    "ampa": {"tau_r": 0.2, "tau_d": 2.0, "E": 0.0, "mg_block": False},
    "nmda": {"tau_r": 2.0, "tau_d": 100.0, "E": 0.0, "mg_block": True},
    "gaba_a": {"tau_r": 0.5, "tau_d": 6.0, "E": -70.0, "mg_block": False},
}

PARAMS_META = {}
for _kind, _p in SYNAPSE_TYPES.items():
    PARAMS_META[f"{_kind}.tau_r"] = {"value": _p["tau_r"], "unit": "ms", "source": _TAU_SRC,
                                     "confidence": "strongly_supported"}
    PARAMS_META[f"{_kind}.tau_d"] = {"value": _p["tau_d"], "unit": "ms", "source": _TAU_SRC,
                                     "confidence": "strongly_supported"}
    PARAMS_META[f"{_kind}.E"] = {"value": _p["E"], "unit": "mV",
                                 "source": _TAU_SRC if _kind == "gaba_a" else "Destexhe1998",
                                 "confidence": "strongly_supported"}
PARAMS_META["Mg"] = {"value": 1.0, "unit": "mM", "source": "JahrStevens1990", "confidence": "established"}
PARAMS_META["g_max"] = {"value": 1.0, "unit": "nS", "source": "ASSUMPTION: unit weight",
                        "confidence": "speculative"}


def norm_factor(tau_r, tau_d):
    tp = tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
    return 1.0 / (np.exp(-tp / tau_d) - np.exp(-tp / tau_r))


def mg_block(V, mg=1.0):
    return 1.0 / (1.0 + np.exp(-0.062 * np.asarray(V, dtype=np.float64)) * mg / 3.57)


class Synapse:
    """StepMechanism + SpikeReceiver for n synapses of one receptor kind.
    on_spikes: event.index selects the synapse; weights scale the peak.
    Input 'V_post_mV' (array/scalar, default -65) gives the postsynaptic V for I_pA."""

    timescale_ms = 2.0
    confidence = CONFIDENCE
    params_meta = PARAMS_META

    def __init__(self, kind="ampa", name=None, g_max_nS=1.0, weights=None, mg_mM=1.0, **overrides):
        p = {**SYNAPSE_TYPES[kind], **overrides}
        self.kind, self.name = kind, name or kind
        self.tau_r, self.tau_d, self.E, self.has_mg = p["tau_r"], p["tau_d"], p["E"], p["mg_block"]
        self.g_max, self.mg, self.weights = g_max_nS, mg_mM, weights
        self.f = norm_factor(self.tau_r, self.tau_d)
        self.timescale_ms = self.tau_r

    def init_state(self, n, rng):
        return {"a": np.zeros(n), "b": np.zeros(n), "g": np.zeros(n), "I_pA": np.zeros(n)}

    def conductance(self, s):
        return self.g_max * self.f * (s["b"] - s["a"])

    def current(self, s, V):
        """pA, I = g B(V) (V-E)."""
        B = mg_block(V, self.mg) if self.has_mg else 1.0
        return self.conductance(s) * B * (V - self.E)

    def on_spike(self, s, idx, weight=1.0):
        s = {k: v.copy() for k, v in s.items()}
        s["a"][idx] += weight
        s["b"][idx] += weight
        s["g"] = self.conductance(s)
        return s

    def on_spikes(self, t_ms, s, events):
        for e in events:
            w = 1.0 if self.weights is None else float(np.asarray(self.weights)[e.index])
            s = self.on_spike(s, e.index, w)
        return s

    def step(self, t, dt, s, inputs):
        a = s["a"] * np.exp(-dt / self.tau_r)
        b = s["b"] * np.exp(-dt / self.tau_d)
        ns = {"a": a, "b": b}
        ns["g"] = self.conductance(ns)
        V = np.asarray(inputs.get("V_post_mV", -65.0), dtype=np.float64)
        ns["I_pA"] = self.current(ns, V) * np.ones_like(a)
        return ns, []

    def outputs(self, s):
        return {"g": s["g"], "I_pA": s["I_pA"]}


def _make(kind):
    class _S(Synapse):
        def __init__(self, name=None, **kw):
            super().__init__(kind, name or kind, **kw)
    _S.__name__ = f"{kind.upper()}Synapse"
    return REGISTRY.register(f"synapse_{kind}")(_S)


AMPASynapse = _make("ampa")
NMDASynapse = _make("nmda")
GABAASynapse = _make("gaba_a")
