"""Dopamine-modulated STDP with an eligibility trace (Izhikevich 2007).

Biological purpose: reward-gated learning; STDP only tags synapses (eligibility
trace c), dopamine d converts the tag into a weight change (three-factor rule).

Equations:
    dc/dt = -c/tau_c + STDP(dt) delta(t - t_spike)
    dd/dt = -(d - d_tonic)/tau_d + DA delta(t - t_k)
    dw/dt = c d
Step order: exact decay of c, d -> STDP increment of c (same traces and order
as PairSTDP: pre: c -= A- x_post; post: c += A+ x_pre) -> trace increments ->
d += da_pulse -> w += c d dt -> clip [w_min, w_max].

Approximation loses: biochemical identity of the eligibility trace
(CaMKII / PKA etc.), DA receptor subtypes, non-reward DA roles, the mapping to
biology is a HYPOTHESIS. With d_tonic = 0 there is no change without DA.

References: Izhikevich2007, Schultz1997, Song2000.
"""
from __future__ import annotations

import numpy as np

from neural.core.registry import REGISTRY

CONFIDENCE = "hypothesis"

DA_STDP_PARAMS = {"tau_c": 1000.0, "tau_d": 200.0, "A_plus": 1.0, "A_minus": 1.5,
                  "tau_plus": 20.0, "tau_minus": 20.0, "w_min": 0.0, "w_max": 4.0, "d_tonic": 0.0}

_IZ = "Izhikevich2007"
PARAMS_META = {
    "tau_c": {"value": 1000.0, "unit": "ms", "source": _IZ, "confidence": "hypothesis"},
    "tau_d": {"value": 200.0, "unit": "ms", "source": _IZ, "confidence": "hypothesis"},
    "A_plus": {"value": 1.0, "unit": "1", "source": _IZ, "confidence": "hypothesis", "doi_status": "needs-verify"},
    "A_minus": {"value": 1.5, "unit": "1", "source": _IZ, "confidence": "hypothesis", "doi_status": "needs-verify"},
    "tau_plus": {"value": 20.0, "unit": "ms", "source": _IZ, "confidence": "hypothesis", "doi_status": "needs-verify"},
    "tau_minus": {"value": 20.0, "unit": "ms", "source": _IZ, "confidence": "hypothesis", "doi_status": "needs-verify"},
    "w_max": {"value": 4.0, "unit": "1", "source": _IZ, "confidence": "hypothesis", "doi_status": "needs-verify"},
    "w_min": {"value": 0.0, "unit": "1", "source": "ASSUMPTION: hard lower bound", "confidence": "speculative"},
    "d_tonic": {"value": 0.0, "unit": "1", "source": "ASSUMPTION: chosen so that no DA gives no change",
                "confidence": "speculative"},
}


@REGISTRY.register("da_stdp")
class DASTDP:
    """StepMechanism. Inputs 'pre_spikes', 'post_spikes' (bool), 'da' (DA pulse added to d this step)."""

    name = "da_stdp"
    timescale_ms = DA_STDP_PARAMS["tau_d"]
    confidence = CONFIDENCE
    params_meta = PARAMS_META

    def __init__(self, n_pre=1, n_post=1, w0=1.0, name="da_stdp", **params):
        self.name, self.n_pre, self.n_post, self.w0 = name, n_pre, n_post, w0
        self.p = {**DA_STDP_PARAMS, **params}

    def init_state(self, n, rng):
        return {"w": np.full((self.n_post, self.n_pre), float(self.w0)),
                "c": np.zeros((self.n_post, self.n_pre)),
                "x_pre": np.zeros(self.n_pre), "x_post": np.zeros(self.n_post),
                "d": np.full(1, self.p["d_tonic"])}

    def step(self, t, dt, s, inputs):
        p = self.p
        pre = np.asarray(inputs.get("pre_spikes", np.zeros(self.n_pre)), dtype=bool)
        post = np.asarray(inputs.get("post_spikes", np.zeros(self.n_post)), dtype=bool)
        c = s["c"] * np.exp(-dt / p["tau_c"])
        d = p["d_tonic"] + (s["d"] - p["d_tonic"]) * np.exp(-dt / p["tau_d"])
        xp = s["x_pre"] * np.exp(-dt / p["tau_plus"])
        xq = s["x_post"] * np.exp(-dt / p["tau_minus"])
        if pre.any():
            c[:, pre] -= p["A_minus"] * xq[:, None]
        if post.any():
            c[post, :] += p["A_plus"] * xp[None, :]
        xp = xp + pre
        xq = xq + post
        d = d + float(inputs.get("da", 0.0))
        w = np.clip(s["w"] + c * d * dt, p["w_min"], p["w_max"])
        return {"w": w, "c": c, "x_pre": xp, "x_post": xq, "d": d}, []
