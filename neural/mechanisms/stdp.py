"""Pair-based spike-timing-dependent plasticity (STDP), online all-to-all.

Biological purpose: Hebbian, timing-dependent change of synaptic efficacy.

Window (dt = t_post - t_pre):
    dw = A+ exp(-dt/tau+)   for dt > 0 (pre before post: potentiation)
    dw = -A- exp(dt/tau-)   for dt < 0 (post before pre: depression)
    dw = 0                  for dt = 0
Online implementation with traces x_pre (tau+), x_post (tau-). Per step:
    (1) x *= exp(-dt/tau)  (2) pre spike:  w -= A- x_post
    (3) post spike: w += A+ x_pre  (4) trace increment  (5) clip [w_min, w_max]
A coincident pre/post pair therefore yields dw = 0.

Approximation loses: triplet / rate / voltage dependence, calcium-based
mechanisms, weight-dependent updates, dendritic location effects. tau values are
fits to Bi & Poo hippocampal culture data; A+ / A- are model choices.

References: BiPoo1998, BiPoo2001, Song2000, DayanAbbott2001.
"""
from __future__ import annotations

import numpy as np

from neural.core.registry import REGISTRY

CONFIDENCE = "strongly_supported"

STDP_PARAMS = {"tau_plus": 16.8, "tau_minus": 33.7, "A_plus": 0.005, "A_minus": 1.05 * 0.005,
               "w_min": 0.0, "w_max": 1.0}

PARAMS_META = {
    "tau_plus": {"value": 16.8, "unit": "ms", "source": "BiPoo2001; BiPoo1998",
                 "confidence": "strongly_supported", "doi_status": "needs-verify"},
    "tau_minus": {"value": 33.7, "unit": "ms", "source": "BiPoo2001; BiPoo1998",
                  "confidence": "strongly_supported", "doi_status": "needs-verify"},
    "A_plus": {"value": 0.005, "unit": "1", "source": "Song2000; ASSUMPTION", "confidence": "hypothesis"},
    "A_minus": {"value": 0.00525, "unit": "1", "source": "Song2000; ASSUMPTION", "confidence": "hypothesis"},
    "w_min": {"value": 0.0, "unit": "1", "source": "ASSUMPTION: hard lower bound", "confidence": "speculative"},
    "w_max": {"value": 1.0, "unit": "1", "source": "ASSUMPTION: hard upper bound", "confidence": "speculative"},
}


def stdp_window(delta_t, A_plus=STDP_PARAMS["A_plus"], A_minus=STDP_PARAMS["A_minus"],
                tau_plus=STDP_PARAMS["tau_plus"], tau_minus=STDP_PARAMS["tau_minus"]):
    """dw as function of delta_t = t_post - t_pre (ms)."""
    d = np.asarray(delta_t, dtype=np.float64)
    return np.where(d > 0, A_plus * np.exp(-np.abs(d) / tau_plus),
                    np.where(d < 0, -A_minus * np.exp(-np.abs(d) / tau_minus), 0.0))


def stdp_update(w, x_pre, x_post, pre, post, dt, p):
    """One online step. w (n_post, n_pre); pre/post boolean spike arrays. Returns (w, x_pre, x_post)."""
    x_pre = x_pre * np.exp(-dt / p["tau_plus"])
    x_post = x_post * np.exp(-dt / p["tau_minus"])
    w = w.copy()
    if pre.any():
        w[:, pre] -= p["A_minus"] * x_post[:, None]
    if post.any():
        w[post, :] += p["A_plus"] * x_pre[None, :]
    x_pre = x_pre + pre
    x_post = x_post + post
    return np.clip(w, p["w_min"], p["w_max"]), x_pre, x_post


@REGISTRY.register("pair_stdp")
class PairSTDP:
    """StepMechanism. Inputs 'pre_spikes' (n_pre bool), 'post_spikes' (n_post bool)."""

    name = "pair_stdp"
    timescale_ms = STDP_PARAMS["tau_plus"]
    confidence = CONFIDENCE
    params_meta = PARAMS_META

    def __init__(self, n_pre=1, n_post=1, w0=0.5, name="pair_stdp", **params):
        self.name, self.n_pre, self.n_post, self.w0 = name, n_pre, n_post, w0
        self.p = {**STDP_PARAMS, **params}

    def init_state(self, n, rng):
        return {"w": np.full((self.n_post, self.n_pre), float(self.w0)),
                "x_pre": np.zeros(self.n_pre), "x_post": np.zeros(self.n_post)}

    def step(self, t, dt, s, inputs):
        pre = np.asarray(inputs.get("pre_spikes", np.zeros(self.n_pre)), dtype=bool)
        post = np.asarray(inputs.get("post_spikes", np.zeros(self.n_post)), dtype=bool)
        w, xp, xq = stdp_update(s["w"], s["x_pre"], s["x_post"], pre, post, dt, self.p)
        return {"w": w, "x_pre": xp, "x_post": xq}, []
