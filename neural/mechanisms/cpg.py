"""Matsuoka half-center oscillator (central pattern generator).

Biological purpose: rhythm generation by two mutually inhibiting populations
with slow self-adaptation (half-center principle, Brown 1911; Marder &
Calabrese 1996). The Matsuoka model is a phenomenological rate model.

Equations (i, j = the two neurons):
    tau_r dx_i/dt = -x_i - beta v_i - a y_j + s
    tau_a dv_i/dt = -v_i + y_i,      y_i = max(0, x_i)
Initial state v = 0, x0 given. RK4 (or Euler) with dt = 0.5 ms.
Defaults a=2.5, beta=2.5, tau_r=50, tau_a=250, s=1 (ASSUMPTION); oscillation
condition 1 + tau_r/tau_a < a < 1 + beta (Matsuoka1985, needs-verify).

Approximation loses: all membrane / ionic detail, spikes, synaptic dynamics.
TODO: replace with a spiking half-center: two HH neurons with reciprocal GABA_A
inhibition and slow K+ adaptation.

References: Brown1911, Matsuoka1985, Matsuoka1987, MarderCalabrese1996.
"""
from __future__ import annotations

import numpy as np

from neural.core.registry import REGISTRY

CONFIDENCE = "strongly_supported"  # half-center principle; Matsuoka model is phenomenological

CPG_PARAMS = {"a": 2.5, "beta": 2.5, "tau_r": 50.0, "tau_a": 250.0, "s": 1.0}

PARAMS_META = {
    "a": {"value": 2.5, "unit": "1", "source": "ASSUMPTION: satisfies Matsuoka1985 condition", "confidence": "speculative"},
    "beta": {"value": 2.5, "unit": "1", "source": "ASSUMPTION: Matsuoka1985", "confidence": "speculative"},
    "tau_r": {"value": 50.0, "unit": "ms", "source": "ASSUMPTION: Matsuoka1985", "confidence": "speculative"},
    "tau_a": {"value": 250.0, "unit": "ms", "source": "ASSUMPTION: Matsuoka1985", "confidence": "speculative"},
    "s": {"value": 1.0, "unit": "1", "source": "ASSUMPTION: tonic drive", "confidence": "speculative"},
}


def oscillation_condition(p=CPG_PARAMS):
    return 1 + p["tau_r"] / p["tau_a"] < p["a"] < 1 + p["beta"]


def cpg_derivs(x, v, p=CPG_PARAMS, s=None):
    """x, v arrays (..., 2). Returns dx, dv."""
    s = p["s"] if s is None else s
    y = np.maximum(0.0, x)
    y_other = y[..., ::-1]
    dx = (-x - p["beta"] * v - p["a"] * y_other + s) / p["tau_r"]
    dv = (-v + y) / p["tau_a"]
    return dx, dv


def _rk4(x, v, dt, p, s):
    def f(xx, vv):
        return cpg_derivs(xx, vv, p, s)
    k1 = f(x, v)
    k2 = f(x + dt / 2 * k1[0], v + dt / 2 * k1[1])
    k3 = f(x + dt / 2 * k2[0], v + dt / 2 * k2[1])
    k4 = f(x + dt * k3[0], v + dt * k3[1])
    return (x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def simulate_cpg(t_stop, dt=0.5, x0=(0.1, 0.0), params=None, method="rk4"):
    p = {**CPG_PARAMS, **(params or {})}
    N = int(round(t_stop / dt))
    x, v = np.array(x0, dtype=np.float64), np.zeros(2)
    X = np.empty((N + 1, 2))
    X[0] = x
    for i in range(N):
        if method == "rk4":
            x, v = _rk4(x, v, dt, p, None)
        else:
            dx, dv = cpg_derivs(x, v, p)
            x, v = x + dt * dx, v + dt * dv
        X[i + 1] = x
    t = np.arange(N + 1) * dt
    Y = np.maximum(0.0, X)
    return {"t": t, "x": X, "y": Y, "out": Y[:, 0] - Y[:, 1]}


def oscillation_period(t, out):
    """Mean period (ms) from upward zero crossings of out = y1 - y2; nan if < 3 crossings."""
    i = np.nonzero((out[:-1] < 0) & (out[1:] >= 0))[0]
    if len(i) < 3:
        return float("nan")
    tc = t[i] + (t[i + 1] - t[i]) * (-out[i]) / (out[i + 1] - out[i])
    return float(np.mean(np.diff(tc)))


@REGISTRY.register("cpg_matsuoka")
class MatsuokaCPG:
    """ODEMechanism: n half-centers; state x, v shape (n, 2). Input 's' (tonic drive)."""

    name = "cpg_matsuoka"
    timescale_ms = CPG_PARAMS["tau_r"]
    confidence = CONFIDENCE
    params_meta = PARAMS_META

    def __init__(self, name="cpg_matsuoka", x0=(0.1, 0.0), **params):
        self.name, self.x0 = name, np.array(x0, dtype=np.float64)
        self.p = {**CPG_PARAMS, **params}

    def init_state(self, n, rng):
        return {"x": np.tile(self.x0, (n, 1)), "v": np.zeros((n, 2))}

    def derivatives(self, t, s, inputs):
        drive = inputs.get("s")
        dx, dv = cpg_derivs(s["x"], s["v"], self.p, None if drive is None else np.asarray(drive)[..., None])
        return {"x": dx, "v": dv}

    def outputs(self, s):
        y = np.maximum(0.0, s["x"])
        return {"y": y, "out": y[:, 0] - y[:, 1]}
