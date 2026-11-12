"""Hodgkin-Huxley point neuron (squid giant axon, 6.3 degC).

Biological purpose: generation of the action potential from voltage-gated
Na+ / K+ conductances (membrane excitability).

Equations (modern convention V = V_in - V_out, rest -65 mV; units mV, ms,
uF/cm^2, mS/cm^2, uA/cm^2):
    C dV/dt = I_ext - g_Na m^3 h (V-E_Na) - g_K n^4 (V-E_K) - g_L (V-E_L)
    dx/dt = a_x(V)(1-x) - b_x(V) x,   x in {m, h, n}
    a_m = 0.1(V+40)/(1-exp(-(V+40)/10)),  b_m = 4 exp(-(V+65)/18)
    a_h = 0.07 exp(-(V+65)/20),           b_h = 1/(1+exp(-(V+35)/10))
    a_n = 0.01(V+55)/(1-exp(-(V+55)/10)), b_n = 0.125 exp(-(V+65)/80)
Removable singularities at V=-40 (a_m -> 1.0) and V=-55 (a_n -> 0.1) are
handled via the limit / Taylor expansion for |x| < 1e-7.

Approximation loses: single compartment (no morphology), lumped channel
kinetics (no stochastic gating, no Na+ inactivation modes), fixed 6.3 degC
(Q10 scaling is optional and an ASSUMPTION), no ion concentration dynamics.
Type II excitability: f-I jump at threshold / bistability near rheobase.

Integrators: 'rk4' (default) or 'exp_euler' (Rush-Larsen for gates, forward
Euler for V; gates are updated first with V_n, then V with updated gates).

References: HH1952, RushLarsen1978, DayanAbbott2001.
"""
from __future__ import annotations

import math

import numpy as np

from neural.core.registry import REGISTRY

CONFIDENCE = "established"

HH_PARAMS = {"C_m": 1.0, "g_Na": 120.0, "g_K": 36.0, "g_L": 0.3,
             "E_Na": 50.0, "E_K": -77.0, "E_L": -54.387}

_HH_SRC = "HH1952"
PARAMS_META = {
    "C_m": {"value": 1.0, "unit": "uF/cm^2", "source": _HH_SRC, "confidence": "established"},
    "g_Na": {"value": 120.0, "unit": "mS/cm^2", "source": _HH_SRC, "confidence": "established"},
    "g_K": {"value": 36.0, "unit": "mS/cm^2", "source": _HH_SRC, "confidence": "established"},
    "g_L": {"value": 0.3, "unit": "mS/cm^2", "source": _HH_SRC, "confidence": "established"},
    # E_Na, E_K, E_L: HH1952 values shifted to the modern convention (V_rest ~ -65 mV);
    # source stays a bibliography key.
    "E_Na": {"value": 50.0, "unit": "mV", "source": "HH1952", "confidence": "established"},
    "E_K": {"value": -77.0, "unit": "mV", "source": "HH1952", "confidence": "established"},
    "E_L": {"value": -54.387, "unit": "mV", "source": "HH1952", "confidence": "established"},
    "temperature": {"value": 6.3, "unit": "degC", "source": _HH_SRC, "confidence": "established"},
    "Q10": {"value": 1.0, "unit": "1", "source": "ASSUMPTION: no temperature scaling by default",
            "confidence": "speculative"},
    "area_cm2": {"value": 1e-3, "unit": "cm^2", "source": "ASSUMPTION: 1 nA -> 1 uA/cm^2 conversion area",
                 "confidence": "speculative"},
}


def nA_to_density(I_nA, area_cm2):
    """Injected current [nA] -> density [uA/cm^2] = I_nA * 1e-3 / area_cm2."""
    return np.asarray(I_nA) * 1e-3 / area_cm2


def _vtrap(x, k):
    """x / (1 - exp(-x/k)) with the removable singularity at x=0 (limit = k)."""
    x = np.asarray(x, dtype=np.float64)
    small = np.abs(x) < 1e-7
    xs = np.where(small, 1.0, x)
    full = xs / (-np.expm1(-xs / k))
    taylor = k + x / 2.0 + x * x / (12.0 * k)
    return np.where(small, taylor, full)


def rates(V):
    V = np.asarray(V, dtype=np.float64)
    am = 0.1 * _vtrap(V + 40.0, 10.0)
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * _vtrap(V + 55.0, 10.0)
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def steady_state(V):
    am, bm, ah, bh, an, bn = rates(V)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def currents(V, m, h, n, p=HH_PARAMS):
    return (p["g_Na"] * m ** 3 * h * (V - p["E_Na"]),
            p["g_K"] * n ** 4 * (V - p["E_K"]),
            p["g_L"] * (V - p["E_L"]))


def derivs(V, m, h, n, I_ext, p=HH_PARAMS, q10=1.0):
    am, bm, ah, bh, an, bn = rates(V)
    iNa, iK, iL = currents(V, m, h, n, p)
    dV = (I_ext - iNa - iK - iL) / p["C_m"]
    return dV, q10 * (am * (1 - m) - bm * m), q10 * (ah * (1 - h) - bh * h), q10 * (an * (1 - n) - bn * n)


def _vtrap_s(x, k):
    if abs(x) < 1e-7:
        return k + x / 2.0 + x * x / (12.0 * k)
    return x / (-math.expm1(-x / k))


def _rates_s(V):
    """Scalar (pure-python) twin of rates(), used in the time loop for speed."""
    return (0.1 * _vtrap_s(V + 40.0, 10.0), 4.0 * math.exp(-(V + 65.0) / 18.0),
            0.07 * math.exp(-(V + 65.0) / 20.0), 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0)),
            0.01 * _vtrap_s(V + 55.0, 10.0), 0.125 * math.exp(-(V + 65.0) / 80.0))


def _derivs_s(V, m, h, n, I, p, q10):
    am, bm, ah, bh, an, bn = _rates_s(V)
    dV = (I - p["g_Na"] * m ** 3 * h * (V - p["E_Na"]) - p["g_K"] * n ** 4 * (V - p["E_K"])
          - p["g_L"] * (V - p["E_L"])) / p["C_m"]
    return (dV, q10 * (am * (1 - m) - bm * m), q10 * (ah * (1 - h) - bh * h), q10 * (an * (1 - n) - bn * n))


def simulate_hh(I_ext, t_stop, dt=0.025, V0=-65.0, method="rk4", params=None, q10=1.0):
    """Integrate one HH neuron. I_ext: float or callable(t) in uA/cm^2.
    Returns dict(t, V, m, h, n) sampled at every step (len N+1)."""
    p = dict(HH_PARAMS if params is None else params)
    f = I_ext if callable(I_ext) else (lambda t, _v=float(I_ext): _v)
    N = int(round(t_stop / dt))
    t = np.arange(N + 1) * dt
    out = np.empty((4, N + 1))
    V = float(V0)
    m, h, n = (float(x) for x in steady_state(V))
    out[:, 0] = V, m, h, n
    for i in range(N):
        ti = float(t[i])
        if method == "rk4":
            I0, Ih, I1 = f(ti), f(ti + dt / 2), f(ti + dt)
            k1 = _derivs_s(V, m, h, n, I0, p, q10)
            k2 = _derivs_s(*(y + dt / 2 * k for y, k in zip((V, m, h, n), k1)), Ih, p, q10)
            k3 = _derivs_s(*(y + dt / 2 * k for y, k in zip((V, m, h, n), k2)), Ih, p, q10)
            k4 = _derivs_s(*(y + dt * k for y, k in zip((V, m, h, n), k3)), I1, p, q10)
            V, m, h, n = (y + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
                          for y, a1, a2, a3, a4 in zip((V, m, h, n), k1, k2, k3, k4))
        elif method == "exp_euler":
            am, bm, ah, bh, an, bn = _rates_s(V)
            m = am / (am + bm) + (m - am / (am + bm)) * math.exp(-q10 * (am + bm) * dt)
            h = ah / (ah + bh) + (h - ah / (ah + bh)) * math.exp(-q10 * (ah + bh) * dt)
            n = an / (an + bn) + (n - an / (an + bn)) * math.exp(-q10 * (an + bn) * dt)
            V = V + dt * _derivs_s(V, m, h, n, f(ti), p, q10)[0]
        else:
            raise ValueError(f"unknown method '{method}'")
        out[:, i + 1] = V, m, h, n
    return {"t": t, "V": out[0], "m": out[1], "h": out[2], "n": out[3]}


def detect_spikes(t, V, threshold=0.0):
    """Upward threshold crossing times (linear interpolation)."""
    t, V = np.asarray(t), np.asarray(V)
    i = np.nonzero((V[:-1] < threshold) & (V[1:] >= threshold))[0]
    return t[i] + (t[i + 1] - t[i]) * (threshold - V[i]) / (V[i + 1] - V[i])


@REGISTRY.register("hh_neuron")
class HHNeuron:
    """ODEMechanism adapter. Inputs: I_ext_nA (nA, converted via area_cm2),
    I_syn_pA (pA, outward positive, subtracted). Spikes: V crossing 0 mV."""

    name = "hh_neuron"
    timescale_ms = 1.0
    confidence = CONFIDENCE
    params_meta = PARAMS_META
    spike_var = "V"
    spike_threshold_mV = 0.0

    def __init__(self, name="hh_neuron", area_cm2=1e-3, params=None):
        self.name = name
        self.area_cm2 = area_cm2
        self.p = dict(HH_PARAMS if params is None else params)

    def init_state(self, n, rng):
        V = np.full(n, -65.0)
        m, h, g = steady_state(V)
        return {"V": V, "m": m, "h": h, "n": g}

    def _I(self, inputs, like):
        I = np.zeros_like(like)
        if "I_ext_nA" in inputs:
            I = I + nA_to_density(inputs["I_ext_nA"], self.area_cm2)
        if "I_syn_pA" in inputs:
            I = I - nA_to_density(np.asarray(inputs["I_syn_pA"]) * 1e-3, self.area_cm2)
        return I

    def derivatives(self, t, s, inputs):
        dV, dm, dh, dn = derivs(s["V"], s["m"], s["h"], s["n"], self._I(inputs, s["V"]), self.p)
        return {"V": dV, "m": dm, "h": dh, "n": dn}

    def linear_coeffs(self, t, s, inputs):
        V, p = s["V"], self.p
        am, bm, ah, bh, an, bn = rates(V)
        gs = p["g_Na"] * s["m"] ** 3 * s["h"] + p["g_K"] * s["n"] ** 4 + p["g_L"]
        ge = (p["g_Na"] * s["m"] ** 3 * s["h"] * p["E_Na"] + p["g_K"] * s["n"] ** 4 * p["E_K"]
              + p["g_L"] * p["E_L"])
        return {"V": ((self._I(inputs, V) + ge) / p["C_m"], -gs / p["C_m"] * np.ones_like(V)),
                "m": (am, -(am + bm)), "h": (ah, -(ah + bh)), "n": (an, -(an + bn))}

    def outputs(self, s):
        return {"V": s["V"]}
