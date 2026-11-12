"""Hodgkin-Huxley Na/K/leak kinetics (modern sign convention).

V = V_in - V_out in mV, rest -65 mV, rates in 1/ms.
Source: Hodgkin & Huxley 1952 (DOI 10.1113/jphysiol.1952.sp004764). The
parameter values below are taken from the paper (converted to the modern
convention): E_Na = 50, E_K = -77, E_L = -54.387 mV, g_Na = 120, g_K = 36,
g_L = 0.3 mS/cm2, C_m = 1 uF/cm2, measured at 6.3 C.
"""
import math

import numpy as np

HH_PARAMS = {
    'C_m': 1.0,        # uF/cm2
    'g_Na': 120.0,     # mS/cm2
    'g_K': 36.0,       # mS/cm2
    'g_L': 0.3,        # mS/cm2
    'E_Na': 50.0,      # mV
    'E_K': -77.0,      # mV
    'E_L': -54.387,    # mV
    'V_rest': -65.0,   # mV
}
T_REF_CELSIUS = 6.3
Q10 = 3.0


def _out(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _vtrap(x, y):
    """x / (1 - exp(-x/y)) with the removable singularity at x = 0 resolved."""
    x = np.asarray(x, dtype=float)
    z = x / y
    small = np.abs(z) < 1e-6
    zs = np.where(small, 1.0, z)
    return np.where(small, y * (1.0 + z / 2.0), x / (1.0 - np.exp(-zs)))


def alpha_m(V):
    return _out(0.1 * _vtrap(np.asarray(V, dtype=float) + 40.0, 10.0))


def beta_m(V):
    return _out(4.0 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 18.0))


def alpha_h(V):
    return _out(0.07 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 20.0))


def beta_h(V):
    return _out(1.0 / (1.0 + np.exp(-(np.asarray(V, dtype=float) + 35.0) / 10.0)))


def alpha_n(V):
    return _out(0.01 * _vtrap(np.asarray(V, dtype=float) + 55.0, 10.0))


def beta_n(V):
    return _out(0.125 * np.exp(-(np.asarray(V, dtype=float) + 65.0) / 80.0))


def steady_state(V):
    """Return (m_inf, h_inf, n_inf) at V (mV)."""
    am, bm = alpha_m(V), beta_m(V)
    ah, bh = alpha_h(V), beta_h(V)
    an, bn = alpha_n(V), beta_n(V)
    return (_out(np.asarray(am) / (np.asarray(am) + np.asarray(bm))),
            _out(np.asarray(ah) / (np.asarray(ah) + np.asarray(bh))),
            _out(np.asarray(an) / (np.asarray(an) + np.asarray(bn))))


def temperature_factor(T_celsius):
    """phi = 3 ** ((T - 6.3) / 10)."""
    return _out(Q10 ** ((np.asarray(T_celsius, dtype=float) - T_REF_CELSIUS) / 10.0))


def _vtrap_scalar(x, y):
    z = x / y
    if abs(z) < 1e-6:
        return y * (1.0 + z / 2.0)
    return x / (1.0 - math.exp(-z))


def rates_scalar(V):
    """Fast scalar path: (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n)."""
    return (0.1 * _vtrap_scalar(V + 40.0, 10.0),
            4.0 * math.exp(-(V + 65.0) / 18.0),
            0.07 * math.exp(-(V + 65.0) / 20.0),
            1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0)),
            0.01 * _vtrap_scalar(V + 55.0, 10.0),
            0.125 * math.exp(-(V + 65.0) / 80.0))
