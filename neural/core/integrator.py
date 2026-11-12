"""Pure integrators (state = dict[str, ndarray])."""
from __future__ import annotations

import numpy as np

from .interfaces import ContractError, State


def _axpy(y: State, k: State, h: float) -> State:
    return {key: y[key] + h * k[key] for key in y}


def rk4_step(f, t: float, y: State, dt: float) -> State:
    k1 = f(t, y)
    k2 = f(t + dt / 2, _axpy(y, k1, dt / 2))
    k3 = f(t + dt / 2, _axpy(y, k2, dt / 2))
    k4 = f(t + dt, _axpy(y, k3, dt))
    return {key: y[key] + dt / 6 * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) for key in y}


def exp_euler_step(coeffs: dict, t: float, y: State, dt: float) -> State:
    """dx/dt = A + B x  ->  x + (A + B x) expm1(B dt)/B ; B == 0 -> x + A dt.
    State keys without coefficients raise ContractError."""
    out = {}
    for key, x in y.items():
        if key not in coeffs:
            raise ContractError(f"exp_euler: missing coefficients for '{key}'")
        A, B = coeffs[key]
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        zero = B == 0
        Bs = np.where(zero, 1.0, B)
        fac = np.where(zero, dt, np.expm1(Bs * dt) / Bs)
        out[key] = x + (A + B * x) * fac
    return out


def detect_threshold_crossings(v_prev, v_next, threshold, t_prev, dt):
    v_prev = np.atleast_1d(np.asarray(v_prev, dtype=np.float64))
    v_next = np.atleast_1d(np.asarray(v_next, dtype=np.float64))
    idx = np.nonzero((v_prev < threshold) & (threshold <= v_next))[0]
    t = t_prev + dt * (threshold - v_prev[idx]) / (v_next[idx] - v_prev[idx])
    return idx, t
