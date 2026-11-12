"""Unbranched multi-compartment cable (Rall) with passive or HH membrane.

c_m dV_i/dt = -i_ion,i + (1/A_i) sum_j G_ij (V_j - V_i) + 1e-3 I_inj,i / A_i
Backward Euler (tridiagonal, Thomas algorithm), sealed ends.
Units: um, ohm*cm, uF/cm2, mS/cm2, mV, ms; injected current in nA.
"""
import math

import numpy as np

from neural.cellular._adapter import Mechanism
from neural.cellular import references as R
from neural.molecular.channels import hh as K

NO_END = 1e300  # "never stops"


def _thomas(lower, diag, upper, rhs):
    """Solve a tridiagonal system; lower/upper have length n-1."""
    n = diag.size
    c = np.empty(n - 1)
    d = np.empty(n)
    c_prev = 0.0
    d_prev = 0.0
    for i in range(n):
        lo = lower[i - 1] if i > 0 else 0.0
        den = diag[i] - lo * c_prev
        if i < n - 1:
            c[i] = upper[i] / den
            c_prev = c[i]
        d[i] = (rhs[i] - lo * d_prev) / den
        d_prev = d[i]
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


class Cable(Mechanism):
    name = 'cable'
    status = 'implemented'
    reliability_class = 'strongly_supported'
    references = [R.RALL1959, R.HH1952, R.HINES1984]

    def __init__(self, n_comp, length_um, diam_um, Ra_ohm_cm=35.4, cm_uF_cm2=1.0,
                 active='hh', g_leak_mS_cm2=0.3, E_leak_mV=None, temperature_celsius=6.3):
        if active not in ('hh', 'passive'):
            raise ValueError("active must be 'hh' or 'passive'")
        self.n_comp = int(n_comp)
        self.active = active
        self.cm = float(cm_uF_cm2)
        self.g_leak = float(g_leak_mS_cm2)
        if E_leak_mV is None:
            E_leak_mV = HH_E_L if active == 'hh' else -65.0
        self.E_leak = float(E_leak_mV)
        self.temperature_celsius = float(temperature_celsius)
        self.phi = float(K.temperature_factor(self.temperature_celsius))
        dx_um = length_um / self.n_comp
        self.x_um = (np.arange(self.n_comp) + 0.5) * dx_um
        d_cm = diam_um * 1e-4
        dx_cm = dx_um * 1e-4
        self.area_cm2 = np.full(self.n_comp, math.pi * d_cm * dx_cm)
        g_axial_mS = 1e3 * math.pi * d_cm ** 2 / (4.0 * Ra_ohm_cm * dx_cm)
        self.k_axial = g_axial_mS / self.area_cm2[0]  # mS/cm2, per neighbour
        self._inj = []
        self.reset()

    def reset(self):
        if self.active == 'hh':
            self.V = np.full(self.n_comp, K.HH_PARAMS['V_rest'])
            m, h, n = K.steady_state(self.V)
            self.m, self.h, self.n = np.array(m), np.array(h), np.array(n)
        else:
            self.V = np.full(self.n_comp, self.E_leak)
        self.t = 0.0

    def state(self):
        s = {'V': self.V.copy(), 't': self.t}
        if self.active == 'hh':
            s.update(m=self.m.copy(), h=self.h.copy(), n=self.n.copy())
        return s

    def inject(self, index, amplitude_nA, t_start_ms=0.0, t_stop_ms=NO_END):
        """Add a constant current (cumulative with earlier calls) in a time window."""
        self._inj.append((int(index), float(amplitude_nA), float(t_start_ms), float(t_stop_ms)))

    def _injected(self, t):
        I = np.zeros(self.n_comp)
        for idx, amp, t0, t1 in self._inj:
            if t0 <= t < t1:
                I[idx] += amp
        return 1e-3 * I / self.area_cm2  # uA/cm2

    def _step(self, dt, t):
        n = self.n_comp
        cm = self.cm
        if self.active == 'hh':
            P = K.HH_PARAMS
            am, bm = K.alpha_m(self.V), K.beta_m(self.V)
            ah, bh = K.alpha_h(self.V), K.beta_h(self.V)
            an, bn = K.alpha_n(self.V), K.beta_n(self.V)
            for attr, a, b in (('m', am, bm), ('h', ah, bh), ('n', an, bn)):
                a = np.asarray(a)
                b = np.asarray(b)
                x_ss = a / (a + b)
                x = getattr(self, attr)
                setattr(self, attr, x_ss + (x - x_ss) * np.exp(-dt * self.phi * (a + b)))
            gna = P['g_Na'] * self.m ** 3 * self.h
            gk = P['g_K'] * self.n ** 4
            gl = self.g_leak
            gtot = gna + gk + gl
            gE = gna * P['E_Na'] + gk * P['E_K'] + gl * self.E_leak
        else:
            gtot = np.full(n, self.g_leak)
            gE = np.full(n, self.g_leak * self.E_leak)
        rhs = cm / dt * self.V + gE + self._injected(t)
        if n == 1:
            self.V = rhs / (cm / dt + gtot)
        else:
            k = self.k_axial
            nb = np.full(n, 2.0)
            nb[0] = nb[-1] = 1.0
            diag = cm / dt + gtot + k * nb
            off = np.full(n - 1, -k)
            self.V = _thomas(off, diag, off, rhs)
        self.t += dt

    def simulate(self, t_stop_ms, dt):
        """Return (t, V) with t[0] = 0 and V[0] the present state."""
        steps = int(round(t_stop_ms / dt))
        t = dt * np.arange(steps + 1)
        V = np.empty((steps + 1, self.n_comp))
        V[0] = self.V
        for i in range(steps):
            self._step(dt, t[i])
            V[i + 1] = self.V
        return t, V


HH_E_L = K.HH_PARAMS['E_L']
