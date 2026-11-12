"""Single-compartment Hodgkin-Huxley neuron.

Gates: exponential Euler (Rush-Larsen) at V^n; voltage: backward Euler with
conductances frozen within the step. Units: mV, ms, uF/cm2, mS/cm2, uA/cm2.
"""
import math

import numpy as np

from neural.cellular._adapter import Mechanism
from neural.cellular import references as R
from neural.molecular.channels.hh import HH_PARAMS, rates_scalar, steady_state, temperature_factor


class HHNeuron(Mechanism):
    name = 'hh_neuron'
    status = 'implemented'
    reliability_class = 'strongly_supported'
    references = [R.HH1952, R.HH1952_CURRENTS, R.HH1952_CONDUCTANCE]

    def __init__(self, temperature_celsius=6.3, params=None):
        self.p = dict(HH_PARAMS)
        if params:
            self.p.update(params)
        self.temperature_celsius = float(temperature_celsius)
        self.phi = float(temperature_factor(self.temperature_celsius))
        self.reset()

    def reset(self, V0=-65.0):
        self.V = float(V0)
        self.m, self.h, self.n = (float(x) for x in steady_state(self.V))
        self.t = 0.0

    def state(self):
        return {'V': self.V, 'm': self.m, 'h': self.h, 'n': self.n, 't': self.t}

    def currents(self):
        """Ionic currents (uA/cm2, outward positive) at the present state."""
        p = self.p
        return {'I_Na': p['g_Na'] * self.m ** 3 * self.h * (self.V - p['E_Na']),
                'I_K': p['g_K'] * self.n ** 4 * (self.V - p['E_K']),
                'I_L': p['g_L'] * (self.V - p['E_L'])}

    def step(self, dt, I_ext=0.0):
        p = self.p
        am, bm, ah, bh, an, bn = rates_scalar(self.V)
        phi = self.phi
        for attr, a, b in (('m', am, bm), ('h', ah, bh), ('n', an, bn)):
            rate = phi * (a + b)
            x_ss = a / (a + b)
            x = getattr(self, attr)
            setattr(self, attr, x_ss + (x - x_ss) * math.exp(-dt * rate))
        gna = p['g_Na'] * self.m ** 3 * self.h
        gk = p['g_K'] * self.n ** 4
        gl = p['g_L']
        cm = p['C_m']
        num = cm * self.V + dt * (gna * p['E_Na'] + gk * p['E_K'] + gl * p['E_L'] + I_ext)
        self.V = num / (cm + dt * (gna + gk + gl))
        self.t += dt
        return self.V


def simulate_hh(I_ext, t_stop_ms, dt=0.01, temperature_celsius=6.3):
    """Simulate; I_ext is a float or callable(t_ms) in uA/cm2. Returns (t, V)."""
    neuron = HHNeuron(temperature_celsius=temperature_celsius)
    n = int(round(t_stop_ms / dt))
    t = dt * np.arange(n + 1)
    V = np.empty(n + 1)
    V[0] = neuron.V
    if callable(I_ext):
        for k in range(n):
            V[k + 1] = neuron.step(dt, float(I_ext(k * dt)))
    else:
        I = float(I_ext)
        for k in range(n):
            V[k + 1] = neuron.step(dt, I)
    return t, V
