"""Tsodyks-Markram short-term plasticity, event driven and exact between spikes."""
import math

from neural.cellular._adapter import Mechanism
from neural.cellular import references as R


class TsodyksMarkramSynapse(Mechanism):
    name = 'tsodyks_markram'
    status = 'implemented'
    reliability_class = 'strongly_supported'
    references = [R.TM1997, R.MARKRAM1998]

    def __init__(self, U, tau_rec_ms, tau_facil_ms, A=1.0):
        self.U = float(U)
        self.tau_rec_ms = float(tau_rec_ms)
        self.tau_facil_ms = float(tau_facil_ms)
        self.A = float(A)
        self.reset()

    def reset(self):
        self.u = 0.0
        self.R = 1.0
        self.t_last = None

    def state(self):
        return {'u': self.u, 'R': self.R, 't_last': self.t_last}

    def on_spike(self, t_ms):
        if self.t_last is None:
            u_minus, r_minus = 0.0, 1.0
        else:
            d = t_ms - self.t_last
            if d < 0.0:
                raise ValueError('spike times must be non-decreasing')
            u_minus = self.u * math.exp(-d / self.tau_facil_ms) if self.tau_facil_ms > 0.0 else 0.0
            r_minus = (1.0 - (1.0 - self.R) * math.exp(-d / self.tau_rec_ms)
                       if self.tau_rec_ms > 0.0 else 1.0)
        u_plus = u_minus + self.U * (1.0 - u_minus)
        self.u = u_plus
        self.R = r_minus - u_plus * r_minus
        self.t_last = t_ms
        return self.A * u_plus * r_minus
