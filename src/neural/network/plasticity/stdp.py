"""Pair-based additive STDP (Bi & Poo 1998; Song, Miller & Abbott 2000).

dt = t_post - t_pre. Defaults are the Song 2000 values; the Bi & Poo
preset only changes the time constants (16.8 / 33.7 ms, TBD-verify).
"""
import math

import numpy as np

from neural.cellular._adapter import Mechanism
from neural.cellular import references as R


def stdp_window(dt_ms, A_plus, A_minus, tau_plus, tau_minus):
    d = np.asarray(dt_ms, dtype=float)
    pos = A_plus * np.exp(-np.where(d > 0.0, d, 0.0) / tau_plus)
    neg = -A_minus * np.exp(np.where(d < 0.0, d, 0.0) / tau_minus)
    w = np.where(d > 0.0, pos, np.where(d < 0.0, neg, 0.0))
    return float(w) if w.ndim == 0 else w


class PairSTDP(Mechanism):
    name = 'pair_stdp'
    status = 'implemented'
    reliability_class = 'strongly_supported'
    references = [R.BIPOO1998, R.SONG2000, R.BIPOO2001]

    def __init__(self, A_plus=0.005, A_minus=0.00525, tau_plus_ms=20.0, tau_minus_ms=20.0,
                 w0=0.5, w_min=0.0, w_max=1.0, mode='all_to_all'):
        if mode not in ('all_to_all', 'nearest'):
            raise ValueError("mode must be 'all_to_all' or 'nearest'")
        self.A_plus = float(A_plus)
        self.A_minus = float(A_minus)
        self.tau_plus_ms = float(tau_plus_ms)
        self.tau_minus_ms = float(tau_minus_ms)
        self.w0 = float(w0)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.mode = mode
        self.reset()

    @classmethod
    def bi_poo_1998(cls, **kw):
        kw.setdefault('tau_plus_ms', 16.8)
        kw.setdefault('tau_minus_ms', 33.7)
        return cls(**kw)

    def reset(self):
        self.w = self.w0
        self.x = 0.0
        self.y = 0.0
        self._t = None
        self.t_last_pre = None
        self.t_last_post = None
        self._x_before = 0.0
        self._y_before = 0.0

    def state(self):
        return {'w': self.w, 'x': self.x, 'y': self.y,
                't_last_pre': self.t_last_pre, 't_last_post': self.t_last_post}

    def _advance(self, t):
        if self._t is not None:
            d = t - self._t
            if d < 0.0:
                raise ValueError('event times must be non-decreasing')
            self.x *= math.exp(-d / self.tau_plus_ms)
            self.y *= math.exp(-d / self.tau_minus_ms)
        self._t = t

    def _clip(self):
        self.w = min(self.w_max, max(self.w_min, self.w))

    def on_pre(self, t_ms):
        self._advance(t_ms)
        y = self._y_before if self.t_last_post == t_ms else self.y
        self.w -= self.w_max * self.A_minus * y
        self._clip()
        self._x_before = self.x
        self.x = 1.0 if self.mode == 'nearest' else self.x + 1.0
        self.t_last_pre = t_ms

    def on_post(self, t_ms):
        self._advance(t_ms)
        x = self._x_before if self.t_last_pre == t_ms else self.x
        self.w += self.w_max * self.A_plus * x
        self._clip()
        self._y_before = self.y
        self.y = 1.0 if self.mode == 'nearest' else self.y + 1.0
        self.t_last_post = t_ms
