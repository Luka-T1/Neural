"""HillMuscle: Hill-tip mišić (ekscitacija, dužina, brzina -> sila).

Klasa pouzdanosti: S za strukturu (aktivna i pasivna komponenta, Hill hiperbola),
I za parametre.

Karakteristike (15 tačaka, skraćeno):
1. Naziv: HillMuscle.
2. Svrha: pretvara ekscitaciju, dužinu i brzinu u silu. (E)
3. Osnova: Hill 1938; Zajac 1989 (Crit Rev Biomed Eng 17:359); Millard et al. 2013 (doi:10.1115/1.4023390).
4. Matematika:
   F = F_max*[a*f_L(l)*f_V(v) + f_PE(l)], a' = (u - a)/tau, tau = tau_act (u > a) ili tau_deact.
   f_L = exp(-((l/l_opt - 1)/w)^2), w = 0.45.
   f_V: skraćivanje s = -v/v_max in [0,1]: (1-s)/(1+4s), =1 u izometriji, 0 pri s=1;
        izduživanje e = v/v_max: 1.8 - 0.8*exp(-e/0.3) (<= 1.8).
   f_PE = k_pe*(exp(c*x) - 1)/(exp(c*x_ref) - 1), x = (l - l_slack)/l_opt, x_ref = (l_opt - l_slack)/l_opt, 0 za l <= l_slack.
5. Implementacija: eksplicitni Euler za aktivaciju (dt <= 1 ms); tetiva je kruta aproksimacija (u fazi 1 nije modelovana).
6. Ulazi: excitation in [0,1], length (m), velocity (m/s; skraćivanje je negativno), dt (s).
7. Izlaz: sila >= 0 u N.
8. Stanje: aktivacija a in [0,1].
9. Zavisnosti: numpy.
10. Interakcije: sila ide u LimbEnv kao moment (r*(F_flex - F_ext)). Parovi agonist/antagonist su dva objekta.
11. Konfiguracija: f_max, l_opt, v_max, l_slack, tau_act, tau_deact, k_pe, c_pe, width.
12. Testovi: tests/contract/test_components.py (maksimum f_L, redosled sila, pasivna sila, postepena aktivacija).
13. Validacija: kvalitativno, poređenje sa literaturnim krivama (nije kvantitativna tvrdnja).
14. Ograničenja: nema elastičnosti tetive, zavisnosti od istorije, zamora. Zamena: Millard model sa tetivom.
15. Integracija: poziva se posle MotorPool. Ne zna ništa o MuJoCo-u.
"""
from __future__ import annotations

import math


class HillMuscle:
    """Hill-tip mišić; vidi docstring modula za 15 tačaka."""

    _E_C = 0.3  # karakteristična brzina izduživanja (normalizovana)

    def __init__(self, f_max: float, l_opt: float, v_max: float, l_slack: float,
                 tau_act: float = 0.01, tau_deact: float = 0.04,
                 k_pe: float = 0.05, c_pe: float = 4.0, width: float = 0.45):
        if l_slack >= l_opt:
            raise ValueError("l_slack mora biti manji od l_opt")
        self.f_max = float(f_max)
        self.l_opt = float(l_opt)
        self.v_max = float(v_max)
        self.l_slack = float(l_slack)
        self.tau_act = float(tau_act)
        self.tau_deact = float(tau_deact)
        self.k_pe = float(k_pe)
        self.c_pe = float(c_pe)
        self.width = float(width)
        self.a = 0.0

    def reset(self) -> None:
        self.a = 0.0

    def _activation(self, excitation: float, dt: float) -> float:
        u = min(1.0, max(0.0, float(excitation)))
        tau = self.tau_act if u > self.a else self.tau_deact
        self.a = min(1.0, max(0.0, self.a + dt * (u - self.a) / tau))
        return self.a

    def _f_length(self, length: float) -> float:
        return math.exp(-((length / self.l_opt - 1.0) / self.width) ** 2)

    def _f_velocity(self, velocity: float) -> float:
        vn = velocity / self.v_max
        if vn <= 0.0:
            s = min(1.0, -vn)
            return (1.0 - s) / (1.0 + 4.0 * s)
        return 1.8 - 0.8 * math.exp(-vn / self._E_C)

    def _f_passive(self, length: float) -> float:
        if length <= self.l_slack:
            return 0.0
        x = (length - self.l_slack) / self.l_opt
        x_ref = (self.l_opt - self.l_slack) / self.l_opt
        return self.k_pe * (math.exp(self.c_pe * x) - 1.0) / (math.exp(self.c_pe * x_ref) - 1.0)

    def step(self, excitation: float, length: float, velocity: float, dt: float) -> float:
        """Vraća silu u N (>= 0). length u m, velocity u m/s (skraćivanje < 0)."""
        if dt <= 0:
            raise ValueError("dt mora biti pozitivan")
        a = self._activation(excitation, dt)
        active = a * self._f_length(length) * self._f_velocity(velocity)
        force = self.f_max * (active + self._f_passive(length))
        return max(0.0, float(force))
