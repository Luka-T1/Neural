"""MotorPool: alfa-motoneuroni sa Henneman redosledom regrutovanja (LIF po jedinici).

Klasa pouzdanosti: S za redosled regrutovanja i strukturu, I za brojeve (opseg veličina, stope).

Karakteristike (15 tačaka, skraćeno):
1. Naziv: MotorPool.
2. Svrha: pretvara ukupan sinaptički pogon u spajkove, od malih ka velikim jedinicama. (E)
3. Osnova: Henneman 1965; Heckman & Enoka 2012 (doi:10.1002/cphy.c100087). Citate proveriti.
4. Matematika: LIF po jedinici, v' = (I_i - v)/tau_m, I_i = drive_gain*drive/thr_i (ulazna otpornost R_i ~ 1/size_i).
   Veličina je geometrijski niz (opseg 1:50); thr_i = size_i/size_max je rastući; thr[0] = 0.02 < 0.1.
   Sila trzaja twitch_peak_i = size_i/sum(size) (sumira se u 1).
5. Implementacija: vektorizovan numpy LIF, N = 50 jedinica, eksaktno integriranje za konstantan ulaz.
6. Ulazi: drive in [0,1] (ukupan sinaptički pogon), dt (s).
7. Izlazi: bool niz spajkova (n,); svojstva thresholds, sizes, twitch_peak.
8. Stanje: membranski potencijal, refraktarni brojač, RNG (samo ako je noise > 0).
9. Zavisnosti: numpy.
10. Interakcije: prima Ia preko sinapsi; kasnije Renshaw i Ib kuke. Spajkovi idu u filter ekscitacije i EMG.
11. Konfiguracija: n_units, size_ratio, tau_m_s, v_th, drive_gain, refractory_s, noise, seed.
12. Testovi: tests/contract/test_components.py (redosled pragova, podskup aktivnih jedinica, monotonost stope).
13. Validacija: kvalitativna; redosled regrutovanja i raspon stopa (maks. 40 Hz po refraktarnosti 25 ms).
14. Ograničenja: nema PIC-ova, dendritskih nelinearnosti, AHP je pojednostavljen.
    Zamena: višekompartmentni model u kasnijoj fazi.
15. Integracija: poziva se posle sinapsi i pre HillMuscle. Nema globalnog stanja.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MotorPool:
    """Vektorizovani LIF motoneuroni; vidi docstring modula za 15 tačaka."""

    def __init__(self, n_units: int = 50, seed: int = 0, size_ratio: float = 50.0,
                 tau_m_s: float = 0.02, v_th: float = 1.0, drive_gain: float = 1.25,
                 refractory_s: float = 0.025, noise: float = 0.0):
        if n_units < 2:
            raise ValueError("n_units mora biti >= 2")
        if size_ratio <= 1.0:
            raise ValueError("size_ratio mora biti > 1")
        self.n_units = int(n_units)
        self.seed = int(seed)
        self.tau_m_s = float(tau_m_s)
        self.v_th = float(v_th)
        self.drive_gain = float(drive_gain)
        self.refractory_s = float(refractory_s)
        self.noise = float(noise)
        self._sizes = np.geomspace(1.0, float(size_ratio), self.n_units)
        self._thr = self._sizes / self._sizes[-1]
        self._twitch = self._sizes / self._sizes.sum()
        self.reset()

    @property
    def thresholds(self) -> np.ndarray:
        return self._thr.copy()

    @property
    def sizes(self) -> np.ndarray:
        return self._sizes.copy()

    @property
    def twitch_peak(self) -> np.ndarray:
        return self._twitch.copy()

    def reset(self) -> None:
        self._v = np.zeros(self.n_units)
        self._refr = np.zeros(self.n_units)
        self._rng = np.random.default_rng(self.seed)

    def drive_to_current(self, drive: float) -> np.ndarray:
        """Normalizovan pogon u ulaznu struju po jedinici (stacionarni ulaz LIF-a)."""
        d = float(np.clip(drive, 0.0, 1.0))
        return self.drive_gain * d / self._thr

    def step(self, drive: float, dt: float) -> np.ndarray:
        """Vraća bool niz spajkova (n_units,) za ovaj korak."""
        if dt <= 0:
            raise ValueError("dt mora biti pozitivan")
        I = self.drive_to_current(drive)
        if self.noise > 0.0:
            I = I + self.noise * self._rng.standard_normal(self.n_units)
        decay = np.exp(-dt / self.tau_m_s)
        self._v = I + (self._v - I) * decay
        ready = self._refr <= 0.0
        spk = ready & (self._v >= self.v_th)
        self._v[spk] = 0.0
        self._refr = np.where(spk, self.refractory_s - dt, np.maximum(self._refr - dt, 0.0))
        return spk
