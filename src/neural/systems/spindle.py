"""MuscleSpindle: aferenti mišićnog vretena Ia i II (faza 1, Prochazka-tip model).

Klasa pouzdanosti: S za strukturu (Ia dinamički + statički odgovor, II statički),
I za numeričke parametre.

Karakteristike (15 tačaka, skraćeno):
1. Naziv: MuscleSpindle.
2. Svrha: Ia kodira dužinu i brzinu, II uglavnom dužinu. (E)
3. Osnova: Prochazka & Gorassini 1998 (doi:10.1113/jphysiol.1998.sp021050);
   Mileusnic et al. 2006 (doi:10.1152/jn.00868.2005); Matthews 1972. Citate treba proveriti pre upisa u docs.
4. Matematika:
   Ia = gamma_d*k_v*sign(v)*|v|^exp + gamma_s*k_L*(L-1) + b_Ia, zatim ograničenje na >= 0;
   II = k_II*(L-1) + b_II, >= 0. L = L/L0 (bezdimenziono), v = dL/dt filtrirano (tau ~ 5 ms), u jedinicama L0/s.
5. Implementacija: stanje je prethodna dužina i filtrirana brzina; izlaz su stope u Hz;
   spajkovi su Poisson po aferentu sa seed-om i refraktarnošću 1 ms.
6. Ulazi: length (L/L0), dt (s), opciono gamma_d i gamma_s (skalari, podrazumevano 1.0).
7. Izlazi: step vraća (ia_hz, ii_hz); atributi spikes_ia (bool, N_ia=30) i spikes_ii (bool, N_ii=20).
8. Interno stanje: prethodna dužina, filtrirana brzina, RNG, refraktarni brojači.
9. Zavisnosti: numpy.
10. Interakcije: Ia hrani motoneurone (monosinaptički, preko sinapsi). II je kuka za polisinaptičku inhibiciju.
11. Konfiguracija: SpindleParams (k_v, k_L, b_Ia, exp, k_II, b_II, tau_v_s, refractory_s, n_ia, n_ii, seed).
12. Testovi: tests/contract/test_components.py (monotonost, brzina, nenegativnost).
13. Validacija: kvalitativno, ramp-and-hold (dinamički pik pa statički plato). Kvantitativna validacija nije tvrdnja.
14. Ograničenja: nema nelinearnosti intrafuzalnih vlakana (bag1/bag2/lanac), nema "initial burst",
    γ-motoneuroni nisu modelovani. Zamena: Mileusnic model u fazi 2 iza istog interfejsa.
15. Integracija: step se zove posle očitavanja LimbEnv, pre MotorPool. Nema globalnog stanja.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SpindleParams:
    k_v: float = 20.0         # Hz po (L0/s)^exp, dinamički dobitak Ia
    k_L: float = 300.0        # Hz po jedinici (L/L0 - 1), statički dobitak Ia
    b_Ia: float = 0.0         # Hz, bazalna stopa Ia
    exp: float = 0.6          # eksponent brzine
    k_II: float = 150.0       # Hz po jedinici (L/L0 - 1), statički dobitak II
    b_II: float = 0.0         # Hz, bazalna stopa II
    tau_v_s: float = 0.005    # s, filter brzine
    refractory_s: float = 0.001
    n_ia: int = 30
    n_ii: int = 20
    seed: int = 0


class MuscleSpindle:
    """Ia i II aferenti; vidi docstring modula za 15 tačaka."""

    def __init__(self, params: Optional[SpindleParams] = None):
        self.params = params or SpindleParams()
        self.reset()

    def reset(self) -> None:
        p = self.params
        self._prev_L: Optional[float] = None
        self._v_f = 0.0
        self._rng = np.random.default_rng(p.seed)
        self._refr_ia = np.zeros(p.n_ia)
        self._refr_ii = np.zeros(p.n_ii)
        self.spikes_ia = np.zeros(p.n_ia, dtype=bool)
        self.spikes_ii = np.zeros(p.n_ii, dtype=bool)
        self.ia_hz = 0.0
        self.ii_hz = 0.0

    def step(self, length: float, dt: float, gamma_d: float = 1.0,
             gamma_s: float = 1.0) -> Tuple[float, float]:
        """Vraća (ia_hz, ii_hz) >= 0. length je L/L0."""
        if dt <= 0:
            raise ValueError("dt mora biti pozitivan")
        p = self.params
        L = float(length)
        v = 0.0 if self._prev_L is None else (L - self._prev_L) / dt
        self._prev_L = L
        alpha = min(1.0, dt / p.tau_v_s)
        self._v_f += alpha * (v - self._v_f)
        vf = self._v_f

        dyn = gamma_d * p.k_v * np.sign(vf) * abs(vf) ** p.exp
        stat = gamma_s * p.k_L * (L - 1.0)
        ia = max(0.0, float(dyn + stat + p.b_Ia))
        ii = max(0.0, float(p.k_II * (L - 1.0) + p.b_II))

        self.ia_hz = ia
        self.ii_hz = ii
        self.spikes_ia = self._draw(ia, self._refr_ia, dt)
        self.spikes_ii = self._draw(ii, self._refr_ii, dt)
        return ia, ii

    def _draw(self, rate: float, refr: np.ndarray, dt: float) -> np.ndarray:
        """Poisson spajkovi sa refraktarnošću; RNG se troši u svakom koraku radi reproducibilnosti."""
        ready = refr <= 0.0
        u = self._rng.random(refr.size)
        spk = ready & (u < min(1.0, rate * dt))
        refr[:] = np.where(spk, self.params.refractory_s - dt, np.maximum(refr - dt, 0.0))
        return spk
