"""Pomoćne funkcije za propriocepciju i filtriranje aktivacija.

Svet (environment) ih koristi da iz SOPSTVENOG privatnog stanja aktuatora
izračuna proprio niz. Nema koordinata i nema RNG-a: šum dolazi spolja.
"""

from __future__ import annotations

import numpy as np

CONFIDENCE = "incomplete"
PARAMS_META = {
    "tau_act_ms": {
        "value": 50.0,
        "unit": "ms",
        "source": "ASSUMPTION: vremenska konstanta filtriranja aktivacije (dizajn eksperimenta)",
        "confidence": "speculative",
    },
}


def activation_step(
    a: np.ndarray, u: np.ndarray, dt_ms: float, tau_act_ms: float
) -> np.ndarray:
    """Euler korak niskopropusnog filtra: a <- a + dt/tau * (u - a)."""
    a = np.asarray(a, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return a + (dt_ms / tau_act_ms) * (u - a)


def proprio_vector(a: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Propriocepcija: aktivacije aktuatora plus već skalirani šum."""
    return np.asarray(a, dtype=np.float64) + np.asarray(noise, dtype=np.float64)
