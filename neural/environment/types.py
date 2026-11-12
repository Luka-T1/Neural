"""Tipovi okruženja i ugovor između sveta i tela.

Ugovorni tipovi (Observation, Probe, Environment, SupportsProbe) dolaze iz
neural.core.interfaces i ovde se samo re-exportuju. Lokalno ostaje samo
ProbeReading, koji služi isključivo harnessu.

Organizam (neural/body) prima isključivo Observation. Koordinate, dimenzija i
Probe su dostupni samo harnessu.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural.core.interfaces import Environment, Observation, Probe, SupportsProbe


@dataclass(frozen=True)
class ProbeReading:
    """Očitavanje za harness (nikad ne ide organizmu)."""

    x: np.ndarray
    x_perp_norm: float
    escaped: bool
    t_escape_ms: float | None
    t_ms: float


__all__ = ["Observation", "Probe", "Environment", "SupportsProbe", "ProbeReading"]
