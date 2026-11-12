"""Protokoli senzora i motora: re-export iz neural.core.interfaces.

Sensor dobija SAMO Observation. Motor vraća akciju (komande aktuatora u [0,1]).
"""

from __future__ import annotations

from neural.core.interfaces import Motor, Observation, Sensor

__all__ = ["Sensor", "Motor", "Observation"]
