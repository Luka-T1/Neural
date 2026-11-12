"""Okruženja: ChannelWorld (eksperiment dimension_escape) i NDWorld (opšti N-D svet).

Bez MuJoCo zavisnosti; samo numpy i stdlib.
"""

from neural.environment.channel_world import ChannelWorld, WorldConfig
from neural.environment.nd_world import NDWorld
from neural.environment.types import (
    Environment,
    Observation,
    Probe,
    ProbeReading,
    SupportsProbe,
)

N_ACTUATORS = 4
N_SENSORS = 9

__all__ = [
    "N_ACTUATORS",
    "N_SENSORS",
    "WorldConfig",
    "ChannelWorld",
    "NDWorld",
    "Observation",
    "ProbeReading",
    "Probe",
    "Environment",
    "SupportsProbe",
]
