"""Local adapter for the Mechanism base class.

If ``neural.core.interfaces`` is published, it is used; otherwise a minimal
local ABC is defined. Only this file changes when the interfaces appear.
"""
from abc import ABC, abstractmethod

try:  # pragma: no cover - depends on the other branch
    from neural.core.interfaces import Mechanism
except ImportError:
    class Mechanism(ABC):  # type: ignore[no-redef]
        name = 'mechanism'
        status = 'implemented'
        reliability_class = 'incomplete'
        references: list[str] = []

        @abstractmethod
        def reset(self):
            """Return the internal state to its deterministic initial value."""

        @abstractmethod
        def state(self) -> dict:
            """Return a copy of the internal state."""
