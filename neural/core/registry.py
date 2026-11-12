"""Mechanism registry and contract validation."""
from __future__ import annotations

from .interfaces import CONFIDENCE_LEVELS, ContractError


class MechanismRegistry:
    def __init__(self) -> None:
        self._d: dict[str, type] = {}

    def register(self, name: str):
        def deco(cls):
            if name in self._d:
                raise ValueError(f"mechanism '{name}' already registered")
            self._d[name] = cls
            return cls
        return deco

    def get(self, name: str):
        if name not in self._d:
            raise KeyError(name)
        return self._d[name]

    def create(self, name: str, **kw):
        return self.get(name)(**kw)

    def names(self) -> list[str]:
        return sorted(self._d)


REGISTRY = MechanismRegistry()


def validate_mechanism(m) -> None:
    name = getattr(m, "name", None)
    if not isinstance(name, str) or not name:
        raise ContractError("mechanism needs a non-empty 'name'")
    ts = getattr(m, "timescale_ms", None)
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not ts > 0:
        raise ContractError(f"{name}: timescale_ms must be > 0")
    if getattr(m, "confidence", None) not in CONFIDENCE_LEVELS:
        raise ContractError(f"{name}: confidence must be one of {CONFIDENCE_LEVELS}")
    pm = getattr(m, "params_meta", None)
    if not isinstance(pm, dict):
        raise ContractError(f"{name}: params_meta must be a dict")
    for p, meta in pm.items():
        if not isinstance(meta, dict) or not all(k in meta for k in ("value", "unit", "source")):
            raise ContractError(f"{name}: params_meta['{p}'] needs value, unit, source")
    if not callable(getattr(m, "derivatives", None)) and not callable(getattr(m, "step", None)):
        raise ContractError(f"{name}: needs derivatives or step")
    if not callable(getattr(m, "init_state", None)):
        raise ContractError(f"{name}: needs init_state")
