"""Configuration loading."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
_TOL = 1e-9


def _is_int(x: float) -> bool:
    return abs(x - round(x)) <= _TOL * max(1.0, abs(x))


@dataclass(frozen=True)
class Config:
    seed: int
    dt_ms: float = 0.025
    t_stop_ms: float = 1000.0
    control_period_ms: float = 1.0
    default_method: str = "rk4"
    mechanisms: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    organism: dict = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dt_ms > 0:
            raise ValueError("dt_ms must be > 0")
        if self.default_method not in ("rk4", "exp_euler"):
            raise ValueError("default_method must be 'rk4' or 'exp_euler'")
        if not self.control_period_ms > 0 or not _is_int(self.control_period_ms / self.dt_ms):
            raise ValueError("control_period_ms must be a positive integer multiple of dt_ms")

    @property
    def steps_per_control(self) -> int:
        return int(round(self.control_period_ms / self.dt_ms))

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown config keys: {sorted(bad)}")
        return cls(**copy.deepcopy(d))

    def to_dict(self) -> dict:
        return {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}


def _mini_yaml(text: str) -> dict:
    """Fallback for flat config files when PyYAML is unavailable.

    Supports only flat ``key: int|float|str|{}`` lines (plus comments and blank
    lines). It understands no bool, null or quotes (values stay raw strings),
    and raises ValueError on nesting, lists, lines without ':' and keys without
    a value. For anything more complex install PyYAML: ``pip install -e .[yaml]``.
    """
    out: dict[str, Any] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line[0] in " \t":
            raise ValueError(f"line {n}: nesting/indentation not supported without PyYAML: {raw!r}")
        if line.startswith("- "):
            raise ValueError(f"line {n}: lists not supported without PyYAML: {raw!r}")
        if ":" not in line:
            raise ValueError(f"line {n}: expected 'key: value', got {raw!r}")
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if not v:
            raise ValueError(f"line {n}: key {k!r} has no value (nesting not supported without PyYAML)")
        if v == "{}":
            out[k] = {}
            continue
        for conv in (int, float):
            try:
                out[k] = conv(v)
                break
            except ValueError:
                pass
        else:
            out[k] = v
    return out


def _deep_merge(a: dict, b: dict) -> dict:
    r = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = _deep_merge(r[k], v)
        else:
            r[k] = copy.deepcopy(v)
    return r


def load_config(path=None, overrides: dict | None = None) -> Config:
    text = Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs; optional dependency
        d = yaml.safe_load(text) or {}
    except ImportError:  # PyYAML is optional
        d = _mini_yaml(text)
    if overrides:
        d = _deep_merge(d, overrides)
    return Config.from_dict(d)
