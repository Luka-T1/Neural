"""Contract: every mechanism module exposes CONFIDENCE and a well-formed, traceable PARAMS_META."""
import importlib
import re
from pathlib import Path

import pytest

MODULES = ["hh", "synapses", "stdp", "da_stdp", "homeostasis", "cpg"]
LEVELS = {"established", "strongly_supported", "incomplete", "hypothesis", "speculative"}
BIB = Path(__file__).resolve().parents[2] / "docs" / "bibliography.md"
KEYS = set(re.findall(r"^- \[([^\]]+)\]", BIB.read_text(encoding="utf-8"), re.M))


def _mod(name):
    return importlib.import_module(f"neural.mechanisms.{name}")


def test_bibliography_has_keys():
    assert KEYS


@pytest.mark.parametrize("name", MODULES)
def test_confidence_allowed(name):
    assert _mod(name).CONFIDENCE in LEVELS


@pytest.mark.parametrize("name", MODULES)
def test_params_meta_complete_and_traceable(name):
    meta = _mod(name).PARAMS_META
    assert isinstance(meta, dict) and meta
    for p, m in meta.items():
        for k in ("value", "unit", "source", "confidence"):
            assert k in m, f"{name}.{p} lacks '{k}'"
        assert m["confidence"] in LEVELS, f"{name}.{p}"
        parts = [s.strip() for s in str(m["source"]).split(";") if s.strip()]
        assert parts, f"{name}.{p} has empty source"
        for s in parts:
            assert s.startswith("ASSUMPTION") or s in KEYS, f"{name}.{p}: unknown source '{s}'"
