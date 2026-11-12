"""Ugovor faze 3: status DOI-ja i usklađenost PHASE3_KEYS sa docs/bibliography.md."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.contract.test_phase3_bibliography import PHASE3_KEYS

BIB = Path(__file__).resolve().parents[2] / "docs" / "bibliography.md"
TEXT = BIB.read_text(encoding="utf-8")


def _blocks():
    parts = re.split(r"(?m)^(?=- \[)", TEXT)
    return {m.group(1): p for p in parts if (m := re.match(r"- \[([A-Za-z0-9_]+)\]", p))}


def test_no_tbd():
    assert "TBD" not in TEXT


@pytest.mark.parametrize("key", PHASE3_KEYS)
def test_status_consistent(key):
    block = _blocks()[key]
    m = re.search(r"doi_status:\s*(needs-verify|verified|none)", block)
    assert m, key
    if m.group(1) == "needs-verify":
        assert "Razlog:" in block, f"{key}: needs-verify bez podstavke Razlog"
    if m.group(1) == "verified":
        assert re.search(r"DOI: 10\.\d{4,9}/\S+", block), key


def test_phase3_keys_match_entries():
    known = {"Bilbao2018", "ShapereWilczek1989", "Hogan1984", "LaugaPowers2009"}
    assert set(PHASE3_KEYS) <= known
    for k in known - set(PHASE3_KEYS):
        assert k not in _blocks(), f"{k} je u bibliografiji ali ne u PHASE3_KEYS"


def test_hogan_is_ieee_tac():
    assert "IEEE Trans Autom Control 29(8):681–690" in _blocks()["Hogan1984"]
