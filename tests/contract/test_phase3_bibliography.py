"""Ugovor faze 3: bibliografija novih izvora i njihova izolovanost od mehanizama.

Ne oslanja se na docs/experiment_design.md.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BIB = ROOT / "docs" / "bibliography.md"
MECHS = ["hh", "synapses", "stdp", "da_stdp", "homeostasis", "cpg"]
PHASE3_KEYS = ["Bilbao2018", "ShapereWilczek1989", "Hogan1984", "LaugaPowers2009"]


def _entries():
    text = BIB.read_text(encoding="utf-8")
    return {
        m.group(1): m.group(0)
        for m in re.finditer(r"^- \[([A-Za-z0-9_]+)\].*$", text, flags=re.M)
    }


@pytest.mark.parametrize("key", PHASE3_KEYS)
def test_phase3_key_in_bibliography(key):
    assert key in _entries(), f"docs/bibliography.md nema unos [{key}]"


@pytest.mark.parametrize("key", PHASE3_KEYS)
def test_phase3_entry_has_doi_field(key):
    entry = _entries().get(key)
    assert entry is not None, key
    assert not re.search(r"(DOI|ISBN):\s*TBD", entry), entry
    assert re.search(r"DOI: 10\.\d{4,9}/\S+|DOI nije proveren|nema DOI", entry), entry
    assert re.search(r"doi_status:\s*(needs-verify|verified|none)", entry), entry
    assert re.search(r"\(\d{4}\)", entry), entry


def test_phase3_keys_unique():
    text = BIB.read_text(encoding="utf-8")
    for key in PHASE3_KEYS:
        assert len(re.findall(rf"^- \[{key}\]", text, flags=re.M)) <= 1, key


@pytest.mark.parametrize("mech", MECHS)
def test_params_meta_sources_do_not_use_phase3_keys(mech):
    mod = importlib.import_module(f"neural.mechanisms.{mech}")
    for name, meta in mod.PARAMS_META.items():
        toks = {t.strip() for t in meta["source"].split(";")}
        assert not toks & set(PHASE3_KEYS), (mech, name, toks)
        for key in PHASE3_KEYS:
            assert key not in meta["source"], (mech, name, key)
