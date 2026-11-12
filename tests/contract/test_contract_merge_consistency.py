"""Konzistentnost posle spajanja: doi_status, tabele specifikacija, bibliografija, sledljivost."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MECHS = ["hh", "synapses", "stdp", "da_stdp", "homeostasis", "cpg"]
DOI_ALLOWED = {"needs-verify", "verified", "none"}
_TEST_TOK = re.compile(r"`?(tests/[^\s`|:]+\.py)::(test_\w+)`?")


def _meta(name):
    return importlib.import_module(f"neural.mechanisms.{name}").PARAMS_META


@pytest.mark.parametrize("mech", MECHS)
def test_doi_status_vocabulary(mech):
    for key, meta in _meta(mech).items():
        status = meta.get("doi_status")
        assert status is None or status in DOI_ALLOWED, f"{mech}.{key}: {status!r}"
        assert "TBD" not in str(status)


def test_no_tbd_in_code_or_bibliography():
    for p in list((ROOT / "neural" / "mechanisms").glob("*.py")) + [ROOT / "docs" / "bibliography.md"]:
        assert "TBD-verify" not in p.read_text(encoding="utf-8"), p.name
    assert "TBD" not in (ROOT / "docs" / "bibliography.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("mech", MECHS)
def test_spec_table_names_equal_params_meta(mech):
    text = (ROOT / "docs" / "specs" / f"{mech}.md").read_text(encoding="utf-8")
    m = re.search(r"^## 5\..*?$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    assert m
    names = set()
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith("|"):
            cell = line.strip("|").split("|")[0].strip()
            if cell and cell != "ime" and not set(cell) <= set("-: "):
                names.add(cell)
    assert names == set(_meta(mech)), names ^ set(_meta(mech))


def test_bibliography_covers_sources():
    bib = (ROOT / "docs" / "bibliography.md").read_text(encoding="utf-8")
    keys = set(re.findall(r"^- \[([A-Za-z0-9_]+)\]", bib, flags=re.M))
    for mech in MECHS:
        for key, meta in _meta(mech).items():
            src = meta["source"]
            if src.startswith("ASSUMPTION"):
                continue
            for tok in re.findall(r"[A-Za-z]+\d{4}[a-z]?", src):
                assert tok in keys, f"{mech}.{key}: {tok} nije u bibliografiji"


def test_traceability_rows_reference_existing_tests():
    text = (ROOT / "docs" / "traceability.md").read_text(encoding="utf-8")
    for path, name in _TEST_TOK.findall(text):
        f = ROOT / path
        assert f.exists(), path
        assert re.search(rf"def {name}\b", f.read_text(encoding="utf-8")), f"{path}::{name}"
