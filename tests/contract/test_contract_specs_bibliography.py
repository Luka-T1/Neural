"""Ugovor: specifikacije, bibliografija i sledljivost."""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "docs" / "specs"
BIB = ROOT / "docs" / "bibliography.md"
TRACE = ROOT / "docs" / "traceability.md"
CLASSES = {"established", "strongly_supported", "incomplete", "hypothesis", "speculative"}
MECHS = ["hh", "synapses", "stdp", "da_stdp", "homeostasis", "cpg"]
_TEST_TOK = re.compile(r"`(tests/[^`:]+\.py)::(\w+)`")


def _read(p):
    return p.read_text(encoding="utf-8")


def _bib_keys():
    return set(re.findall(r"^- \[([A-Za-z0-9_]+)\]", _read(BIB), flags=re.M))


def _section(text, num):
    m = re.search(rf"^## {num}\..*?$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    assert m, f"nema sekcije {num}"
    return m.group(1)


def _rows(block):
    out = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        out.append(cells)
    return out


def _num(s):
    return float(s.replace("−", "-").replace(" ", "").replace("`", ""))


def _assert_tokens_exist(text, where):
    toks = _TEST_TOK.findall(text)
    for path, name in toks:
        f = ROOT / path
        assert f.is_file(), (where, path)
        assert re.search(rf"^\s*def {name}\(", _read(f), flags=re.M), (where, path, name)
    return toks


@pytest.mark.parametrize("mech", MECHS)
def test_spec_params_have_source_and_class(mech):
    text = _read(SPECS / f"{mech}.md")
    keys = _bib_keys()
    rows = _rows(_section(text, 5))
    assert rows[0][:5] == ["ime", "vrednost", "jedinica", "izvor", "klasa pouzdanosti"], rows[0]
    assert len(rows) > 1
    for r in rows[1:]:
        assert len(r) == 5, r
        assert "ASSUMPTION" in r[3] or any(k in r[3] for k in keys), (mech, r)
        assert r[4] in CLASSES, (mech, r)
    mod = importlib.import_module(f"neural.mechanisms.{mech}")
    m = re.search(r"\*\*(\w+)\*\*", _section(text, 6))
    assert m and m.group(1) == mod.CONFIDENCE, (mech, m and m.group(1), mod.CONFIDENCE)


@pytest.mark.parametrize("mech", MECHS)
def test_spec_table_matches_params_meta(mech):
    mod = importlib.import_module(f"neural.mechanisms.{mech}")
    rows = {r[0].strip("`"): r for r in _rows(_section(_read(SPECS / f"{mech}.md"), 5))[1:]}
    for key, meta in mod.PARAMS_META.items():
        assert key in rows, f"{mech}: {key} iz PARAMS_META nema red u specifikaciji"
        r = rows[key]
        assert r[4] == meta["confidence"], (mech, key, r[4], meta["confidence"])
        v = meta["value"]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert _num(r[1]) == pytest.approx(float(v), rel=1e-9, abs=1e-12), (mech, key, r[1], v)


@pytest.mark.parametrize("mech", MECHS)
def test_params_meta_sources_in_bibliography(mech):
    mod = importlib.import_module(f"neural.mechanisms.{mech}")
    keys = _bib_keys()
    for name, meta in mod.PARAMS_META.items():
        for tok in (t.strip() for t in meta["source"].split(";")):
            assert tok.startswith("ASSUMPTION") or tok in keys, (mech, name, tok)


def test_bibliography_doi_fields():
    entries = re.findall(r"^- \[[^\]]+\].*$", _read(BIB), flags=re.M)
    assert len(entries) >= 15
    for e in entries:
        assert not re.search(r"(DOI|ISBN):\s*TBD", e), e
        assert re.search(r"DOI: 10\.\d{4,9}/\S+|DOI nije proveren|nema DOI", e), e
    for k in ["HH1952", "BiPoo1998", "Izhikevich2007", "Turrigiano1998", "JahrStevens1990", "Matsuoka1985"]:
        assert k in _bib_keys(), k


def test_threshold_table_has_real_tests():
    text = _read(SPECS / "README.md")
    assert "nije pojedinačno potvrđeno" not in text
    rows = _rows(text)
    col = rows[0].index("ugovorni test")
    body = rows[1:]
    for r in body:
        assert _TEST_TOK.search(r[col]), r
    _assert_tokens_exist(text, "specs/README.md")
    for mech in MECHS:
        bullets = [l for l in _section(_read(SPECS / f"{mech}.md"), 10).splitlines() if l.strip().startswith("- ")]
        assert sum(1 for r in body if r[0] == mech) == len(bullets), mech
    assert "bez d promenljive" not in _read(SPECS / "da_stdp.md")


def test_traceability_rows_point_to_existing_tests():
    text = _read(TRACE)
    _assert_tokens_exist(text, "traceability.md")
    n = 0
    for block in re.split(r"^## ", text, flags=re.M)[1:]:
        rows = _rows(block)
        if not rows or "test" not in rows[0]:
            continue
        col = rows[0].index("test")
        for r in rows[1:]:
            n += 1
            assert len(r) == len(rows[0]), r
            assert _TEST_TOK.search(r[col]), f"red bez `tests/...py::test_...`: {r}"
    assert n >= 10
    assert "DA_t+DA_p" not in text
