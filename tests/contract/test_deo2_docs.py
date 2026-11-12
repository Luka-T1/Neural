"""Ugovor dokumentacije Dela 2: poređenje sa bebama, plan, VALIDATION („Deo 2“), README i sledljivost."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
VAL = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
BIB = (DOCS / "bibliography.md").read_text(encoding="utf-8")
INFANT = (DOCS / "infant_comparison.md").read_text(encoding="utf-8")
PLAN = (DOCS / "plan_deo2.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")

BIB_KEYS = set(re.findall(r"^- \[([^\]]+)\]", BIB, flags=re.M))
DEO2_KEYS = ["ThelenFisher1982", "Forssberg1985", "Dominici2011", "RoveeRovee1969", "ThelenSmith1994", "Azevedo2009"]


def _h2_section(text: str, title: str) -> str:
    m = re.search(rf"(?ms)^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", text)
    assert m, f"nema sekcije '## {title}'"
    return m.group(1)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table(section: str) -> list[list[str]]:
    lines = section.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
    assert idx, "nema tabele"
    assert idx == list(range(idx[0], idx[0] + len(idx))), "tabela nije neprekinuta"
    return [_cells(lines[i]) for i in idx]


# ------------------------------------------------------------ docs/infant_comparison.md

def test_infant_sections_present():
    heads = re.findall(r"(?m)^## (.+)$", INFANT)
    for word in ("Svrha", "Organizam", "Bebe", "Tabela analogija", "Iskreno ograničenje"):
        assert any(word in h for h in heads), word


def test_infant_status_and_no_doi_or_verified():
    assert "speculative" in INFANT
    assert "validirano = NE" in INFANT
    assert "verified" not in INFANT.lower()
    assert "DOI" not in INFANT
    assert not re.search(r"10\.\d{4,9}/", INFANT)


def test_infant_cites_only_existing_keys_and_all_deo2_keys():
    cited = set(re.findall(r"\[([A-Za-z]+\d{4})\]", INFANT))
    assert cited, "nema citata"
    assert cited <= BIB_KEYS, cited - BIB_KEYS
    for key in DEO2_KEYS:
        assert key in BIB_KEYS and f"[{key}]" in INFANT, key


def test_infant_organism_numbers_match_code():
    assert "205" in INFANT
    assert "9×5" in INFANT and "80 ekscitatornih" in INFANT and "20 inhibitornih" in INFANT
    assert "4×10" in INFANT and "4×5" in INFANT
    audit = (ROOT / "docs" / "specs" / "organism_audit.md").read_text(encoding="utf-8")
    assert "−0.918" in audit or "-0.918" in audit  # brojke CPG-a su iz revizije


def test_infant_table_and_honest_limitation():
    rows = _table(_h2_section(INFANT, "Tabela analogija"))
    assert rows[0] == ["aspekt", "beba", "organizam", "snaga analogije"]
    aspects = " ".join(r[0] for r in rows[2:])
    for word in ("ritam", "ko-kontrakcija", "učenje", "razvoj", "razmera"):
        assert word in aspects, word
    lim = _h2_section(INFANT, "Iskreno ograničenje")
    assert "NIJE validiran" in lim
    assert "anatomije" in lim.lower() or "anatomij" in lim.lower()
    assert "razvojne plastičnosti" in lim


# ------------------------------------------------------------ docs/plan_deo2.md

def test_plan_title_and_removed_draft_sentence():
    assert PLAN.splitlines()[0] == "# Deo 2: stanje implementacije"
    assert "Nacrt, ništa od ovoga nije implementirano" not in PLAN


def test_plan_each_numbered_section_has_status():
    parts = re.split(r"(?m)^(?=## \d\.)", PLAN)[1:]
    assert len(parts) == 6
    for part in parts:
        assert re.search(r"(?m)^Status: ", part), part.splitlines()[0]


# ------------------------------------------------------------ VALIDATION.md („Deo 2“)

def test_validation_deo2_position_and_single_table():
    i_f4 = VAL.index("## Faza 4")
    i_d2 = VAL.index("## Deo 2")
    i_pr = VAL.index("## Provere pokrenute")
    assert i_f4 < i_d2 < i_pr
    rows = _table(_h2_section(VAL, "Deo 2"))
    assert rows[0] == ["stavka", "implementirano", "testirano", "validirano"]
    body = rows[2:]
    assert len(body) == 5
    names = [r[0] for r in body]
    for needle in ("Motorni šum sa dt (noise_scale)", "Snimanje (recording)", "Vremenska linija (timeline)",
                   "Poređenje sa bebama", "Rerun eksperimenata posle popravki faze 4"):
        assert any(n.startswith(needle) for n in names), needle
    for r in body:
        assert len(r) == 4, r
        assert r[3].startswith("NE (") and r[3].endswith(")"), r


def test_validation_deo2_paths_for_implemented_items():
    sec = _h2_section(VAL, "Deo 2")
    assert "neural/body/motor.py" in sec
    assert "experiments/dimension_escape/recording.py" in sec
    assert "experiments/dimension_escape/timeline.py" in sec
    assert "docs/infant_comparison.md" in sec


# ------------------------------------------------------------ README.md

def _readme_section(title: str) -> str:
    m = re.search(rf"(?ms)^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)", README)
    assert m, f"nema sekcije '## {title}' u README.md"
    return m.group(1)


def test_readme_required_headings_in_order():
    heads = ["## Instalacija", "## Pokretanje", "## Struktura", "## Šta je validirano, a šta nije",
             "## Poznata ograničenja"]
    pos = [README.index(h) for h in heads]
    assert pos == sorted(pos)


def test_readme_install_and_run_commands():
    assert "pip install -e ." in _readme_section("Instalacija")
    run = _readme_section("Pokretanje")
    assert "neural run <preset>" in run and "--preset" in run
    for preset in ("quick", "phase1", "dims3", "roll", "phase4"):
        assert preset in run, preset
    assert "izlaz 2" in run
    assert "python -m experiments.dimension_escape" in run


def test_readme_structure_lists_packages():
    st = _readme_section("Struktura")
    for pkg in ("neural/core", "mechanisms", "organism", "body", "environment", "cli",
                "experiments/", "docs/", "tests/"):
        assert pkg in st, pkg


def test_readme_validation_and_limits():
    val = _readme_section("Šta je validirano, a šta nije")
    assert "random_motor" in val and "2D" in val
    for item in ("DA učenje", "dims3", "roll", "escape_sign", "bebama"):
        assert item in val, item
    lim = _readme_section("Poznata ograničenja")
    for item in ("motorni šum", "mali organizam", "needs-verify", "rerun"):
        assert item.lower() in lim.lower(), item


def test_readme_backtick_paths_exist():
    missing = []
    for p in re.findall(r"`((?:neural|tests|experiments|docs|src|configs)/[^`\s]*)`", README):
        if any(ch in p for ch in "*<>"):
            continue
        if not (ROOT / p.rstrip("/")).exists():
            missing.append(p)
    assert not missing, missing
