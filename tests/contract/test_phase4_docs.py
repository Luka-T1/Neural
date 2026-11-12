"""Ugovor dokumentacije faze 4 (napisan po opisu zadatka; zvanični tekst ugovora nije bio u repou)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VAL = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
BIB = (ROOT / "docs" / "bibliography.md").read_text(encoding="utf-8")
PLAN = ROOT / "docs" / "plan_deo2.md"


def _section(title_prefix: str) -> str:
    m = re.search(rf"(?ms)^## {re.escape(title_prefix)}[^\n]*\n(.*?)(?=^## |\Z)", VAL)
    assert m, title_prefix
    return m.group(1)


def _cells(line: str) -> list[str]:
    line = line.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", line)]


def _table_rows(sec: str) -> list[list[str]]:
    lines = [ln for ln in sec.splitlines() if ln.lstrip().startswith("|")]
    # tačno jedna tabela: svi redovi su uzastopni
    idx = [i for i, ln in enumerate(sec.splitlines()) if ln.lstrip().startswith("|")]
    assert idx == list(range(idx[0], idx[0] + len(idx))), "više od jedne tabele"
    return [_cells(ln) for ln in lines]


def _roll_in_reports() -> bool:
    for f in (ROOT / "experiments" / "dimension_escape" / "results").glob("**/report.json"):
        try:
            if "roll" in json.loads(f.read_text(encoding="utf-8")):
                return True
        except (OSError, ValueError):
            continue
    return False


def _check(sec_name: str):
    rows = _table_rows(_section(sec_name))
    assert rows[0] == ["stavka", "implementirano", "testirano", "validirano"]
    body = rows[2:]
    assert body
    for r in body:
        assert len(r) == 4, r
        assert r[3].startswith("NE"), r
    return body


def test_phase3_table():
    _check("Faza 3")


def test_phase4_table_and_rows():
    body = _check("Faza 4")
    text = "\n".join(r[0] for r in body)
    for key in ("escape_sign", "Revizija organizma", "RNG", "touch[2]", "CLI preset"):
        assert key in text, key
    assert "ostaju validirano=NE" in _section("Faza 4")


def test_roll_not_validated_without_roll_report():
    if not _roll_in_reports():
        assert "NE" in _check("Faza 3")[0][3]


def test_verified_entries_have_pages_or_article_number():
    for block in re.split(r"(?m)^(?=- \[)", BIB):
        if block.startswith("- [") and "doi_status: verified" in block:
            head = block.splitlines()[0]
            assert re.search(r"\d+(\(\d+\))?:\s*[\w–-]+", head), head


def test_bilbao_needs_verify_with_reason():
    block = next(b for b in re.split(r"(?m)^(?=- \[)", BIB) if b.startswith("- [Bilbao2018]"))
    assert "doi_status: needs-verify" in block and "Razlog:" in block


def test_plan_deo2():
    txt = PLAN.read_text(encoding="utf-8")
    assert "verified" not in txt and "TBD" not in txt
    assert "Poklapanje 1-1 sa ljudskim mozgom nije izvodljivo u organizmu ove veličine." in txt
    assert "Poređenje je samo kvalitativno i zasnovano na izvorima; status: validirano = NE dok ne postoje podaci." in txt
    assert "doi_status: needs-verify" in txt
    assert not re.search(r"10\.\d{4,9}/", txt)
    assert len(re.findall(r"(?m)^## \d\.", txt)) == 6
