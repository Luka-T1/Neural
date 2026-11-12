"""docs/traceability.md i docs/specs smeju da navode samo putanje koje postoje."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _paths(text):
    for m in re.finditer(r"`((?:neural|tests|experiments|docs|configs)/[^`\s:]*)(?:::[^`]*)?`", text):
        yield m.group(1)


def test_documented_paths_exist():
    files = [ROOT / "docs" / "traceability.md", ROOT / "VALIDATION.md", *(ROOT / "docs" / "specs").glob("*.md")]
    missing = [f"{f.name}: {p}" for f in files for p in _paths(f.read_text(encoding="utf-8"))
               if not (ROOT / p).exists()]
    assert not missing, missing
