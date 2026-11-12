"""Poreklo rezultata u VALIDATION.md: run na commit-ovanom kodu, brojevi iz report.json."""
import json
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "dimension_escape" / "results"
RUNS = {"dims2_seeds10": (10, 2, 60000.0), "dims3_short": (3, 3, 10000.0)}
OLD = "28d9a7d"


def _val():
    return (ROOT / "VALIDATION.md").read_text(encoding="utf-8")


def _rep(name):
    return json.loads((RES / name / "report.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(RUNS))
def test_report_hash_is_committed(name):
    h = _rep(name)["config"]["git_hash"]
    assert re.fullmatch(r"[0-9a-f]{40}", h), h
    assert not h.startswith(OLD), "run nije ponovljen"
    try:
        r = subprocess.run(["git", "cat-file", "-e", h + "^{commit}"], cwd=ROOT, capture_output=True)
    except FileNotFoundError:
        pytest.skip("git nije dostupan")
    assert r.returncode == 0, f"{h} nije commit u repou"


@pytest.mark.parametrize("name", sorted(RUNS))
def test_run_args_and_single_log(name):
    rep = _rep(name)
    seeds, dims, dur = RUNS[name]
    a = rep["config"]["args"]
    assert (a["seeds"], a["dims"], a["condition"], a["duration_ms"], a["dt_ms"]) == (seeds, dims, "all", dur, 0.1)
    log = (RES / name / "run.log").read_text(encoding="utf-8")
    starts = [ln for ln in log.splitlines() if " start: " in ln]
    assert len(starts) == 1, starts
    assert starts[0].rstrip().endswith("git=" + rep["config"]["git_hash"])
    assert "gotovo:" in log


def test_validation_cites_hash_without_dirty_note():
    v = _val()
    assert "necommitovan" not in v.lower()
    heads = [ln for ln in v.splitlines() if ln.startswith("### Rezultati")]
    assert heads, "nema naslova '### Rezultati'"
    assert _rep("dims2_seeds10")["config"]["git_hash"][:7] in heads[0]
    for name in RUNS:
        assert _rep(name)["config"]["git_hash"][:7] in v, name


def test_counts_match_reports():
    v = _val()
    for c, d in _rep("dims2_seeds10")["conditions"].items():
        m = re.search(rf"^\|\s*{c}\s*\|\s*(\d+)/(\d+)\s*\|", v, re.M)
        assert m, f"dims2 red za {c}"
        assert (int(m.group(1)), int(m.group(2))) == (d["escaped"], d["n"]), c
    for c, d in _rep("dims3_short")["conditions"].items():
        assert f"{c} {d['escaped']}/{d['n']}" in v, f"dims3 {c}"


def test_status_rows_follow_holm():
    comps = {c["b"]: c for c in _rep("dims2_seeds10")["comparisons"]}
    lines = _val().splitlines()
    da_row = next(ln for ln in lines if "Doprinos DA-STDP" in ln)
    if all(comps[b]["p_holm"] >= 0.05 for b in ("frozen", "no_da")):
        assert "NE — nije validirano" in da_row
    cpg_row = next(ln for ln in lines if ln.startswith("|") and "Bekstvo: CPG" in ln)
    if comps["random_motor"]["p_holm"] >= 0.05:
        assert "**DA**" not in cpg_row.split("|")[-2]
