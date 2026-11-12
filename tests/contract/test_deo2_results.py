"""Ugovor: rezultati Dela 2 (run-ovi a–e) imaju očekivanu šemu report.json i provenijenciju."""
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "dimension_escape" / "results"
RUNS = [
    "deo2_dims2_seeds10",
    "deo2_dims3_seeds10",
    "deo2_roll_seeds10",
    "deo2_phase4_seeds10",
    "deo2_phase4_learn_seeds10",
]
CONDS = {"full", "random_motor", "frozen", "no_da"}


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _report(run):
    return json.loads((RES / run / "report.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("run", RUNS)
def test_run_files_present(run):
    for name in ("report.json", "report.md", "run.log"):
        assert (RES / run / name).is_file(), f"{run}/{name} nedostaje"


@pytest.mark.parametrize("run", RUNS)
def test_seeds_and_conditions(run):
    rep = _report(run)
    assert rep["seeds"] == list(range(10))
    assert set(rep["conditions"]) == CONDS
    for c, d in rep["conditions"].items():
        assert d["n"] == 10, (run, c)
        assert "wilson95" in d and "t_escape_median_ms" in d, (run, c)


@pytest.mark.parametrize("run", RUNS)
def test_three_holm_comparisons(run):
    rep = _report(run)
    assert len(rep["comparisons"]) == 3
    assert all("p_holm" in cmp for cmp in rep["comparisons"])


@pytest.mark.parametrize("run", RUNS)
def test_git_hash_is_ancestor_of_head(run):
    # Rezultati se commit-uju posle run-a, pa HEAD više nije hash iz run-a; dovoljno je da je predak.
    h = _report(run)["config"]["git_hash"]
    assert _git("cat-file", "-e", f"{h}^{{commit}}").returncode == 0
    assert _git("merge-base", "--is-ancestor", h, "HEAD").returncode == 0


@pytest.mark.parametrize("run", RUNS)
def test_optional_keys_only_where_expected(run):
    rep = _report(run)
    assert ("roll" in rep) == (run == "deo2_roll_seeds10")
    assert ("escape_side" in rep) == run.startswith("deo2_phase4")


@pytest.mark.parametrize("run", [r for r in RUNS if r.startswith("deo2_phase4")])
def test_escape_side_counts_per_condition(run):
    es = _report(run)["escape_side"]["conditions"]
    assert set(es) == CONDS
    for c, d in es.items():
        assert "escaped_toward_sign" in d and "escaped_against_sign" in d, c


def test_traces_not_tracked():
    r = _git("ls-files", "experiments/dimension_escape/results")
    assert all("/traces/" not in line for line in r.stdout.split())
