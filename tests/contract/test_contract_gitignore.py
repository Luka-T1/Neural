"""Ugovor: .gitignore ne sme da ignoriše praćene rezultate runova."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = "experiments/dimension_escape/results"
TRACKED = [
    f"{RESULTS}/{run}/{name}"
    for run in ("dims2_seeds10", "dims3_short", "deo2_dims2_seeds10", "deo2_dims3_seeds10", "deo2_roll_seeds10",
                "deo2_phase4_seeds10", "deo2_phase4_learn_seeds10")
    for name in ("report.json", "report.md", "run.log")
]


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_gitignore_has_no_conflict_markers_or_duplicates():
    lines = [l for l in (ROOT / ".gitignore").read_text().splitlines() if l.strip()]
    assert not any(l.startswith(("<<<<<<<", "=======", ">>>>>>>")) for l in lines)
    assert len(lines) == len(set(lines))
    assert lines[:3] == ["__pycache__/", "*.pyc", f"{RESULTS}/**/traces/"]


def test_tracked_results_not_ignored():
    for path in TRACKED:
        r = _git("check-ignore", "--no-index", "-v", path)
        assert r.returncode == 1 and r.stdout == "", f"{path} je ignorisan: {r.stdout}"


def test_traces_ignored():
    r = _git("check-ignore", "--no-index", f"{RESULTS}/dims2_seeds10/traces/x.npz")
    assert r.returncode == 0


def test_six_result_files_tracked():
    r = _git("ls-files", RESULTS)
    if r.returncode != 0:
        return
    assert sorted(r.stdout.split()) == sorted(TRACKED)
