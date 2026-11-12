"""Ugovor faze 3 za harness: roll_gain (rad/ms, podrazumevano 0.0, samo dims=3).

Pri roll_gain=0 izlaz je bit-po-bit isti kao ranije (bez dodatnih ključeva). Pri roll_gain>0
(gate NEURAL_REQUIRE_PHASE3=1) organizam vidi iste ulaze, pa su x, cmd i da identični, a φ se
čita isključivo iz probe().snapshot().
"""
import os

import numpy as np
import pytest

from experiments.dimension_escape import harness
from experiments.dimension_escape.__main__ import main
from experiments.dimension_escape.report import build_report, render_markdown

PHASE3 = pytest.mark.skipif(os.environ.get("NEURAL_REQUIRE_PHASE3") != "1", reason="faza 3")
COND = "full"


def _run(**kw):
    return harness.run_one(COND, 0, 3, 300.0, 0.1, 10.0, None, **kw)


def _same(a, b):
    for k in ("t", "x", "da", "w_cross", "q", "cmd"):
        np.testing.assert_array_equal(a[k], b[k])
    assert a["escaped"] == b["escaped"] and a["t_escape_ms"] == b["t_escape_ms"]


def test_roll_zero_is_default_and_bit_identical():
    r_def, r_zero = _run(), _run(roll_gain=0.0)
    _same(r_def, r_zero)
    assert "phi" not in r_def and "roll_gain" not in r_zero


def test_roll_zero_report_has_no_roll_section():
    r = _run(roll_gain=0.0)
    rep = build_report({COND: [r]}, 3, 300.0, 0.1, [0], {})
    assert "roll" not in rep
    assert "Roll" not in render_markdown(rep)


def test_roll_nonzero_requires_dims3():
    with pytest.raises(ValueError):
        harness.run_one(COND, 0, 2, 50.0, 0.1, 10.0, None, roll_gain=0.05)
    with pytest.raises(SystemExit):
        main(["--dims", "2", "--roll-gain", "0.05", "--seeds", "1", "--out", os.devnull])


def test_run_task_accepts_8_and_9_tuple():
    a = harness.run_task((COND, 0, 3, 50.0, 0.1, 10.0, None, None))
    b = harness.run_task((COND, 0, 3, 50.0, 0.1, 10.0, None, None, 0.0))
    _same(a, b)


@PHASE3
def test_roll_positive_does_not_change_organism_path_and_reports_phi():
    r0, r1 = _run(roll_gain=0.0), _run(roll_gain=0.05)
    for k in ("x", "cmd", "da", "q"):
        np.testing.assert_array_equal(r0[k], r1[k])
    assert r1["roll_gain"] == 0.05
    assert len(r1["phi"]) == len(r1["t"]) and np.all(np.isfinite(r1["phi"]))
    rep = build_report({COND: [r1]}, 3, 300.0, 0.1, [0], {})
    assert COND in rep["roll"] and "Roll" in render_markdown(rep)


@PHASE3
def test_phi_comes_from_probe_snapshot(monkeypatch):
    calls = []
    orig = harness._snapshot

    def snap(world):
        d = orig(world)
        calls.append(d["phi"])
        return d

    monkeypatch.setattr(harness, "_snapshot", snap)
    r = _run(roll_gain=0.05)
    np.testing.assert_array_equal(r["phi"], np.array(calls))
