"""Ugovor faze 4: harness, report i CLI za jednostrani zid (escape_sign, side_wall_mm)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.dimension_escape import harness
from experiments.dimension_escape.__main__ import main
from experiments.dimension_escape.report import build_report, render_markdown

KW = dict(dims=2, duration_ms=300.0, dt_ms=0.1, trace_every_ms=10.0, out_dir=None)


def _run(cond="frozen", seed=0, **extra):
    return harness.run_one(cond, seed, KW["dims"], KW["duration_ms"], KW["dt_ms"],
                           KW["trace_every_ms"], KW["out_dir"], **extra)


def test_world_config_kwargs_defaults():
    assert harness.world_config_kwargs(2) == {"dims": 2}
    assert harness.world_config_kwargs(3, roll_gain=0.01) == {"dims": 3, "roll_gain": 0.01}
    assert harness.world_config_kwargs(2, escape_sign=-1) == {"dims": 2, "escape_sign": -1}
    assert harness.world_config_kwargs(2, escape_sign=1, side_wall_mm=0.3) == {
        "dims": 2, "escape_sign": 1, "side_wall_mm": 0.3}


@pytest.mark.parametrize("bad", [True, False, 0.0, 1.0, "1", None])
def test_world_config_kwargs_rejects_non_int_sign(bad):
    with pytest.raises(ValueError):
        harness.world_config_kwargs(2, escape_sign=bad)


def test_world_config_kwargs_side_wall_requires_sign():
    with pytest.raises(ValueError):
        harness.world_config_kwargs(2, escape_sign=0, side_wall_mm=0.5)


def test_run_one_default_has_no_phase4_keys(tmp_path):
    r = harness.run_one("frozen", 0, 2, 300.0, 0.1, 10.0, str(tmp_path))
    assert not {"escape_sign", "side_wall_mm", "x_perp_n"} & set(r)
    with np.load(tmp_path / "traces" / "frozen_seed0.npz") as z:
        assert "x_perp_n" not in z.files


def test_run_one_sign_zero_bit_identical_to_default():
    a, b = _run(), _run(escape_sign=0)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["cmd"], b["cmd"])


@pytest.mark.parametrize("s", [1, -1])
def test_run_one_with_sign(s, tmp_path):
    r = harness.run_one("frozen", 0, 2, 300.0, 0.1, 10.0, str(tmp_path), escape_sign=s)
    assert r["escape_sign"] == s and isinstance(r["escape_sign"], int)
    assert r["side_wall_mm"] == 0.5 and isinstance(r["side_wall_mm"], float)
    assert isinstance(r["x_perp_n"], np.ndarray) and len(r["x_perp_n"]) == len(r["t"])
    with np.load(tmp_path / "traces" / "frozen_seed0.npz") as z:
        assert np.array_equal(z["x_perp_n"], r["x_perp_n"])


def test_run_one_explicit_side_wall():
    assert _run(escape_sign=1, side_wall_mm=0.25)["side_wall_mm"] == 0.25
    with pytest.raises(ValueError):
        _run(side_wall_mm=0.25)


def test_run_task_shapes():
    base = ("frozen", 0, 2, 100.0, 0.1, 10.0, None, None)
    assert "escape_sign" not in harness.run_task(base)
    assert "escape_sign" not in harness.run_task(base + (0.0,))
    assert harness.run_task(base + (0.0, -1))["escape_sign"] == -1
    assert harness.run_task(base + (0.0, 1, 0.3))["side_wall_mm"] == 0.3


def _fake(cond, seed, sign, xpn, escaped, wall=0.5):
    return {"cond": cond, "seed": seed, "escaped": escaped, "t_escape_ms": 5.0 if escaped else None,
            "t": np.array([0.0, 10.0]), "x": np.zeros((2, 2)), "da": np.zeros(2),
            "w_cross": np.zeros(2), "q": np.zeros((2, 2)), "cmd": np.zeros((2, 4)),
            "secondary": {}, "escape_sign": sign, "side_wall_mm": wall,
            "x_perp_n": np.array(xpn, float)}


def test_report_escape_side_section():
    res = {"full": [_fake("full", 1, 1, [0.0, 2.0], True), _fake("full", 0, 1, [0.0, 0.1], False)]}
    rep = build_report(res, 2, 100.0, 0.1, [0, 1], {})
    es = rep["escape_side"]
    assert es["escape_sign"] == 1 and es["side_wall_mm"] == 0.5
    c = es["conditions"]["full"]
    assert (c["n"], c["escaped"], c["escaped_toward_sign"], c["escaped_against_sign"]) == (2, 1, 1, 0)
    assert [p["seed"] for p in c["per_seed"]] == [0, 1]
    assert [p["escape_side"] for p in c["per_seed"]] == [0, 1]
    md = render_markdown(rep)
    assert "Strana bekstva" in md and "| full |" in md
    json.dumps(rep)


def test_report_without_sign_has_no_section():
    r = _fake("full", 0, 1, [0.0], False)
    for k in ("escape_sign", "side_wall_mm", "x_perp_n"):
        del r[k]
    rep = build_report({"full": [r]}, 2, 100.0, 0.1, [0], {})
    assert "escape_side" not in rep and "Strana bekstva" not in render_markdown(rep)


def test_report_mismatched_signs_raise():
    res = {"full": [_fake("full", 0, 1, [0.0], False), _fake("full", 1, -1, [0.0], False)]}
    with pytest.raises(ValueError):
        build_report(res, 2, 100.0, 0.1, [0, 1], {})


@pytest.mark.parametrize("argv", [["--escape-sign", "2"], ["--escape-sign", "0.5"], ["--escape-sign", "true"],
                                  ["--side-wall-mm", "0.5"], ["--escape-sign", "1", "--side-wall-mm", "0"],
                                  ["--escape-sign", "1", "--side-wall-mm", "-1"],
                                  ["--escape-sign", "1", "--side-wall-mm", "inf"]])
def test_main_rejects_bad_args(argv, tmp_path):
    with pytest.raises(SystemExit) as e:
        main(["--seeds", "1", "--out", str(tmp_path)] + argv)
    assert e.value.code == 2


def test_main_smoke_report(tmp_path):
    rc = main(["--seeds", "1", "--condition", "frozen", "--duration-ms", "200", "--workers", "1",
               "--escape-sign", "-1", "--side-wall-mm", "0.4", "--out", str(tmp_path)])
    assert rc == 0
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert rep["config"]["args"]["escape_sign"] == -1
    assert rep["config"]["args"]["side_wall_mm"] == 0.4
    assert rep["escape_side"]["escape_sign"] == -1


def test_main_default_omits_phase4_args(tmp_path):
    main(["--seeds", "1", "--condition", "frozen", "--duration-ms", "100", "--workers", "1",
          "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "escape_sign" not in rep["config"]["args"] and "side_wall_mm" not in rep["config"]["args"]
    assert "escape_side" not in rep
