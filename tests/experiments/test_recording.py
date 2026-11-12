"""Snimanje (--record): identičnost izlaza, ključevi npz-a, read-only, greške."""
import json
import os

import numpy as np
import pytest

from experiments.dimension_escape.__main__ import main
from experiments.dimension_escape.harness import run_one

BASE = ["--seeds", "1", "--duration-ms", "250", "--workers", "1", "--trace-every-ms", "1.0", "--condition", "all"]
KEYS = {"t", "spike_t_ms", "spike_idx", "pop_names", "pop_bounds", "da_phasic", "d", "novelty_bar",
        "elig_sens_exc", "elig_exc_cpg", "elig_cross", "premotor", "commands"}


def _scrub(o):
    if isinstance(o, dict):
        return {k: _scrub(v) for k, v in o.items() if k not in ("wall_s", "wall_s_mean", "git_hash")}
    if isinstance(o, list):
        return [_scrub(v) for v in o]
    return o


def test_run_one_identical_with_and_without_record(tmp_path):
    a = run_one("full", 0, 2, 250.0, 0.1, 1.0, str(tmp_path / "a"))
    b = run_one("full", 0, 2, 250.0, 0.1, 1.0, str(tmp_path / "b"), None, 0.0, 0, None, True)
    assert set(b) == set(a) | {"recording_path"}
    for k in a:
        if k in ("wall_s", "secondary"):
            continue
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
    za, zb = np.load(tmp_path / "a/traces/full_seed0.npz"), np.load(tmp_path / "b/traces/full_seed0.npz")
    for k in za.files:
        assert np.array_equal(za[k], zb[k]), k


def test_record_keys_and_shapes(tmp_path):
    r = run_one("full", 0, 2, 250.0, 0.1, 1.0, str(tmp_path), None, 0.0, 0, None, True)
    assert r["recording_path"] == os.path.join(str(tmp_path), "recordings", "full_seed0.npz")
    z = np.load(r["recording_path"], allow_pickle=False)
    assert set(z.files) == KEYS
    S = len(z["t"])
    assert S == len(r["t"]) and z["premotor"].shape == (S, 4) and z["commands"].shape[0] == S
    assert z["pop_bounds"].shape == (5, 2) and list(z["pop_names"]) == ["sens", "exc", "inh", "cpg", "cpg_inh"]
    assert z["spike_t_ms"].shape == z["spike_idx"].shape and z["spike_idx"].dtype == np.int64
    assert np.array_equal(z["da_phasic"], r["da"])
    assert np.array_equal(z["premotor"], r["cmd"])


def test_random_motor_record_has_nan(tmp_path):
    r = run_one("random_motor", 0, 2, 100.0, 0.1, 1.0, str(tmp_path), None, 0.0, 0, None, True)
    z = np.load(r["recording_path"])
    assert z["spike_idx"].size == 0 and np.isnan(z["d"]).all() and np.isnan(z["elig_cross"]).all()


def test_record_requires_out_dir():
    with pytest.raises(ValueError):
        run_one("full", 0, 2, 50.0, 0.1, 1.0, None, None, 0.0, 0, None, True)


def test_main_record_report_identical_and_no_new_keys_without(tmp_path):
    o1, o2 = tmp_path / "plain", tmp_path / "rec"
    main(BASE + ["--out", str(o1)])
    main(BASE + ["--record", "--out", str(o2)])
    assert not (o1 / "recordings").exists() and not (o1 / "timeline.json").exists()
    assert (o2 / "recordings" / "full_seed0.npz").exists() and (o2 / "timeline.json").exists()
    r1 = json.load(open(o1 / "report.json", encoding="utf-8"))
    r2 = json.load(open(o2 / "report.json", encoding="utf-8"))
    assert "recording_dir" not in r1 and "record" not in r1["config"]["args"]
    assert r2.pop("recording_dir") == "recordings"
    r2["config"]["args"].pop("record", None)
    r1["config"]["args"].pop("out", None)
    r2["config"]["args"].pop("out", None)
    assert _scrub(r1) == _scrub(r2)
    for f in sorted(os.listdir(o1 / "traces")):
        za, zb = np.load(o1 / "traces" / f), np.load(o2 / "traces" / f)
        for k in za.files:
            assert np.array_equal(za[k], zb[k]), (f, k)


def test_recording_is_read_only(tmp_path):
    a = run_one("full", 1, 2, 200.0, 0.1, 1.0, str(tmp_path / "x"), None, 0.0, 0, None, True)
    b = run_one("full", 1, 2, 200.0, 0.1, 1.0, str(tmp_path / "y"), None, 0.0, 0, None, True)
    za, zb = np.load(a["recording_path"]), np.load(b["recording_path"])
    assert np.array_equal(za["spike_idx"], zb["spike_idx"]) and np.array_equal(za["spike_t_ms"], zb["spike_t_ms"])
