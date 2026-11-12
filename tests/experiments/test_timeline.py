import json
import sys

import numpy as np

from experiments.dimension_escape import visualize
from experiments.dimension_escape.__main__ import main
from experiments.dimension_escape.timeline import build_timeline, timeline_from_dir

BASE = ["--seeds", "2", "--duration-ms", "250", "--workers", "1", "--trace-every-ms", "1.0"]


def test_timeline_matches_report(tmp_path):
    main(BASE + ["--record", "--out", str(tmp_path)])
    tl = json.load(open(tmp_path / "timeline.json", encoding="utf-8"))
    rep = json.load(open(tmp_path / "report.json", encoding="utf-8"))
    assert [(r["cond"], r["seed"]) for r in tl["runs"]] == [
        (c, s) for c in ("full", "random_motor", "frozen", "no_da") for s in (0, 1)]
    for r in tl["runs"]:
        ps = rep["conditions"][r["cond"]]["per_seed"][r["seed"]]
        assert r["t_escape_ms"] == ps["t_escape_ms"]
        z = np.load(tmp_path / "traces" / f"{r['cond']}_seed{r['seed']}.npz")
        assert np.allclose(r["x_perp"], np.linalg.norm(z["x"][:, 1:], axis=1))
        assert np.allclose(r["q_cpg"], z["cmd"][:, 0] - z["cmd"][:, 1])
        da = z["da"]
        ev = [float(z["t"][k]) for k in range(len(da)) if da[k] > 0 and (k == 0 or da[k - 1] <= 0)]
        assert r["da_events_ms"] == ev


def test_build_timeline_synthetic():
    tr = {"t": np.arange(5.0), "x": np.array([[0, 3, 4]] * 5, float), "escaped": np.array(False),
          "t_escape_ms": -1.0, "da": np.array([0, 1, 1, 0, 2.0]),
          "cmd": np.array([[1, 0.5, 0, 0]] * 5, float)}
    tl = build_timeline(tr, "full", 3)
    assert tl["t_escape_ms"] is None and tl["x_perp"] == [5.0] * 5
    assert tl["da_events_ms"] == [1.0, 4.0] and tl["q_cpg"] == [0.5] * 5
    tr["escaped"], tr["t_escape_ms"] = np.array(True), 3.0
    assert build_timeline(tr, "full", 3)["t_escape_ms"] == 3.0


def test_timeline_from_missing_dir(tmp_path):
    assert timeline_from_dir(str(tmp_path)) == {"runs": []}


def test_visualize_without_matplotlib(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    p = tmp_path / "t.png"
    assert visualize.plot_timeline({"runs": []}, str(p)) is False
    assert not p.exists() and "matplotlib nije instaliran" in capsys.readouterr().out
