"""Stvarni svet i telo (bez lažnih klasa): kratak run, dims 2 i 3."""
import json

import pytest

from experiments.dimension_escape.__main__ import main


@pytest.mark.parametrize("dims", [2, 3])
def test_real_world_short_run(tmp_path, dims):
    out = tmp_path / f"real{dims}"
    rc = main(["--seeds", "1", "--dims", str(dims), "--condition", "all", "--duration-ms", "200",
               "--workers", "1", "--out", str(out)])
    assert rc == 0
    rep = json.load(open(out / "report.json", encoding="utf-8"))
    assert rep["dims"] == dims
    assert rep["config"]["world"]["dims"] == dims
