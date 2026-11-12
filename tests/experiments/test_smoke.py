import json
import os

import pytest

from experiments.dimension_escape.__main__ import main


@pytest.mark.parametrize("dims", [2, 3])
def test_smoke_all_conditions(fake_world, tmp_path, dims):
    out = tmp_path / f"d{dims}"
    rc = main(["--seeds", "2", "--dims", str(dims), "--condition", "all", "--duration-ms", "200",
               "--workers", "1", "--out", str(out)])
    assert rc == 0
    rep = json.load(open(out / "report.json", encoding="utf-8"))
    assert rep["dims"] == dims and rep["seeds"] == [0, 1]
    assert set(rep["conditions"]) == {"full", "random_motor", "frozen", "no_da"}
    assert len(rep["comparisons"]) == 3
    for c in rep["comparisons"]:
        assert 0 <= c["p_holm"] <= 1 and 0 <= c["fisher_p"] <= 1 and 0 <= c["mwu_p"] <= 1
    assert "da_leakage_audit" in rep and "config" in rep
    assert (out / "report.md").exists() and (out / "run.log").exists()
    assert os.path.exists(out / "traces" / "full_seed0.npz")


def test_single_condition_has_no_comparisons(fake_world, tmp_path):
    out = tmp_path / "one"
    main(["--seeds", "2", "--condition", "full", "--duration-ms", "100", "--workers", "1", "--out", str(out)])
    assert json.load(open(out / "report.json", encoding="utf-8"))["comparisons"] == []
