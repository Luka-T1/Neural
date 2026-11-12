"""Ugovorni integracioni test (odluka lead-a): stvarni ChannelWorld i Body, dims=2, 1 seed, 4 uslova.

Proverava: body.reset() se poziva, `escaped`/`t_escape_ms` dolaze iz probe snapshot-a sveta,
ponovljen run sa istim seedom je determinističan, organizam dobija samo (9,) i vraća (4,).
"""
import numpy as np
import pytest

from experiments.dimension_escape import harness
from neural.organism import Organism

CONDS = ("full", "random_motor", "frozen", "no_da")


@pytest.fixture
def spies(monkeypatch):
    log = {"reset": 0, "in_shapes": set(), "out_shapes": set(), "snaps": []}

    orig_reset = harness.Body.reset

    def reset(self, *a, **k):
        log["reset"] += 1
        return orig_reset(self, *a, **k)

    monkeypatch.setattr(harness.Body, "reset", reset)

    orig_step = Organism.step

    def step(self, s, dt):
        log["in_shapes"].add(np.asarray(s).shape)
        out = orig_step(self, s, dt)
        log["out_shapes"].add(np.asarray(out).shape)
        return out

    monkeypatch.setattr(Organism, "step", step)

    orig_snap = harness._snapshot

    def snap(world):
        d = orig_snap(world)
        log["snaps"].append(d)
        return d

    monkeypatch.setattr(harness, "_snapshot", snap)
    return log


def _run(cond, seed=0):
    return harness.run_one(cond, seed, 2, 300.0, 0.1, 10.0, None)


@pytest.mark.parametrize("cond", CONDS)
def test_run_uses_reset_and_probe_snapshot(spies, cond):
    resets_before = spies["reset"]
    res = _run(cond)
    assert spies["reset"] == resets_before + 1
    assert spies["snaps"], "bekstvo se mora čitati iz probe snapshot-a"
    snap_escaped = any(bool(s["escaped"]) for s in spies["snaps"])
    assert bool(res["escaped"]) == snap_escaped
    if res["escaped"]:
        assert res["t_escape_ms"] == spies["snaps"][-1]["t_escape_ms"]
    else:
        assert res["t_escape_ms"] is None


def test_organism_io_shapes(spies):
    for cond in ("full", "frozen", "no_da"):
        _run(cond)
    assert spies["in_shapes"] == {(9,)}
    assert spies["out_shapes"] == {(4,)}


@pytest.mark.parametrize("cond", CONDS)
def test_same_seed_deterministic(cond):
    a, b = _run(cond, 1), _run(cond, 1)
    assert a["t_escape_ms"] == b["t_escape_ms"]
    assert a["escaped"] == b["escaped"]
    np.testing.assert_array_equal(a["x"], b["x"])
