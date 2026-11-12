"""Roll φ: validacija konfiguracije, neutralnost pri roll=0 i bit-po-bit referenca faze 1/2."""
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from neural.environment.channel_world import PARAMS_META, ChannelWorld, WorldConfig

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import phase12_digest as pd  # noqa: E402

REF = json.loads((pathlib.Path(__file__).parent / "data" / "phase12_digests.json").read_text())


@pytest.mark.parametrize("dims", [2, 4])
def test_roll_requires_dims3(dims):
    with pytest.raises(ValueError):
        WorldConfig(dims=dims, roll_gain=0.001)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("dims", [2, 3])
def test_roll_gain_must_be_finite(dims, bad):
    with pytest.raises(ValueError):
        WorldConfig(dims=dims, roll_gain=bad)


def test_roll_meta_and_no_ctor_arg():
    assert WorldConfig().roll_gain == 0.0
    assert PARAMS_META["roll_gain"]["unit"] == "rad/ms"
    with pytest.raises(TypeError):
        ChannelWorld(WorldConfig(dims=3), 1, roll_gain=0.1)  # type: ignore[call-arg]


def test_public_names_unchanged():
    w = ChannelWorld(WorldConfig(dims=3), 1)
    pub = {n for n in dir(w) if not n.startswith("_")}
    assert pub == {"n_actuators", "n_sensors", "reset", "step", "make_probe", "probe"}


@pytest.mark.parametrize("dims", [2, 3])
def test_phi_zero_without_roll(dims):
    w = ChannelWorld(WorldConfig(dims=dims), 4)
    w.reset()
    p = w.make_probe()
    rng = np.random.default_rng(0)
    for _ in range(100):
        w.step(rng.uniform(0, 1, 4))
        s = p.snapshot()
        assert s["phi"] == 0.0 and s["n"].shape == (dims - 1,) and s["n"].dtype == np.float64


def test_roll_rotates_n_and_resets():
    w = ChannelWorld(WorldConfig(dims=3, roll_gain=0.001), 4)
    w.reset()
    p = w.make_probe()
    n0 = p.snapshot()["n"]
    for _ in range(2000):
        w.step(np.array([1.0, 1.0, 0.0, 0.0]))
    s = p.snapshot()
    assert s["phi"] > 0
    c, sn = math.cos(s["phi"]), math.sin(s["phi"])
    np.testing.assert_allclose(s["n"], [c * n0[0] - sn * n0[1], sn * n0[0] + c * n0[1]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(s["n"]), 1.0, atol=1e-12)
    w.reset()
    assert p.snapshot()["phi"] == 0.0


def test_symmetric_commands_leave_trajectory_unchanged():
    def run(g):
        w = ChannelWorld(WorldConfig(dims=3, roll_gain=g), 9)
        w.reset()
        xs = []
        for _ in range(500):
            w.step(np.full(4, 0.5))
            xs.append(w.probe().x.copy())
        return np.array(xs), w.make_probe().snapshot()["phi"]

    (x0, p0), (x1, p1) = run(0.0), run(0.001)
    np.testing.assert_array_equal(x0, x1)
    assert p0 == p1 == 0.0


@pytest.mark.parametrize("case", list(pd.CASES))
def test_phase12_bitwise_digest(case):
    assert pd.digest(case) == REF[case]
