"""Ugovor: determinizam sveta za isti seed i isti niz akcija."""
import os

import numpy as np
import pytest

from neural.core.seed import derive_seed
from neural.environment.channel_world import ChannelWorld, WorldConfig


PHASE3 = pytest.mark.skipif(os.environ.get('NEURAL_REQUIRE_PHASE3') != '1', reason='faza 3')
CASES = [(2, 0.0), (3, 0.0), pytest.param(3, 0.05, marks=PHASE3)]


def _run(dims, master, n=200, actions_seed=7, roll_gain=0.0):
    cfg = WorldConfig(dims=dims) if roll_gain == 0.0 else WorldConfig(dims=dims, roll_gain=roll_gain)
    w = ChannelWorld(cfg, seed=derive_seed(master, "world"))
    obs = [w.reset().as_array()]
    probe = w.make_probe()
    snaps = [probe.snapshot()]
    arng = np.random.default_rng(actions_seed)
    for _ in range(n):
        obs.append(w.step(arng.uniform(0, 1, 4)).as_array())
        snaps.append(probe.snapshot())
    return np.array(obs), snaps


@pytest.mark.parametrize(("dims", "roll_gain"), CASES)
def test_same_seed_bit_identical(dims, roll_gain):
    o1, s1 = _run(dims, 5, roll_gain=roll_gain)
    o2, s2 = _run(dims, 5, roll_gain=roll_gain)
    assert np.array_equal(o1, o2)
    for a, b in zip(s1, s2):
        assert a.keys() == b.keys()
        for k in a:
            assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


@pytest.mark.parametrize(("dims", "roll_gain"), CASES)
def test_other_seed_changes_noise_stream(dims, roll_gain):
    o1, _ = _run(dims, 5, roll_gain=roll_gain)
    o2, _ = _run(dims, 6, roll_gain=roll_gain)
    assert not np.array_equal(o1, o2)
