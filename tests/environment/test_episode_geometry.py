"""Epizodni reset: isti n kroz epizode (geometry_seed), nov x0/izvor; k počinje od 0."""

import numpy as np

from neural.core.seed import derive_seed
from neural.environment import ChannelWorld, WorldConfig


def _geom(w):
    return float(w._x0), float(w._x_src), w._n.copy()


def test_geometry_seed_keeps_n_across_episodes():
    cfg = WorldConfig(dims=3)
    seed = 7
    w = ChannelWorld(cfg, derive_seed(seed, "world.ep0"), geometry_seed=derive_seed(seed, "world"))
    w.reset()
    x0a, xsa, na = _geom(w)
    w.reset(derive_seed(seed, "world.ep1"))
    x0b, xsb, nb = _geom(w)
    assert np.array_equal(na, nb)
    assert (x0a, xsa) != (x0b, xsb)
    assert abs(np.linalg.norm(na) - 1.0) < 1e-12


def test_without_geometry_seed_n_follows_episode_seed():
    cfg = WorldConfig(dims=3)
    w = ChannelWorld(cfg, 1)
    w.reset()
    n1 = w._n.copy()
    w.reset(2)
    assert not np.array_equal(n1, w._n)


def test_dims2_n_is_unit_with_geometry_seed():
    w = ChannelWorld(WorldConfig(dims=2), 3, geometry_seed=5)
    w.reset()
    assert np.array_equal(w._n, np.array([1.0]))
