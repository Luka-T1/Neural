"""Ugovor faze 4: jednostrani zid (escape_sign, side_wall_mm) u ChannelWorld.

Napomena: ovaj fajl je napisao izvršilac po specifikaciji zadatka j.1 jer u worktree-u
nije postojao; lead treba da ga zameni službenim ugovorom ako postoji.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from neural.environment.channel_world import PARAMS_META, ChannelWorld, WorldConfig


def _quad(k: int) -> np.ndarray:
    ph = 2 * np.pi * k / 1000.0
    return np.array([0.5 + 0.5 * np.sin(ph), 0.5 - 0.5 * np.sin(ph), 0.5 + 0.5 * np.cos(ph), 0.5 - 0.5 * np.cos(ph)])


def _rev_quad(k: int) -> np.ndarray:
    # qA = sin, qB = -cos: d_area > 0, x_perp ide ka +n
    ph = 2 * np.pi * k / 1000.0
    return np.array([0.5 + 0.5 * np.sin(ph), 0.5 - 0.5 * np.sin(ph), 0.5 - 0.5 * np.cos(ph), 0.5 + 0.5 * np.cos(ph)])


# --- konfiguracija -----------------------------------------------------------

def test_defaults_and_meta():
    c = WorldConfig()
    assert c.escape_sign == 0 and c.side_wall_mm == 0.5
    assert PARAMS_META["escape_sign"]["unit"] == "1"
    assert PARAMS_META["side_wall_mm"]["unit"] == "mm"
    assert {f.name for f in dataclasses.fields(WorldConfig)} >= {"escape_sign", "side_wall_mm"}


@pytest.mark.parametrize("bad", [0.5, 2, -2, 1.0, True])
def test_escape_sign_only_minus_one_zero_one(bad):
    with pytest.raises(ValueError):
        WorldConfig(escape_sign=bad)


@pytest.mark.parametrize("good", [-1, 0, 1])
def test_escape_sign_accepts_exact_values(good):
    assert WorldConfig(escape_sign=good).escape_sign == good


@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan, math.inf])
def test_side_wall_must_be_positive_finite(bad):
    with pytest.raises(ValueError):
        WorldConfig(side_wall_mm=bad)


# --- snapshot: ključevi ------------------------------------------------------

def test_snapshot_keys_unchanged_at_zero_sign():
    w = ChannelWorld(WorldConfig(dims=3), 2)
    w.reset()
    p = w.make_probe()
    assert set(p.snapshot()) == {"x", "dims", "t_ms", "escaped", "x_perp_norm", "t_escape_ms", "phi", "n"}


@pytest.mark.parametrize("sign", [-1, 1])
def test_snapshot_adds_keys_when_sign_nonzero(sign):
    w = ChannelWorld(WorldConfig(dims=3, escape_sign=sign), 2)
    w.reset()
    p = w.make_probe()
    w.step(_rev_quad(0), 1.0)
    s = p.snapshot()
    assert s["escape_sign"] == sign and isinstance(s["escape_sign"], int)
    assert isinstance(s["x_perp_n"], float)
    assert s["x_perp_n"] == pytest.approx(float(np.dot(w.probe().x[1:], s["n"])), abs=1e-12)


# --- zid: sila i bekstvo po projekciji ---------------------------------------

def test_wall_pushes_back_with_documented_formula():
    # Bez šuma i kappa člana, jedini pomak x_perp je zid: x += dt·k·(−side − p)·s·n.
    cfg = WorldConfig(
        dims=2, escape_sign=1, side_wall_mm=0.5, obs_noise=0.0,
        sigma_perp_mm_per_sqrt_ms=0.0, kappa_per_ms=0.0,
    )
    w = ChannelWorld(cfg, 7)
    w.reset()
    w._x_perp = np.array([-2.0])  # p = -2 < -0.5
    n = w._n.copy()
    p0 = float(np.dot(w._x_perp, n))
    depth = -cfg.side_wall_mm - p0  # 1.5 mm
    expect = w._x_perp + 0.01 * cfg.k_wall_per_ms * depth * 1.0 * n
    w.step(np.full(4, 0.5), 0.01)
    np.testing.assert_allclose(w._x_perp, expect, rtol=0, atol=1e-12)


def test_escape_uses_projection_not_norm():
    # escape_sign=-1: p = -(x·n). Pomak ka -n (x·n<0) NE sme da bude bekstvo, ka +n jeste.
    cfg = WorldConfig(dims=2, escape_sign=-1, obs_noise=0.0, escape_threshold_mm=1.0, escape_hold_ms=5.0)
    w = ChannelWorld(cfg, 3)
    w.reset()
    w._x_perp = np.array([-3.0])  # p = +3 > 1 -> bekstvo nakon 5 ms
    for _ in range(10):
        w.step(np.full(4, 0.5), 1.0)
    assert w.probe().escaped
    w2 = ChannelWorld(cfg, 3)
    w2.reset()
    w2._x_perp = np.array([3.0])  # p = -3 < -0.5 -> zid, nikad bekstvo
    for _ in range(50):
        w2.step(np.full(4, 0.5), 1.0)
    assert not w2.probe().escaped
