"""Digest putanje sveta (faza 1 i 2) za dokaz bit-po-bit jednakosti pri roll_gain=0."""
from __future__ import annotations

import hashlib

import numpy as np

from neural.core.seed import derive_seed
from neural.environment.channel_world import ChannelWorld, WorldConfig

CASES = {
    "dims2": (2, False),
    "dims3": (3, False),
    "dims3_geom": (3, True),
}


def build(case: str) -> ChannelWorld:
    dims, geom = CASES[case]
    if geom:
        return ChannelWorld(
            WorldConfig(dims=dims), derive_seed(5, "world.ep0"), geometry_seed=derive_seed(5, "world")
        )
    return ChannelWorld(WorldConfig(dims=dims), seed=derive_seed(5, "world"))


def digest(case: str, n: int = 3000) -> str:
    w = build(case)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(w.reset().as_array()).tobytes())
    probe = w.probe
    arng = np.random.default_rng(11)
    for k in range(n):
        if k % 2:
            a = arng.uniform(0, 1, 4)
        else:
            q = 2 * np.pi * k / 1000.0
            a = np.clip([0.5 + 0.5 * np.sin(q), 0.5 - 0.5 * np.sin(q), 0.5 + 0.5 * np.cos(q), 0.5 - 0.5 * np.cos(q)], 0, 1)
        h.update(np.ascontiguousarray(w.step(a).as_array()).tobytes())
        r = probe()
        h.update(np.ascontiguousarray(r.x).tobytes())
        h.update(repr((r.x_perp_norm, r.escaped, r.t_escape_ms, r.t_ms)).encode())
    return h.hexdigest()
