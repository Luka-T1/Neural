"""Deterministic per-subsystem random streams."""
from __future__ import annotations

from hashlib import sha256

import numpy as np


def _key(name: str) -> int:
    return int.from_bytes(sha256(name.encode("utf-8")).digest()[:4], "little")


def subsystem_rng(master_seed: int, name: str) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(_key(name),))
    return np.random.Generator(np.random.PCG64(ss))


def derive_seed(master_seed: int, name: str) -> int:
    ss = np.random.SeedSequence(entropy=master_seed, spawn_key=(_key(name),))
    return int(ss.generate_state(1, dtype=np.uint32)[0])
