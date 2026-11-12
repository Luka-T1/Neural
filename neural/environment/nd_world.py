"""Opšti N-D svet sa zidovima i jednom skrivenom osom.

Osa `_hidden` nema nijedan senzor: dodir vidi samo zidove vidljivih osa,
hemija je lokalna koncentracija na receptorima (skalari), a propriocepcija
su aktivacije aktuatora. Skrivena osa se pomera samo Lévyjevom površinom
parova aktuatora (0,1) i (2,3), kao u ChannelWorld.

Koordinate, n_dims, geometrija i probe su privatni (`_`) ili dostupni samo
preko make_probe(). Organizam dobija isključivo Observation.
Sav RNG je lokalni np.random.default_rng(seed) iz reset(seed).
"""

from __future__ import annotations

import numpy as np

from neural.body.proprio import activation_step, proprio_vector
from neural.environment.types import Observation

CONFIDENCE = "speculative"

_DEFAULTS: dict[str, float] = {
    "dt_ms": 1.0,
    "half_size_mm": 10.0,
    "k_drive_mm_per_ms": 0.002,
    "k_perp_mm": 4.0,
    "kappa_per_ms": 0.002,
    "sigma_perp_mm_per_sqrt_ms": 0.0063,
    "touch_scale_mm": 0.05,
    "groove_width_mm": 0.5,
    "chem_lambda_mm": 5.0,
    "chem_offset_mm": 0.5,
    "tau_act_ms": 50.0,
    "obs_noise": 0.01,
    "escape_threshold_mm": 1.0,
    "escape_hold_ms": 500.0,
}
_UNITS = {
    "dt_ms": "ms",
    "half_size_mm": "mm",
    "k_drive_mm_per_ms": "mm/ms",
    "k_perp_mm": "mm",
    "kappa_per_ms": "1/ms",
    "sigma_perp_mm_per_sqrt_ms": "mm/sqrt(ms)",
    "touch_scale_mm": "mm",
    "groove_width_mm": "mm",
    "chem_lambda_mm": "mm",
    "chem_offset_mm": "mm",
    "tau_act_ms": "ms",
    "obs_noise": "1",
    "escape_threshold_mm": "mm",
    "escape_hold_ms": "ms",
}
PARAMS_META = {
    k: {
        "value": v,
        "unit": _UNITS[k],
        "source": "ASSUMPTION: podrazumevani parametri sveta (dizajn eksperimenta), nije iz literature",
        "confidence": "speculative",
    }
    for k, v in _DEFAULTS.items()
}


class NDProbe:
    """Samo za harness. Organizam ga nikad ne dobija."""

    def __init__(self, world: "NDWorld") -> None:
        self._world = world

    def snapshot(self) -> dict:
        w = self._world
        return {
            "position": w._pos.copy(),
            "n_dims": w._n_dims,
            "walls": w._walls.copy(),
            "escaped": bool(w._escaped),
            "x": w._pos.astype(np.float64, copy=True),
            "dims": int(w._n_dims),
            "t_ms": float(w._t),
        }


class NDWorld:
    CONFIDENCE = CONFIDENCE
    PARAMS_META = PARAMS_META

    def __init__(self, n_dims: int, n_touch: int, n_chemo: int, n_actuators: int, **params: float) -> None:
        if int(n_dims) < 2:
            raise ValueError(f"n_dims mora biti >= 2, dobijeno {n_dims}")
        if int(n_touch) < 1 or int(n_chemo) < 1:
            raise ValueError("n_touch i n_chemo moraju biti >= 1")
        if int(n_actuators) < 4 or int(n_actuators) % 2:
            raise ValueError("n_actuators mora biti paran broj >= 4")
        unknown = set(params) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"nepoznati parametri sveta: {sorted(unknown)}")

        self._p = {**_DEFAULTS, **params}
        self._n_dims = int(n_dims)
        self._n_touch = int(n_touch)
        self._n_chemo = int(n_chemo)
        self._n_pairs = int(n_actuators) // 2
        self._n_act = int(n_actuators)
        self._rng: np.random.Generator | None = None

        # javni atributi (samo ovi)
        self.n_actuators = self._n_act
        self.n_sensors = self._n_touch + self._n_act + self._n_chemo

    # --- životni ciklus ---------------------------------------------------

    def reset(self, seed: int) -> Observation:
        p = self._p
        d = self._n_dims
        rng = np.random.default_rng(int(seed))
        self._rng = rng

        self._hidden = int(rng.integers(d))
        self._vis = np.array([i for i in range(d) if i != self._hidden], dtype=int)

        half = p["half_size_mm"] * rng.uniform(0.8, 1.2, size=d)
        walls = np.column_stack([-half, half])
        walls[self._hidden] = (-np.inf, np.inf)
        self._walls = walls

        M = rng.standard_normal((d, self._n_pairs))
        M[self._hidden, :] = 0.0
        self._M = M

        src = np.zeros(d)
        vis = self._vis
        src[vis] = rng.uniform(-0.5 * half[vis], 0.5 * half[vis])
        self._src = src

        off = rng.standard_normal((self._n_chemo, d))
        off[:, self._hidden] = 0.0
        norms = np.linalg.norm(off, axis=1, keepdims=True)
        self._offsets = off / norms * p["chem_offset_mm"]

        self._pos = np.zeros(d)
        self._a = np.full(self._n_act, 0.5)
        self._t = 0.0
        self._hold = 0.0
        self._escaped = False
        self._t_escape: float | None = None

        noise = rng.standard_normal(self.n_sensors)
        nv = len(vis)
        return self._observe(noise, np.zeros(nv), np.zeros(nv))

    def step(self, action: np.ndarray, dt_ms: float | None = None) -> Observation:
        if self._rng is None:
            raise RuntimeError("NDWorld: pozovi reset(seed) pre step()")
        p = self._p
        act = np.asarray(action, dtype=np.float64)
        if act.shape != (self._n_act,):
            raise ValueError(f"akcija mora imati oblik ({self._n_act},), dobijeno {act.shape}")
        dt = float(p["dt_ms"] if dt_ms is None else dt_ms)
        if not (0.0 < dt <= 1.0):
            raise ValueError(f"dt_ms mora biti u (0, 1], dobijeno {dt}")

        rng = self._rng
        xi_h = rng.standard_normal()
        noise = rng.standard_normal(self.n_sensors)
        u = np.clip(act, 0.0, 1.0)

        a_old = self._a
        q_old = a_old[0::2] - a_old[1::2]
        a = activation_step(a_old, u, dt, p["tau_act_ms"])
        self._a = a
        q = a[0::2] - a[1::2]

        # vidljive ose: kinematika, zatim zid (tvrdo ograničenje); prodor se meri pre korekcije
        vis = self._vis
        cand = self._pos[vis] + dt * p["k_drive_mm_per_ms"] * (self._M[vis] @ q)
        lo = self._walls[vis, 0]
        hi = self._walls[vis, 1]
        depth_hi = np.maximum(0.0, cand - hi)
        depth_lo = np.maximum(0.0, lo - cand)
        pos = self._pos.copy()
        pos[vis] = np.clip(cand, lo, hi)

        # skrivena osa: samo Lévyjeva površina parova (0,1)
        h = self._hidden
        d_area = 0.5 * (q_old[0] * (q[1] - q_old[1]) - q_old[1] * (q[0] - q_old[0]))
        pos[h] = (
            pos[h]
            + p["k_perp_mm"] * d_area
            - dt * p["kappa_per_ms"] * pos[h]
            + p["sigma_perp_mm_per_sqrt_ms"] * np.sqrt(dt) * xi_h
        )
        self._pos = pos

        if abs(pos[h]) > p["escape_threshold_mm"]:
            self._hold += dt
        else:
            self._hold = 0.0
        if (not self._escaped) and self._hold >= p["escape_hold_ms"] - 1e-9:
            self._escaped = True
            self._t_escape = self._t
        self._t += dt

        return self._observe(noise, depth_hi, depth_lo)

    def make_probe(self) -> NDProbe:
        return NDProbe(self)

    # --- interno ----------------------------------------------------------

    def _observe(self, noise: np.ndarray, depth_hi: np.ndarray, depth_lo: np.ndarray) -> Observation:
        p = self._p
        s = p["obs_noise"]
        nv = len(self._vis)
        nt, na = self._n_touch, self._n_act

        faces = np.arange(nt) % (2 * nv)
        axis_idx = faces // 2
        upper = (faces % 2) == 0
        depths = np.where(upper, depth_hi[axis_idx], depth_lo[axis_idx])
        touch = np.tanh(depths / p["touch_scale_mm"])
        touch = np.maximum(touch + s * noise[:nt], 0.0)

        proprio = proprio_vector(self._a, s * noise[nt:nt + na])

        pts = self._pos[self._vis][None, :] + self._offsets[:, self._vis]
        d2 = np.sum((pts - self._src[self._vis][None, :]) ** 2, axis=1)
        chem = np.exp(-d2 / (2.0 * p["chem_lambda_mm"] ** 2))
        chem = np.maximum(chem + s * noise[nt + na:], 0.0)

        return Observation(touch=touch, proprio=proprio, chem_gradient=chem)
