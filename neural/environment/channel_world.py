"""Kanal sa skrivenom osom: svet za eksperiment „1D organizam otkriva višu dimenziju“.

Organizam dobija samo Observation. Koordinate vidi isključivo harness preko probe().
Jedinice: mm i ms. Dinamika je Euler / Euler–Maruyama, redom:
aktivacije -> roll (samo dims=3, roll_gain != 0) -> aksijalni pomak (bez x_perp, q_B i šuma) -> Lévyjeva površina
-> jednostrani zid (samo escape_sign != 0) -> bekstvo (histerezis) -> vreme.

Jednostrani zid (escape_sign = s ≠ 0): p = s·(x_perp·n). Ako p < −side_wall_mm,
x_perp += dt·k_wall_per_ms·(−side_wall_mm − p)·s·n, a dubina prodora (−side_wall_mm − p ≥ 0)
ulazi bez znaka u ventralni dodirni kanal: touch[2] = min(1, exp(...) + tanh(dubina / _TOUCH_SCALE_MM)).
Bekstvo se tada meri preko p (ne ‖x_perp‖). Pri escape_sign = 0 ništa od toga se ne izvršava.

RNG: dinamika koristi np.random.default_rng(seed) sa redosledom x0, x_src,
pa n (samo dims>=3), pa xi (dims−1) po koraku. Šum opservacije (9 po
opservaciji) ima POSEBAN tok subsystem_rng(seed, "channel_world.obs_noise").
Zato broj izvlačenja xi i n (zavisan od dims) ne meša tok šuma opservacije,
ali Observation i dalje zavisi od dims preko x_perp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from neural.core.seed import subsystem_rng
from neural.body.proprio import activation_step, proprio_vector
from neural.environment.types import Observation, ProbeReading

CONFIDENCE = "speculative"
N_ACTUATORS = 4
N_SENSORS = 9
_TOUCH_SCALE_MM = 0.05  # ASSUMPTION: skala tanh za dodir (dizajn eksperimenta)


@dataclass(frozen=True)
class WorldConfig:
    dims: int = 2
    channel_length_mm: float = 20.0
    body_length_mm: float = 1.0
    k_axial_mm_per_ms: float = 0.002
    k_wall_per_ms: float = 0.05
    k_perp_mm: float = 4.0
    kappa_per_ms: float = 0.002
    sigma_perp_mm_per_sqrt_ms: float = 0.0063
    groove_width_mm: float = 0.5
    chem_lambda_mm: float = 5.0
    tau_act_ms: float = 50.0
    obs_noise: float = 0.01
    escape_threshold_mm: float = 1.0
    escape_hold_ms: float = 500.0
    roll_gain: float = 0.0
    escape_sign: int = 0
    side_wall_mm: float = 0.5

    def __post_init__(self) -> None:
        if self.dims < 2:
            raise ValueError(f"dims mora biti >= 2, dobijeno {self.dims}")
        if not math.isfinite(self.roll_gain):
            raise ValueError(f"roll_gain mora biti konačan, dobijeno {self.roll_gain}")
        if self.roll_gain != 0.0 and self.dims != 3:
            raise ValueError("roll_gain != 0 je dozvoljen samo za dims == 3")
        if isinstance(self.escape_sign, bool) or not isinstance(self.escape_sign, int) or self.escape_sign not in (-1, 0, 1):
            raise ValueError(f"escape_sign mora biti tačno -1, 0 ili 1, dobijeno {self.escape_sign!r}")
        if not (math.isfinite(self.side_wall_mm) and self.side_wall_mm > 0.0):
            raise ValueError(f"side_wall_mm mora biti konačan i > 0, dobijeno {self.side_wall_mm}")


_UNITS = {
    "dims": "1",
    "channel_length_mm": "mm",
    "body_length_mm": "mm",
    "k_axial_mm_per_ms": "mm/ms",
    "k_wall_per_ms": "1/ms",
    "k_perp_mm": "mm",
    "kappa_per_ms": "1/ms",
    "sigma_perp_mm_per_sqrt_ms": "mm/sqrt(ms)",
    "groove_width_mm": "mm",
    "chem_lambda_mm": "mm",
    "tau_act_ms": "ms",
    "obs_noise": "1",
    "escape_threshold_mm": "mm",
    "escape_hold_ms": "ms",
    "roll_gain": "rad/ms",
    "escape_sign": "1",
    "side_wall_mm": "mm",
}
PARAMS_META = {
    f.name: {
        "value": f.default,
        "unit": _UNITS[f.name],
        "source": "ASSUMPTION: dizajn eksperimenta dimension_escape, nije izvedeno iz literature",
        "confidence": "speculative",
    }
    for f in fields(WorldConfig)
}


class ChannelWorld:
    def __init__(
        self, config: WorldConfig, seed: int, geometry_seed: int | None = None
    ) -> None:
        """geometry_seed (opciono): ako je zadat, osa n (dims>=3) se crta iz
        subsystem_rng(geometry_seed, "channel_world.n"), a x0 i x_src ostaju iz
        epizodnog seeda. Epizodni seed je derive_seed(seed, "world.ep<k>"), k od 0.
        Bez njega ponašanje je kao u fazi 1 (bit-po-bit)."""
        self._cfg = config
        self._seed = int(seed)
        self._geometry_seed = None if geometry_seed is None else int(geometry_seed)
        self._rng: np.random.Generator | None = None
        self._obs_rng: np.random.Generator | None = None
        self.n_actuators = N_ACTUATORS
        self.n_sensors = N_SENSORS
        self._t = 0.0
        self._x0 = 0.0
        self._x_perp = np.zeros(config.dims - 1)
        self._a = np.full(N_ACTUATORS, 0.5)
        self._x_src = 0.0
        self._n0 = np.zeros(config.dims - 1)
        self._phi = 0.0
        self._n = np.zeros(config.dims - 1)
        self._hold = 0.0
        self._escaped = False
        self._t_escape: float | None = None
        self._wall_depth = 0.0

    def reset(
        self, seed: int | None = None, geometry_seed: int | None = None
    ) -> Observation:
        c = self._cfg
        if seed is not None:
            self._seed = int(seed)
        if geometry_seed is not None:
            self._geometry_seed = int(geometry_seed)
        self._obs_rng = subsystem_rng(self._seed, "channel_world.obs_noise")
        rng = np.random.default_rng(self._seed)
        self._rng = rng
        self._t = 0.0
        self._x0 = c.channel_length_mm / 2 + rng.uniform(-2.0, 2.0)
        self._x_perp = np.zeros(c.dims - 1)
        self._a = np.full(N_ACTUATORS, 0.5)
        self._x_src = rng.uniform(2.0, c.channel_length_mm - 2.0)
        if c.dims >= 3:
            if self._geometry_seed is not None:
                # n iz posebnog toka: isti n kroz epizode sa različitim seedom
                grng = subsystem_rng(self._geometry_seed, "channel_world.n")
                v = grng.standard_normal(c.dims - 1)
            else:
                v = rng.standard_normal(c.dims - 1)
            self._n0 = v / np.linalg.norm(v)
        else:
            self._n0 = np.array([1.0])
        self._phi = 0.0
        self._n = self._n0.copy()
        self._hold = 0.0
        self._escaped = False
        self._t_escape = None
        self._wall_depth = 0.0
        return self._observe(self._obs_rng.standard_normal(N_SENSORS))

    def step(self, action: np.ndarray, dt_ms: float = 1.0) -> Observation:
        c = self._cfg
        if self._rng is None or self._obs_rng is None:
            raise RuntimeError("ChannelWorld: pozovi reset() pre step()")
        dt = float(dt_ms)
        if not (0.0 < dt <= 1.0):
            raise ValueError(f"dt_ms mora biti u (0, 1], dobijeno {dt}")
        cmd = np.asarray(action, dtype=np.float64)
        if cmd.shape != (N_ACTUATORS,):
            raise ValueError(f"action mora imati oblik (4,), dobijeno {cmd.shape}")

        rng = self._rng
        xi = rng.standard_normal(c.dims - 1)
        noise = self._obs_rng.standard_normal(N_SENSORS)
        u = np.clip(cmd, 0.0, 1.0)

        # 1) aktivacije; q_A_old / q_B_old pre ažuriranja
        a_old = self._a
        qA_old = a_old[0] - a_old[1]
        qB_old = a_old[2] - a_old[3]
        a = activation_step(a_old, u, dt, c.tau_act_ms)
        self._a = a
        qA = a[0] - a[1]
        qB = a[2] - a[3]

        # 1b) roll: rotacija ose n oko ose kanala (samo dims=3)
        if c.roll_gain != 0.0:
            self._phi += dt * c.roll_gain * (min(a[0], a[1]) - min(a[2], a[3]))
            cp, sp = math.cos(self._phi), math.sin(self._phi)
            n0 = self._n0
            self._n = np.array([cp * n0[0] - sp * n0[1], sp * n0[0] + cp * n0[1]])

        # 2) aksijalno (ne zavisi od x_perp, q_B ni od šuma)
        half = c.body_length_mm / 2
        L = c.channel_length_mm
        x0 = self._x0
        p_f = max(0.0, x0 + half - L)
        p_r = max(0.0, half - x0)
        self._x0 = x0 + dt * (c.k_axial_mm_per_ms * qA - c.k_wall_per_ms * p_f + c.k_wall_per_ms * p_r)

        # 3) Lévyjeva površina i poprečni pomak
        d_area = 0.5 * (qA_old * (qB - qB_old) - qB_old * (qA - qA_old))
        self._x_perp = (
            self._x_perp
            + c.k_perp_mm * d_area * self._n
            - dt * c.kappa_per_ms * self._x_perp
            + c.sigma_perp_mm_per_sqrt_ms * np.sqrt(dt) * xi
        )

        # 3b) jednostrani meki zid (samo escape_sign != 0); dubina ide u touch[2]
        self._wall_depth = 0.0
        if c.escape_sign != 0:
            s = float(c.escape_sign)
            p = s * float(np.dot(self._x_perp, self._n))
            depth = -c.side_wall_mm - p
            if depth > 0.0:
                self._wall_depth = depth
                self._x_perp = self._x_perp + dt * c.k_wall_per_ms * depth * s * self._n

        # 4) bekstvo: histerezis, zaključano kad se prvi put dogodi
        if c.escape_sign != 0:
            crit = float(c.escape_sign) * float(np.dot(self._x_perp, self._n))
        else:
            crit = float(np.linalg.norm(self._x_perp))
        if crit > c.escape_threshold_mm:
            self._hold += dt
        else:
            self._hold = 0.0
        if (not self._escaped) and self._hold >= c.escape_hold_ms - 1e-9:
            self._escaped = True
            self._t_escape = self._t

        # 5) vreme
        self._t += dt
        return self._observe(noise)

    def make_probe(self) -> "ChannelProbe":
        return ChannelProbe(self)

    def probe(self) -> ProbeReading:
        x = np.concatenate(([self._x0], self._x_perp)).astype(np.float64)
        return ProbeReading(
            x=x,
            x_perp_norm=float(np.linalg.norm(self._x_perp)),
            escaped=self._escaped,
            t_escape_ms=self._t_escape,
            t_ms=self._t,
        )

    # --- interno ---------------------------------------------------------

    def _conc(self, x: float) -> float:
        lam = self._cfg.chem_lambda_mm
        return float(np.exp(-((x - self._x_src) ** 2) / (2.0 * lam * lam)))

    def _observe(self, noise: np.ndarray) -> Observation:
        c = self._cfg
        half = c.body_length_mm / 2
        x0 = self._x0
        p_f = max(0.0, x0 + half - c.channel_length_mm)
        p_r = max(0.0, half - x0)
        perp = float(np.linalg.norm(self._x_perp))
        s = c.obs_noise
        ventral = float(np.exp(-((perp / c.groove_width_mm) ** 2)))
        if self._wall_depth > 0.0:
            ventral += float(np.tanh(self._wall_depth / _TOUCH_SCALE_MM))
            ventral = min(1.0, ventral)
        touch = np.array(
            [
                np.tanh(p_f / _TOUCH_SCALE_MM),
                np.tanh(p_r / _TOUCH_SCALE_MM),
                ventral,
            ]
        )
        touch = np.maximum(touch + s * noise[0:3], 0.0)
        proprio = proprio_vector(self._a, s * noise[3:7])
        chem = np.array([self._conc(x0 + half), self._conc(x0 - half)])
        chem = np.maximum(chem + s * noise[7:9], 0.0)
        return Observation(touch=touch, proprio=proprio, chem_gradient=chem)


class ChannelProbe:
    """Sonda samo za harness: nikad se ne prosleđuje organizmu."""

    def __init__(self, world: ChannelWorld) -> None:
        self._w = world

    def snapshot(self) -> dict:
        w = self._w
        x = np.concatenate(([w._x0], w._x_perp)).astype(np.float64)
        snap = {
            "x": x,
            "dims": int(w._cfg.dims),
            "t_ms": float(w._t),
            "escaped": bool(w._escaped),
            "x_perp_norm": float(np.linalg.norm(w._x_perp)),
            "t_escape_ms": w._t_escape,
            "phi": float(w._phi),
            "n": np.array(w._n, dtype=np.float64, copy=True),
        }
        if w._cfg.escape_sign != 0:
            snap["escape_sign"] = int(w._cfg.escape_sign)
            snap["x_perp_n"] = float(np.dot(w._x_perp, w._n))
        return snap
