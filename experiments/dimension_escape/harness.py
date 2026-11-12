"""Harness: jedini deo koji vidi svet, telo i probe. Organizam dobija samo (9,) nA i vraća (4,) nA.

Stvarni API: ChannelWorld(WorldConfig(dims=d), seed=derive_seed(s,'world')),
Body(world, BodyConfig(dt_ms=dt), rng=subsystem_rng(s,'body')), I_s = body.reset(),
I_s = body.step(I_pm_nA, dt_ms); bekstvo se čita isključivo iz probe snapshot-a sveta
(ključevi 'x', 'escaped', 't_escape_ms'), bez sopstvenog praga. q se rekonstruiše istim
niskopropusnim filtrom aktivacije kao u svetu (bez šuma) iz body.last_commands.
"""
from __future__ import annotations

import dataclasses
import os
import time

import numpy as np

from neural.body import Body, BodyConfig
from neural.body.proprio import activation_step
from neural.core.seed import derive_seed, subsystem_rng
from neural.environment import ChannelWorld, WorldConfig
from neural.organism import Organism, OrganismConfig

CONDITIONS = ("full", "random_motor", "frozen", "no_da")
OU_TAU_MS = 200.0
OU_MEAN_NA = 1.0
OU_SD_NA = 0.5


def _snapshot(world) -> dict:
    """Probe snapshot sveta (probe se kešira jednom po svetu)."""
    if hasattr(world, "make_probe"):
        pr = getattr(world, "_harness_probe", None)
        if pr is None:
            pr = world.make_probe()
            world._harness_probe = pr
        return pr.snapshot()
    return dataclasses.asdict(world.probe())


class OUMotor:
    """OU generator (tau=200 ms) sa srednjom 1.0 nA i SD 0.5 nA; izbor fiksnih vrednosti
    umesto vrednosti iz frozen uslova je zbog jednostavnosti (docs/experiment_design.md)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.x = OU_MEAN_NA + OU_SD_NA * self.rng.standard_normal(4)

    def step(self, dt_ms: float) -> np.ndarray:
        a = np.exp(-dt_ms / OU_TAU_MS)
        self.x = OU_MEAN_NA + a * (self.x - OU_MEAN_NA) + OU_SD_NA * np.sqrt(1 - a * a) * self.rng.standard_normal(4)
        return np.maximum(self.x, 0.0)


def _analytic_phase(x: np.ndarray) -> np.ndarray:
    n = len(x)
    X = np.fft.fft(x - x.mean())
    h = np.zeros(n)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.angle(np.fft.ifft(X * h))


def secondary_metrics(t_ms, qA, qB) -> dict:
    t = np.asarray(t_ms, float) / 1000.0
    if len(t) < 8:
        return {"plv": None, "mean_sin_dphi": None, "mean_abs_area_rate": None}
    dphi = _analytic_phase(qA) - _analytic_phase(qB)
    dtq = np.gradient(t)
    dA, dB = np.gradient(qA, t), np.gradient(qB, t)
    area = 0.5 * (qA * dB - qB * dA)
    return {"plv": float(np.abs(np.mean(np.exp(1j * dphi)))),
            "mean_sin_dphi": float(np.mean(np.sin(dphi))),
            "mean_abs_area_rate": float(np.mean(np.abs(area)))}


def world_config_kwargs(dims: int, roll_gain: float = 0.0, escape_sign: int = 0,
                        side_wall_mm: float | None = None) -> dict:
    """Argumenti za WorldConfig; podrazumevane vrednosti daju tačno {'dims': dims} (faze 1/2)."""
    if isinstance(escape_sign, bool) or not isinstance(escape_sign, int):
        raise ValueError(f"escape_sign mora biti int iz {{-1, 0, 1}}, dobijeno {escape_sign!r}")
    if side_wall_mm is not None and escape_sign == 0:
        raise ValueError("side_wall_mm ima smisla samo uz escape_sign != 0")
    kw: dict = {"dims": dims}
    if roll_gain != 0.0:
        kw["roll_gain"] = roll_gain
    if escape_sign != 0:
        kw["escape_sign"] = escape_sign
        if side_wall_mm is not None:
            kw["side_wall_mm"] = side_wall_mm
    return kw


def run_one(cond: str, seed: int, dims: int, duration_ms: float, dt_ms: float,
            trace_every_ms: float, out_dir: str | None, org_cfg: dict | None = None,
            roll_gain: float = 0.0, escape_sign: int = 0,
            side_wall_mm: float | None = None, record: bool = False) -> dict:
    if cond not in CONDITIONS:
        raise ValueError(f"nepoznat uslov: {cond}")
    if record and not out_dir:
        raise ValueError("record=True zahteva out_dir")
    # podrazumevane vrednosti: isti WorldConfig(dims=dims) kao u fazama 1 i 2 (bit-po-bit isti tok)
    wcfg = WorldConfig(**world_config_kwargs(dims, roll_gain, escape_sign, side_wall_mm))
    world = ChannelWorld(wcfg, seed=derive_seed(seed, "world"))
    body = Body(world, BodyConfig(dt_ms=dt_ms), rng=subsystem_rng(seed, "body"))
    cfg = OrganismConfig(**org_cfg) if org_cfg else OrganismConfig()
    org = None
    ou = None
    if cond == "random_motor":
        ou = OUMotor(derive_seed(seed, "random_motor"))
    else:
        org = Organism(cfg, seed=derive_seed(seed, "organism"), plasticity=(cond != "frozen"),
                       dopamine=(cond != "no_da"))

    rec = None
    if record:
        from .recording import Recorder
        rec = Recorder(org)
    n_steps = int(round(duration_ms / dt_ms))
    every = max(1, int(round(trace_every_ms / dt_ms)))
    I_s = body.reset()
    I_pm = np.zeros(4)
    a = np.full(4, 0.5)
    ts, xs, cmds, das, ws, qs = [], [], [], [], [], []
    escaped, t_escape = False, None
    phis: list[float] = []
    xpn: list[float] = []
    t0 = time.time()
    for k in range(n_steps):
        I_s = body.step(I_pm, dt_ms)
        a = activation_step(a, np.clip(body.last_commands, 0, 1), dt_ms, wcfg.tau_act_ms)
        if ou is not None:
            I_pm = ou.step(dt_ms)
        else:
            assert org is not None
            I_pm = org.step(I_s, dt_ms)
            if rec is not None:
                rec.on_step(k * dt_ms, org)
        if k % every == 0:
            snap = _snapshot(world)
            ts.append(k * dt_ms)
            xs.append(np.asarray(snap["x"], float).ravel())
            cmds.append(I_pm.copy())
            das.append(org.da_phasic if org is not None else 0.0)
            if rec is not None:
                rec.on_sample(k * dt_ms, I_pm, body.last_commands, das[-1], org)
            ws.append(float(org.cross_weights().mean()) if org is not None else 0.0)
            qs.append(np.array([a[0] - a[1], a[2] - a[3]]))
            if roll_gain != 0.0:
                phis.append(float(snap["phi"]))  # φ samo iz probe snapshot-a, nikad iz Observation
            if escape_sign != 0:
                xpn.append(float(snap["x_perp_n"]))  # samo iz probe snapshot-a
            if snap["escaped"]:
                escaped, t_escape = True, snap["t_escape_ms"]
                break
    t_arr = np.array(ts)
    x_arr = np.array(xs)
    q_arr = np.array(qs)
    cmd_arr: np.ndarray = np.asarray(cmds, dtype=float)
    da_arr: np.ndarray = np.asarray(das, dtype=float)
    w_arr: np.ndarray = np.asarray(ws, dtype=float)
    res: dict = {"cond": cond, "seed": seed, "escaped": escaped, "t_escape_ms": t_escape,
           "wall_s": time.time() - t0, "t": t_arr, "x": x_arr, "da": da_arr,
           "w_cross": w_arr, "q": q_arr, "cmd": cmd_arr}
    if roll_gain != 0.0:
        res["roll_gain"] = float(roll_gain)
        res["phi"] = np.asarray(phis, dtype=float)
    if escape_sign != 0:
        res["escape_sign"] = int(escape_sign)
        res["side_wall_mm"] = float(wcfg.side_wall_mm)
        res["x_perp_n"] = np.asarray(xpn, dtype=float)
    res["secondary"] = secondary_metrics(t_arr, q_arr[:, 0], q_arr[:, 1]) if len(t_arr) else {}
    if rec is not None:
        assert out_dir is not None
        res["recording_path"] = rec.save(os.path.join(out_dir, "recordings", f"{cond}_seed{seed}.npz"))
    if out_dir:
        tdir = os.path.join(out_dir, "traces")
        os.makedirs(tdir, exist_ok=True)
        extra = {"x_perp_n": res["x_perp_n"]} if escape_sign != 0 else {}
        np.savez_compressed(os.path.join(tdir, f"{cond}_seed{seed}.npz"), t=t_arr, x=x_arr,
                            cmd=cmd_arr, da=da_arr, w_cross_mean=w_arr,
                            q_A=q_arr[:, 0], q_B=q_arr[:, 1], escaped=escaped,
                            t_escape_ms=-1.0 if t_escape is None else t_escape, **extra)
    return res


def run_task(args):
    """Ulaz za ProcessPoolExecutor (picklable na nivou modula; Windows spawn).

    Oblici args: 8 članova (faze 1/2), 9 (+roll_gain), 10 (+roll_gain, escape_sign),
    11 (+roll_gain, escape_sign, side_wall_mm), 12 (+record)."""
    return run_one(*args)
