"""CLI: python -m experiments.dimension_escape --seeds N --dims 2 --condition all ..."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from neural.environment import WorldConfig
from neural.organism import OrganismConfig

from .harness import CONDITIONS, run_task, world_config_kwargs
from .report import build_report, render_markdown


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _collect(it, results):
    for r in it:
        results[r["cond"]].append(r)
        logging.info("%s seed=%d escaped=%s t_escape=%s wall=%.1fs", r["cond"], r["seed"], r["escaped"],
                     r["t_escape_ms"], r["wall_s"])
        print(f"{r['cond']:>12} seed={r['seed']:<3} escaped={r['escaped']} wall={r['wall_s']:.0f}s", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experiments.dimension_escape")
    ap.add_argument("--seeds", type=int, default=20, help="N: seedovi 0..N-1")
    ap.add_argument("--dims", type=int, default=2)
    ap.add_argument("--condition", default="all", choices=("all",) + CONDITIONS)
    ap.add_argument("--duration-ms", type=float, default=60000.0)
    ap.add_argument("--roll-gain", type=float, default=0.0, help="rad/ms; samo za dims=3 (faza 3)")
    ap.add_argument("--escape-sign", type=int, choices=(-1, 0, 1), default=0,
                    help="jednostrani zid: +1/-1 (faza 4); 0 = isključeno")
    ap.add_argument("--side-wall-mm", type=float, default=None, help="mm, >0; samo uz --escape-sign != 0")
    ap.add_argument("--record", action="store_true", help="snimi spajkove i stanja u <out>/recordings + timeline.json")
    ap.add_argument("--dt-ms", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--trace-every-ms", type=float, default=10.0)
    a = ap.parse_args(argv)
    if a.roll_gain != 0.0 and a.dims != 3:
        ap.error("--roll-gain != 0 je dozvoljen samo uz --dims 3")

    if a.side_wall_mm is not None:
        if not math.isfinite(a.side_wall_mm) or a.side_wall_mm <= 0:
            ap.error("--side-wall-mm mora biti konačan i > 0")
        if a.escape_sign == 0:
            ap.error("--side-wall-mm zahteva --escape-sign != 0")

    out = a.out or os.path.join("experiments", "dimension_escape", "results", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    logging.basicConfig(filename=os.path.join(out, "run.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s", force=True)
    seeds = list(range(a.seeds))
    conds = list(CONDITIONS) if a.condition == "all" else [a.condition]
    extra: tuple = ()
    if a.roll_gain != 0.0 or a.escape_sign != 0:
        extra = (a.roll_gain,)
    if a.escape_sign != 0:
        extra += (a.escape_sign,)
        if a.side_wall_mm is not None:
            extra += (a.side_wall_mm,)
    if a.record:
        extra = (a.roll_gain, a.escape_sign, a.side_wall_mm, True)
    tasks = [(c, s, a.dims, a.duration_ms, a.dt_ms, a.trace_every_ms, out, None) + extra
             for c in conds for s in seeds]
    workers = a.workers or min(os.cpu_count() or 1, len(tasks))
    omit = {"roll_gain": 0.0, "escape_sign": 0, "side_wall_mm": None, "record": False}
    args_cfg = {k: v for k, v in vars(a).items() if not (k in omit and v == omit[k])}
    logging.info("start: %s workers=%d git=%s", args_cfg, workers, _git_hash())
    results: dict[str, list[dict]] = {c: [] for c in conds}
    if workers <= 1:
        _collect(map(run_task, tasks), results)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            _collect(ex.map(run_task, tasks), results)
    config = {"args": args_cfg, "git_hash": _git_hash(), "organism": OrganismConfig().to_dict(),
              "world": dataclasses.asdict(WorldConfig(**world_config_kwargs(
                  a.dims, a.roll_gain, a.escape_sign, a.side_wall_mm))),
              "seed_scheme": "neural.core.seed.derive_seed(seed, name), names: world/body/organism/random_motor"}
    rep = build_report(results, a.dims, a.duration_ms, a.dt_ms, seeds, config)
    if a.record:
        from .timeline import write_timeline
        rep["recording_dir"] = "recordings"
        write_timeline(out)
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(rep))
    logging.info("gotovo: %s", out)
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
