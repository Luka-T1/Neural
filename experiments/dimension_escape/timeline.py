"""Vremenska linija bekstva iz rezultata run_one ili iz traces/*.npz (Deo 2)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .harness import CONDITIONS


def _get(trace, key):
    if key in trace:
        return trace[key]
    return None


def build_timeline(trace, cond: str, seed: int) -> dict:
    t = np.asarray(trace["t"], dtype=float).ravel()
    x = np.asarray(trace["x"], dtype=float)
    x = x.reshape(len(t), -1)
    escaped = bool(np.asarray(trace["escaped"]))
    te = trace["t_escape_ms"]
    te = None if (te is None or not escaped) else float(te)
    da = np.asarray(trace["da"], dtype=float).ravel()
    cmd = np.asarray(trace["cmd"], dtype=float).reshape(len(t), -1)
    ev = [float(t[k]) for k in range(len(t)) if da[k] > 0 and (k == 0 or da[k - 1] <= 0)]
    tl = {"cond": cond, "seed": int(seed), "escaped": escaped, "t_escape_ms": te,
          "t_ms": t.tolist(), "x_perp": np.linalg.norm(x[:, 1:], axis=1).tolist(),
          "da_events_ms": ev, "q_cpg": (cmd[:, 0] - cmd[:, 1]).tolist()}
    xpn = _get(trace, "x_perp_n")
    if xpn is not None:
        tl["x_perp_n"] = np.asarray(xpn, dtype=float).ravel().tolist()
    return tl


def timeline_from_dir(out_dir: str) -> dict:
    tdir = os.path.join(out_dir, "traces")
    runs = []
    if os.path.isdir(tdir):
        for ci, cond in enumerate(CONDITIONS):
            for fn in os.listdir(tdir):
                pre = f"{cond}_seed"
                if fn.startswith(pre) and fn.endswith(".npz") and fn[len(pre):-4].isdigit():
                    seed = int(fn[len(pre):-4])
                    with np.load(os.path.join(tdir, fn)) as z:
                        tr = {k: z[k] for k in z.files}
                    runs.append((ci, seed, build_timeline(tr, cond, seed)))
    runs.sort(key=lambda r: (r[0], r[1]))
    return {"runs": [r[2] for r in runs]}


def write_timeline(out_dir: str) -> str:
    path = os.path.join(out_dir, "timeline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline_from_dir(out_dir), f)
    return path


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="experiments.dimension_escape.timeline")
    ap.add_argument("out_dir")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(argv)
    p = write_timeline(a.out_dir)
    print(p)
    if a.plot:
        from .visualize import plot_timeline
        plot_timeline(json.load(open(p, encoding="utf-8")), os.path.join(a.out_dir, "timeline.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
