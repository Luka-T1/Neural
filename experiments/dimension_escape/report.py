"""Agregacija: report.json i report.md iz rezultata pojedinačnih (uslov, seed) pokretanja."""
from __future__ import annotations

import numpy as np

from .stats import fisher_exact_2x2, holm, mann_whitney_u, spearman, wilson_interval

CONTROLS = ("random_motor", "frozen", "no_da")


def _leak(runs) -> dict:
    d_all, vperp, vax = [], [], []
    for r in runs:
        if len(r["t"]) < 3 or not np.any(r["da"] > 0):
            continue
        x = r["x"]
        dt = np.gradient(r["t"]) / 1000.0
        vp = np.linalg.norm(np.gradient(x[:, 1:], axis=0), axis=1) / dt
        v0 = np.abs(np.gradient(x[:, 0])) / dt
        d_all.append(r["da"])
        vperp.append(vp)
        vax.append(v0)
    if not d_all:
        return {"n_samples": 0, "spearman_da_vs_abs_dxperp": None, "spearman_da_vs_abs_dx0": None}
    d, vp, v0 = map(np.concatenate, (d_all, vperp, vax))
    sp = spearman(d, vp)
    s0 = spearman(d, v0)
    f = lambda v: None if v != v else v
    return {"n_samples": int(len(d)), "spearman_da_vs_abs_dxperp": f(sp), "spearman_da_vs_abs_dx0": f(s0)}


def _roll_summary(results: dict) -> dict:
    """φ(T) po uslovu; φ dolazi isključivo iz probe().snapshot() (harness ga čuva u run['phi'])."""
    out: dict[str, dict] = {}
    for cond, runs in results.items():
        fin = [float(r["phi"][-1]) for r in runs if len(r.get("phi", ())) > 0]
        if fin:
            out[cond] = {"n": len(fin), "phi_final_mean_rad": float(np.mean(fin)),
                         "phi_final_abs_mean_rad": float(np.mean(np.abs(fin))),
                         "phi_final_abs_max_rad": float(np.max(np.abs(fin)))}
    return out


def _escape_side_summary(results: dict) -> dict:
    """Strana bekstva po uslovu (faza 4); x_perp_n dolazi iz probe snapshot-a (run['x_perp_n'])."""
    runs_all = [r for runs in results.values() for r in runs if "escape_sign" in r]
    if not runs_all:
        return {}
    sign, wall = int(runs_all[0]["escape_sign"]), float(runs_all[0]["side_wall_mm"])
    for r in runs_all:
        if int(r["escape_sign"]) != sign or float(r["side_wall_mm"]) != wall:
            raise ValueError("escape_sign/side_wall_mm se razlikuju među runovima")
    conds: dict[str, dict] = {}
    for cond, runs in results.items():
        runs = sorted((r for r in runs if "escape_sign" in r and len(r.get("x_perp_n", ())) > 0),
                      key=lambda r: r["seed"])
        if not runs:
            continue
        per_seed = [{"seed": r["seed"],
                     "escape_side": int(np.sign(r["x_perp_n"][-1])) if r["escaped"] else 0,
                     "x_perp_n_final": float(r["x_perp_n"][-1])} for r in runs]
        conds[cond] = {
            "n": len(runs), "escaped": sum(1 for r in runs if r["escaped"]),
            "escaped_toward_sign": sum(1 for p in per_seed if p["escape_side"] == sign),
            "escaped_against_sign": sum(1 for p in per_seed if p["escape_side"] == -sign),
            "x_perp_n_final_mean": float(np.mean([p["x_perp_n_final"] for p in per_seed])),
            "per_seed": per_seed}
    return {"escape_sign": sign, "side_wall_mm": wall, "conditions": conds}


def build_report(results: dict, dims: int, duration_ms: float, dt_ms: float, seeds, config: dict) -> dict:
    conds: dict[str, dict] = {}
    t_unc: dict[str, list] = {}
    for cond, runs in results.items():
        runs = sorted(runs, key=lambda r: r["seed"])
        n = len(runs)
        k = sum(1 for r in runs if r["escaped"])
        te = [r["t_escape_ms"] for r in runs if r["escaped"]]
        conds[cond] = {
            "n": n, "escaped": k, "rate": k / n if n else 0.0,
            "wilson95": list(wilson_interval(k, n)),
            "t_escape_median_ms": float(np.median(te)) if te else None,
            "t_escape_iqr_ms": [float(np.percentile(te, 25)), float(np.percentile(te, 75))] if te else None,
            "per_seed": [{"seed": r["seed"], "escaped": bool(r["escaped"]),
                          "t_escape_ms": r["t_escape_ms"]} for r in runs],
        }
        t_unc[cond] = [r["t_escape_ms"] if r["escaped"] else duration_ms + 1.0 for r in runs]
    comps: list[dict] = []
    if "full" in conds:
        for c in CONTROLS:
            if c not in conds:
                continue
            a, b = conds["full"], conds[c]
            pf = fisher_exact_2x2(int(a["escaped"]), int(a["n"]) - int(a["escaped"]),
                                  int(b["escaped"]), int(b["n"]) - int(b["escaped"]))
            _, pm = mann_whitney_u(t_unc["full"], t_unc[c])
            comps.append({"a": "full", "b": c, "fisher_p": pf, "mwu_p": pm})
        # Holm preko Fisherovih p-vrednosti (primarna mera, 3 poređenja)
        adj = holm([cmp["fisher_p"] for cmp in comps])
        for cmp, p in zip(comps, adj):
            cmp["p_holm"] = p
    audit = {c: _leak(runs) for c, runs in results.items() if c in ("full", "frozen")}
    sec: dict[str, dict] = {}
    for cond, runs in results.items():
        vals = {m: [r["secondary"].get(m) for r in runs if r["secondary"].get(m) is not None]
                for m in ("plv", "mean_sin_dphi", "mean_abs_area_rate")}
        sec[cond] = {m: (float(np.mean(v)) if v else None) for m, v in vals.items()}
        wc = [(r["w_cross"][0], r["w_cross"][-1]) for r in runs if len(r["w_cross"])]
        sec[cond]["w_cross_start_mean"] = float(np.mean([a for a, _ in wc])) if wc else None
        sec[cond]["w_cross_end_mean"] = float(np.mean([b for _, b in wc])) if wc else None
    rep = {"dims": dims, "duration_ms": duration_ms, "dt_ms": dt_ms, "seeds": list(seeds),
           "conditions": conds, "comparisons": comps, "da_leakage_audit": audit,
           "secondary": sec, "config": config}
    roll = _roll_summary(results)
    if roll:  # samo uz roll_gain != 0; pri 0 izlaz ostaje isti kao u fazama 1 i 2
        rep["roll"] = roll
    es = _escape_side_summary(results)
    if es:  # samo uz escape_sign != 0 (faza 4)
        rep["escape_side"] = es
    return rep


def _fmt(v, nd=3):
    return "-" if v is None else f"{v:.{nd}g}"


def render_markdown(rep: dict) -> str:
    L = [f"# Dimension escape: dims={rep['dims']}, T={rep['duration_ms']} ms, dt={rep['dt_ms']} ms, "
         f"N seedova={len(rep['seeds'])}", "",
         "| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |", "|---|---|---|---|---|---|---|"]
    for c, d in rep["conditions"].items():
        iqr = "-" if d["t_escape_iqr_ms"] is None else f"{d['t_escape_iqr_ms'][0]:.0f}–{d['t_escape_iqr_ms'][1]:.0f}"
        L.append(f"| {c} | {d['n']} | {d['escaped']} | {d['rate']:.2f} | "
                 f"{d['wilson95'][0]:.2f}–{d['wilson95'][1]:.2f} | {_fmt(d['t_escape_median_ms'])} | {iqr} |")
    L += ["", "## Poređenja (full vs kontrola)", "", "| kontrola | Fisher p | MWU p | Holm p |", "|---|---|---|---|"]
    for c in rep["comparisons"]:
        L.append(f"| {c['b']} | {c['fisher_p']:.4g} | {c['mwu_p']:.4g} | {c['p_holm']:.4g} |")
    if not rep["comparisons"]:
        L.append("| (nema: nije pokrenut full ili kontrole) | | | |")
    L += ["", "## DA leakage audit (Spearman)", "", "| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |", "|---|---|---|---|"]
    for c, d in rep["da_leakage_audit"].items():
        L.append(f"| {c} | {d['n_samples']} | {_fmt(d['spearman_da_vs_abs_dxperp'])} | {_fmt(d['spearman_da_vs_abs_dx0'])} |")
    L += ["", "## Sekundarne metrike (srednje po seedu)", "",
          "| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |", "|---|---|---|---|---|---|"]
    for c, d in rep["secondary"].items():
        L.append(f"| {c} | {_fmt(d['plv'])} | {_fmt(d['mean_sin_dphi'])} | {_fmt(d['mean_abs_area_rate'])} | "
                 f"{_fmt(d['w_cross_start_mean'])} | {_fmt(d['w_cross_end_mean'])} |")
    if rep.get("roll"):
        L += ["", "## Roll φ (iz probe snapshot-a; nije u Observation)", "",
              "| uslov | n | srednje φ(T) (rad) | srednje |φ(T)| (rad) | max |φ(T)| (rad) |", "|---|---|---|---|---|"]
        for c, d in rep["roll"].items():
            L.append(f"| {c} | {d['n']} | {_fmt(d['phi_final_mean_rad'])} | {_fmt(d['phi_final_abs_mean_rad'])} | "
                     f"{_fmt(d['phi_final_abs_max_rad'])} |")
    if rep.get("escape_side"):
        L += ["", "## Strana bekstva (escape_sign=±1, zid na −s strani)", "",
              "| uslov | n | bekstava | ka s | protiv s | srednje x⊥·n(T) |", "|---|---|---|---|---|---|"]
        for c, d in rep["escape_side"]["conditions"].items():
            L.append(f"| {c} | {d['n']} | {d['escaped']} | {d['escaped_toward_sign']} | "
                     f"{d['escaped_against_sign']} | {_fmt(d['x_perp_n_final_mean'])} |")
    return "\n".join(L) + "\n"
