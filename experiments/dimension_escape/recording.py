"""Snimanje aktivnosti organizma (Deo 2): spajkovi po koraku i uzorci sa istom frekvencijom kao trace.

Recorder je samo posmatrač: ne menja stanje organizma i ne troši RNG.
"""
from __future__ import annotations

import numpy as np

POPS = ("sens", "exc", "inh", "cpg", "cpg_inh")
_POP_ATTRS = ("sl_s", "sl_e", "sl_i", "sl_c", "sl_ci")
RULE_KEYS = ("sens_exc", "exc_cpg", "cross")


def _org_readouts(org) -> dict:
    """Očitavanja d, novelty i eligibility.

    read-only; Organism nema javni accessor (neural/organism van grane)."""
    out = {"d": float(org._rs["cross"]["d"][0]), "novelty_bar": float(org._nbar)}
    for key in RULE_KEYS:
        mask = org._blocks[key][2] > 0
        out[f"elig_{key}"] = float(np.mean(np.asarray(org._rs[key]["c"])[mask]))
    return out


class Recorder:
    def __init__(self, org):
        self.has_org = org is not None
        self._spike_t: list[np.ndarray] = []
        self._spike_i: list[np.ndarray] = []
        self._t: list[float] = []
        self._pm: list[np.ndarray] = []
        self._cmd: list[np.ndarray] = []
        self._da: list[float] = []
        self._ro: dict[str, list[float]] = {k: [] for k in ("d", "novelty_bar", *[f"elig_{r}" for r in RULE_KEYS])}
        self._bounds = None
        if org is not None:
            self._bounds = np.array([[getattr(org, a).start, getattr(org, a).stop] for a in _POP_ATTRS], dtype=int)

    def on_step(self, t_ms: float, org) -> None:
        if org is None:
            return
        idx = np.flatnonzero(org.last_spikes)
        if idx.size:
            self._spike_i.append(idx.astype(np.int64))
            self._spike_t.append(np.full(idx.size, float(t_ms)))

    def on_sample(self, t_ms: float, premotor, commands, da_phasic, org) -> None:
        self._t.append(float(t_ms))
        self._pm.append(np.array(premotor, dtype=float))
        self._cmd.append(np.array(commands, dtype=float))
        self._da.append(float(da_phasic))
        ro = _org_readouts(org) if org is not None else {}
        for k, lst in self._ro.items():
            lst.append(ro.get(k, float("nan")))

    def save(self, path: str) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        st = np.concatenate(self._spike_t) if self._spike_t else np.zeros(0)
        si = np.concatenate(self._spike_i) if self._spike_i else np.zeros(0, dtype=np.int64)
        bounds = self._bounds if self._bounds is not None else np.full((5, 2), -1, dtype=int)
        S = len(self._t)
        pm = np.array(self._pm, dtype=float).reshape(S, 4) if S else np.zeros((0, 4))
        cmd = np.array(self._cmd, dtype=float)
        if not S:
            cmd = np.zeros((0, 4))
        np.savez_compressed(
            path, t=np.array(self._t, dtype=float), spike_t_ms=st.astype(float), spike_idx=si.astype(np.int64),
            pop_names=np.array(POPS, dtype="U16"), pop_bounds=bounds,
            da_phasic=np.array(self._da, dtype=float),
            d=np.array(self._ro["d"], dtype=float), novelty_bar=np.array(self._ro["novelty_bar"], dtype=float),
            elig_sens_exc=np.array(self._ro["elig_sens_exc"], dtype=float),
            elig_exc_cpg=np.array(self._ro["elig_exc_cpg"], dtype=float),
            elig_cross=np.array(self._ro["elig_cross"], dtype=float), premotor=pm, commands=cmd)
        return path
