import pathlib
import re

import numpy as np

from neural.organism import Organism

ORG_DIR = pathlib.Path(__file__).resolve().parents[2] / "neural" / "organism"


def run(org, n, s=None, dt=0.1):
    s = np.zeros(9) if s is None else s
    return np.array([org.step(s, dt) for _ in range(n)])


def test_shape_sign_finite():
    out = run(Organism(seed=1), 500, np.full(9, 0.3))
    assert out.shape == (500, 4) and np.all(out >= 0) and np.all(np.isfinite(out))


def test_determinism():
    s = np.linspace(-0.2, 0.5, 9)
    a, b = Organism(seed=5), Organism(seed=5)
    assert np.array_equal(run(a, 3000, s), run(b, 3000, s))
    assert np.array_equal(a.plastic_weights(), b.plastic_weights())
    assert not np.array_equal(run(Organism(seed=6), 3000, s), run(Organism(seed=5), 3000, s))


def test_frozen_keeps_weights_full_changes_them():
    s = np.full(9, 0.3)
    fz = Organism(seed=2, plasticity=False)
    w0 = fz.plastic_weights()
    run(fz, 20000, s)
    assert np.array_equal(w0, fz.plastic_weights())
    assert all(not np.any(st["c"]) for st in fz._rs.values())   # frozen ne zove rules[*].step
    fu = Organism(seed=2)
    run(fu, 20000, s)
    assert not np.array_equal(w0, fu.plastic_weights())
    assert any(np.any(st["c"] != 0) for st in fu._rs.values())   # tragovi podobnosti rade
    assert np.all(fu.plastic_weights() >= 0)


def test_no_da_phasic_zero_and_da_nonzero_on_change():
    nd = Organism(seed=3, dopamine=False)
    run(nd, 3000, np.full(9, 0.3))
    nd.step(np.full(9, 2.0), 0.1)
    assert nd.da_phasic == 0.0
    d = Organism(seed=3)
    run(d, 3000, np.full(9, 0.3))
    d.step(np.full(9, 2.0), 0.1)
    assert d.da_phasic > 0.0


def test_cpg_oscillates_without_input():
    o = Organism(seed=0, plasticity=False)
    out = run(o, 30000)           # 3 s
    q = (out[:, 0] - out[:, 1])[::10]
    q = q - q.mean()
    ac = np.correlate(q, q, "full")[len(q) - 1:]
    ac /= ac[0]
    lags = np.arange(len(ac))      # ms
    sel = (lags >= 500) & (lags <= 2000)   # 0.5-2 Hz
    assert ac[sel].max() > 0.1, ac[sel].max()


def test_no_forbidden_content_in_source():
    banned = [r"\bprobe\b", r"Probe", r"x_perp", r"escaped", r"ChannelWorld", r"neural\.environment",
              r"neural\.body", r"\btorch\b", r"tensorflow", r"\bjax\b", r"sklearn", r"keras",
              r"\bworld\b", r"\bbody\b", r"\benv\b"]
    for f in ORG_DIR.glob("*.py"):
        txt = f.read_text(encoding="utf-8")
        for b in banned:
            assert not re.search(b, txt), (f.name, b)


def test_constructor_params_clean():
    import inspect
    names = set(inspect.signature(Organism.__init__).parameters) - {"self"}
    assert names == {"config", "seed", "plasticity", "dopamine"}
    for k in vars(Organism(seed=0)):
        assert "world" not in k and "probe" not in k and "env" not in k
