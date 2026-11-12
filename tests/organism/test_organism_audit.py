"""Revizija organizma (faza 4). Vidi docs/specs/organism_audit.md."""
import hashlib
import math

import numpy as np
import pytest

from neural.organism import Organism
from neural.organism.organism import noise_scale

# sha256 od o._noise.tobytes() posle prvog step() za seed=0; izmereno pre popravki (RNG redosled)
NOISE_DIGEST_SEED0 = "e5ccd55eca1c75c4245a6f9dac7d576ffd4a048105b6592cbc8a2f49242cab07"


def run(org, n, s=None, dt=0.1):
    s = np.zeros(9) if s is None else s
    return np.array([org.step(s, dt) for _ in range(n)])


def test_noise_digest_unchanged():
    o = Organism(seed=0)
    o.step(np.zeros(9), 0.1)
    assert hashlib.sha256(o._noise.tobytes()).hexdigest() == NOISE_DIGEST_SEED0


def test_noise_scale_values_and_errors():
    assert noise_scale(0.1) == 1.0
    assert math.isclose(noise_scale(0.025), 2.0)
    for bad in (0.0, -0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            noise_scale(bad)


def test_autapses_zero_and_stay_zero():
    o = Organism(seed=1)
    We, Wi = o.weight_matrices()
    assert not np.any(np.diag(We)) and not np.any(np.diag(Wi))
    run(o, 20000, np.full(9, 0.3))
    o.reset()
    We, Wi = o.weight_matrices()
    assert not np.any(np.diag(We)) and not np.any(np.diag(Wi))


def test_weight_matrices_are_copies_and_signs():
    o = Organism(seed=0)
    We, Wi = o.weight_matrices()
    assert We.min() >= 0 and Wi.min() >= 0
    We[:] = 5.0
    assert not np.any(o.weight_matrices()[0] == 5.0)


def test_no_da_weights_frozen_without_phasic_or_tonic():
    o = Organism(seed=3, dopamine=False)
    for r in o.rules.values():
        assert r.p["d_tonic"] == 0.0
    w0 = o.plastic_weights()
    run(o, 5000, np.full(9, 0.3))
    assert np.array_equal(w0, o.plastic_weights())
    assert all(r.p["d_tonic"] > 0 for r in Organism(seed=3).rules.values())


def test_reset_restores_dynamic_state_keeps_weights():
    o = Organism(seed=2)
    V0 = o.V.copy()
    run(o, 3000, np.full(9, 0.3))
    w = o.plastic_weights()
    o.reset()
    assert np.array_equal(o.V, V0)
    assert np.array_equal(w, o.plastic_weights())
    assert not o.ge.any() and not o.gi.any() and not o._rate.any()
    assert o._k == 0 and o.da_phasic == 0.0 and o._first
    for key, st in o._rs.items():
        assert not st["c"].any() and not st["x_pre"].any() and not st["x_post"].any()
        assert np.array_equal(st["d"], np.full(1, o.rules[key].p["d_tonic"]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cpg_rhythm_antiphase_cocontraction(seed):
    o = Organism(seed=seed, plasticity=False)
    out = run(o, 30000)
    q = (out[:, 0] - out[:, 1])[::10]
    q = q - q.mean()
    f = np.fft.rfftfreq(len(q), d=1e-3)
    P = np.abs(np.fft.rfft(q)) ** 2
    P[0] = 0.0
    assert 0.5 <= f[P.argmax()] <= 2.0
    assert np.corrcoef(out[:, 0], out[:, 1])[0, 1] < 0
    assert np.corrcoef(out[:, 2], out[:, 3])[0, 1] < 0
    cci = np.minimum(out[:, 0], out[:, 1]).mean() / np.maximum(out[:, 0], out[:, 1]).mean()
    assert math.isfinite(cci) and 0.0 <= cci < 1.0
