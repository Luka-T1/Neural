"""DASTDP bez dopamina: d ≡ 0 ⇒ w se ne menja, iako tragovi c rade.

Ugovor (specifikacija da_stdp, kontrolni uslov 'bez DA' u organizmu = d ≡ 0):
- pri d_tonic = 0 i da = 0 težina w ostaje tačno početna vrednost na svakom koraku,
  dok se eligibility trag c menja (STDP je aktivan i tagovanje se događa);
- pri d_tonic > 0 težina se menja (tonični DA je dovoljan da c pretvori u Δw).
"""
from __future__ import annotations

import numpy as np

from neural.core.seed import subsystem_rng
from neural.mechanisms.da_stdp import DASTDP

RNG = subsystem_rng(0, "test_da_stdp_no_da")
DT = 0.5  # ms
T = 1500.0  # ms


def _run(pre_ms, post_ms, *, da_pulses=None, dt=DT, T=T, **params):
    """Vraća (lista w po koraku, lista c po koraku, krajnje stanje) za 1x1 sinapsu."""
    m = DASTDP(1, 1, w0=1.0, **params)
    s = m.init_state(1, RNG)
    pre_i = {int(round(x / dt)) for x in pre_ms}
    post_i = {int(round(x / dt)) for x in post_ms}
    da_i = {} if da_pulses is None else {int(round(t / dt)): a for t, a in da_pulses}
    ws, cs = [], []
    for i in range(int(round(T / dt))):
        inputs = {"pre_spikes": np.array([i in pre_i]), "post_spikes": np.array([i in post_i]),
                  "da": float(da_i.get(i, 0.0))}
        s, _ = m.step(i * dt, dt, s, inputs)
        ws.append(float(s["w"][0, 0]))
        cs.append(float(s["c"][0, 0]))
    return ws, cs, s


def test_no_da_w_frozen_while_eligibility_trace_active():
    # pre pre post: STDP puni c, ali d_tonic = 0 i da = 0 ⇒ d ≡ 0 ⇒ w ≡ w0
    ws, cs, s = _run([10.0, 40.0], [15.0, 45.0], d_tonic=0.0)
    print(f"\nDASTDP bez DA: max|c| = {max(abs(c) for c in cs):.4f}, "
          f"min w = {min(ws)!r}, max w = {max(ws)!r}")
    assert max(abs(c) for c in cs) > 0.1, "trag c mora da radi u ovom testu"
    assert all(w == 1.0 for w in ws)
    assert s["w"][0, 0] == 1.0
    assert float(s["d"][0]) == 0.0


def test_no_da_w_frozen_for_reversed_timing_too():
    ws, cs, _ = _run([15.0], [10.0], d_tonic=0.0)
    assert max(abs(c) for c in cs) > 0.1
    assert all(w == 1.0 for w in ws)


def test_tonic_da_changes_w_with_same_spikes():
    ws, cs, s = _run([10.0], [15.0], d_tonic=0.5)
    print(f"\nDASTDP d_tonic=0.5: w_final = {ws[-1]:.6f}, max|c| = {max(abs(c) for c in cs):.4f}")
    assert max(abs(c) for c in cs) > 0.1
    assert ws[-1] > 1.0, "pre→post uz tonični DA mora da potencira"
    assert s["w"][0, 0] != 1.0


def test_phasic_da_pulse_changes_w_when_tonic_zero():
    # d_tonic = 0 bez pulsa ⇒ bez promene; sa pulsom posle pre→post ⇒ promena
    ws0, _, _ = _run([10.0], [15.0], d_tonic=0.0)
    ws1, _, _ = _run([10.0], [15.0], d_tonic=0.0, da_pulses=[(100.0, 0.5)])
    assert all(w == 1.0 for w in ws0)
    assert ws1[-1] > 1.0
