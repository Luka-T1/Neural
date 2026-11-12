import numpy as np
import pytest

from neural.mechanisms import hh
from neural.mechanisms.hh import detect_spikes, nA_to_density, simulate_hh, steady_state


def n_spikes(I, T, dt=0.025, **kw):
    r = simulate_hh(I, T, dt, **kw)
    return len(detect_spikes(r["t"], r["V"]))


def test_singularities_and_rest():
    am, *_ = hh.rates(-40.0)
    assert am == pytest.approx(1.0)
    assert hh.rates(-55.0)[4] == pytest.approx(0.1)
    assert hh.rates(-40.0 + 1e-6)[0] == pytest.approx(1.0, abs=1e-4)
    m, h, n = steady_state(-65.0)
    assert (m, h, n) == pytest.approx((0.0529, 0.5961, 0.3177), abs=2e-3)
    r = simulate_hh(0.0, 50.0)
    assert abs(r["V"][-1] - r["V"][0]) < 0.1  # rest is (nearly) a fixed point of the shifted HH


def test_nA_to_density():
    assert nA_to_density(1.0, 1e-3) == pytest.approx(1.0)


def test_rheobase():
    lo, hi = 0.0, 20.0
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if n_spikes(mid, 200.0) > 0 else (mid, hi)
    print(f"\nRHEOBASE_uA_cm2 = {hi:.3f} (200 ms step, dt=0.025, rk4)")
    assert 1 < hi < 10


def test_fI_curve():
    I = [6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 100]
    f = {i: n_spikes(i, 1000.0) for i in I}
    print("\nF_I_Hz (1000 ms window):", f)
    seq = [f[i] for i in I if 12 <= i <= 50]
    assert all(b >= a for a, b in zip(seq, seq[1:])), seq
    # onset frequency at the lowest suprathreshold current is only recorded (type II: not asserted)


@pytest.mark.parametrize("dt", [0.01, 0.025, 0.05])
def test_exp_euler_vs_rk4(dt):
    a, b = n_spikes(20.0, 200.0, dt), n_spikes(20.0, 200.0, dt, method="exp_euler")
    assert abs(a - b) <= 1


@pytest.mark.parametrize("dt", [0.01, 0.025, 0.05])
def test_action_potential_waveform(dt):
    r = simulate_hh(lambda t: 20.0 if 5.0 <= t < 6.0 else 0.0, 30.0, dt)
    t, V = r["t"], r["V"]
    peak, trough = V.max(), V.min()
    assert 30 < peak < 55 and trough < -70
    half = (peak + (-65.0)) / 2
    up = detect_spikes(t, V, half)
    down = detect_spikes(t, -V, -half)
    width = down[0] - up[0]
    print(f"\nAP dt={dt}: peak={peak:.2f} mV, trough={trough:.2f} mV, half-width={width:.3f} ms")
    assert 0.3 < width < 3.0


def test_ode_adapter_matches_function():
    from neural.core.config import Config
    from neural.core.simulator import Simulator

    s = Simulator(Config.from_dict({"seed": 0}))
    s.add_mechanism(hh.HHNeuron(), 1)
    ev = []
    s.run(50.0)  # at rest: no spikes, V stays
    assert abs(s.state.mech["hh_neuron"]["V"][0] + 65.0) < 0.1
    s2 = Simulator(Config.from_dict({"seed": 0}))
    s2.add_mechanism(hh.HHNeuron(), 1, method="exp_euler")
    s2.add_mechanism(type("Src", (), dict(name="src", timescale_ms=1e3, confidence="hypothesis",
                                          params_meta={}, init_state=lambda self, n, rng: {"I": np.full(n, 20.0)},
                                          step=lambda self, t, dt, s, i: (s, []),
                                          outputs=lambda self, s: s))(), 1)
    s2.connect("src", "I", "hh_neuron", "I_ext_nA")
    s2.run(100.0, recorder=lambda st, e: ev.extend(e))
    assert len(ev) > 2
