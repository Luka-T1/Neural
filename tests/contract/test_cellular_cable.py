import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import pytest
from neural.cellular.multicompartment import Cable

ALLOWED = {'established', 'strongly_supported', 'incomplete', 'hypothesis', 'speculative'}


def test_metadata():
    assert Cable.status == 'implemented'
    assert Cable.reliability_class in ALLOWED


def test_isopotential_charging_time_constant():
    Rm = 20000.0
    c = Cable(n_comp=1, length_um=100.0, diam_um=10.0, Ra_ohm_cm=100.0, cm_uF_cm2=1.0, active='passive', g_leak_mS_cm2=1000.0 / Rm, E_leak_mV=-65.0)
    c.reset()
    c.inject(0, 0.01)
    t, V = c.simulate(t_stop_ms=200.0, dt=0.01)
    dv = V[:, 0] + 65.0
    tau_ms = Rm * 1.0e-6 * 1.0e3
    k = int(round(tau_ms / 0.01))
    assert t[k] == pytest.approx(tau_ms, abs=0.02)
    assert dv[k] / dv[-1] == pytest.approx(1.0 - np.exp(-1.0), rel=0.02)


def test_passive_sealed_cable_profile_and_input_resistance():
    d_um, L_um, Ra, Rm = 2.0, 1000.0, 100.0, 20000.0
    c = Cable(n_comp=100, length_um=L_um, diam_um=d_um, Ra_ohm_cm=Ra, cm_uF_cm2=1.0, active='passive', g_leak_mS_cm2=1000.0 / Rm, E_leak_mV=-65.0)
    c.reset()
    c.inject(0, 0.1)
    t, V = c.simulate(t_stop_ms=300.0, dt=0.1)
    dv = V[-1] + 65.0
    lam = np.sqrt(Rm * d_um * 1e-4 / (4.0 * Ra))
    x = np.asarray(c.x_um) * 1e-4
    L = L_um * 1e-4
    shape = np.cosh((L - x) / lam) / np.cosh((L - x[0]) / lam)
    np.testing.assert_allclose(dv / dv[0], shape, rtol=0.02)
    ra = 4.0 * Ra / (np.pi * (d_um * 1e-4) ** 2)
    rin_theory = ra * lam / np.tanh(L / lam)
    rin = dv[0] * 1e-3 / 0.1e-9
    assert rin == pytest.approx(rin_theory, rel=0.03)


def test_hh_axon_conduction_velocity_18_5C():
    c = Cable(n_comp=250, length_um=50000.0, diam_um=476.0, Ra_ohm_cm=35.4, cm_uF_cm2=1.0, active='hh', temperature_celsius=18.5)
    c.reset()
    c.inject(0, 10000.0, t_start_ms=0.0, t_stop_ms=0.5)
    t, V = c.simulate(t_stop_ms=6.0, dt=0.01)
    xs = np.asarray(c.x_um)
    i1 = int(np.argmin(np.abs(xs - 15000.0)))
    i2 = int(np.argmin(np.abs(xs - 35000.0)))
    assert V[:, i1].max() > 0.0
    assert V[:, i2].max() > 0.0
    t1 = t[np.argmax(V[:, i1] > 0.0)]
    t2 = t[np.argmax(V[:, i2] > 0.0)]
    v = (xs[i2] - xs[i1]) * 1e-6 / ((t2 - t1) * 1e-3)
    assert 15.0 <= v <= 23.0
