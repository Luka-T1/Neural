import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import pytest
from neural.network.synapses.kinetics import DoubleExpSynapse, mg_block, peak_time, ampa, nmda, gaba_a

ALLOWED = {'established', 'strongly_supported', 'incomplete', 'hypothesis', 'speculative'}
CASES = [(ampa, 0.2, 2.0, 0.0), (nmda, 2.0, 100.0, 0.0), (gaba_a, 0.5, 5.0, -70.0)]


def test_metadata():
    assert DoubleExpSynapse.status == 'implemented'
    assert DoubleExpSynapse.reliability_class in ALLOWED


def test_mg_block_jahr_stevens_1990():
    assert mg_block(-65.0, 1.0) == pytest.approx(1.0 / (1.0 + np.exp(0.062 * 65.0) / 3.57), rel=1e-9)
    assert mg_block(-65.0, 1.0) == pytest.approx(0.0597, abs=1e-3)
    assert mg_block(0.0, 1.0) == pytest.approx(0.7812, abs=1e-3)
    assert mg_block(-65.0, 0.0) == pytest.approx(1.0)
    b = np.asarray(mg_block(np.linspace(-100.0, 40.0, 50), 1.0))
    assert np.all(np.diff(b) > 0.0)


def test_peak_time_formula():
    assert peak_time(0.2, 2.0) == pytest.approx(0.5117, abs=1e-3)
    assert peak_time(2.0, 100.0) == pytest.approx(7.984, abs=1e-2)


@pytest.mark.parametrize('factory,tr,td,E', CASES)
def test_defaults(factory, tr, td, E):
    s = factory()
    assert s.tau_rise_ms == pytest.approx(tr)
    assert s.tau_decay_ms == pytest.approx(td)
    assert s.E_rev_mV == pytest.approx(E)


@pytest.mark.parametrize('factory,tr,td,E', CASES)
def test_double_exponential_kinetics(factory, tr, td, E):
    s = factory(g_max_nS=1.0)
    s.reset()
    s.on_spike(1.0)
    dt = tr / 40.0
    n = int(round(4.0 * td / dt))
    g = np.array([s.step(dt) for _ in range(n)])
    t = dt * np.arange(1, n + 1)
    assert g.max() == pytest.approx(1.0, rel=0.01)
    assert t[np.argmax(g)] == pytest.approx(peak_time(tr, td), abs=2.0 * dt)
    tail = (t > peak_time(tr, td) + 8.0 * tr) & (t < 3.0 * td)
    slope = np.polyfit(t[tail], np.log(g[tail]), 1)[0]
    assert -1.0 / slope == pytest.approx(td, rel=0.03)


def test_linear_summation_and_nmda_current():
    s = nmda(g_max_nS=1.0)
    s.reset()
    s.on_spike(1.0)
    s.on_spike(1.0)
    g = max(s.step(0.05) for _ in range(400))
    assert g == pytest.approx(2.0, rel=0.01)
    I = s.current(-65.0)
    assert I == pytest.approx(s.g * mg_block(-65.0, 1.0) * (-65.0 - 0.0), rel=1e-9)
    assert I < 0.0


def test_gaba_current_sign():
    s = gaba_a(g_max_nS=1.0)
    s.reset()
    s.on_spike(1.0)
    for _ in range(100):
        s.step(0.0125)
    assert s.current(-50.0) > 0.0
    assert s.current(-80.0) < 0.0
