import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import pytest
from neural.molecular.channels.hh import HH_PARAMS, alpha_m, alpha_n, steady_state, temperature_factor
from neural.cellular.hh_neuron import HHNeuron, simulate_hh
from neural.cellular.analysis import detect_spikes

ALLOWED = {'established', 'strongly_supported', 'incomplete', 'hypothesis', 'speculative'}


def _rate(I):
    t, V = simulate_hh(I, t_stop_ms=500.0, dt=0.01)
    s = np.asarray(detect_spikes(t, V, threshold=0.0))
    return float(np.sum(s > 200.0)) / 0.3


def test_metadata():
    assert HHNeuron.status == 'implemented'
    assert HHNeuron.reliability_class in ALLOWED


def test_original_parameters_modern_convention():
    p = HH_PARAMS
    for k, v in [('C_m', 1.0), ('g_Na', 120.0), ('g_K', 36.0), ('g_L', 0.3), ('E_Na', 50.0), ('E_K', -77.0), ('V_rest', -65.0)]:
        assert p[k] == pytest.approx(v)
    assert p['E_L'] == pytest.approx(-54.387, abs=1e-3)
    assert temperature_factor(6.3) == pytest.approx(1.0)
    assert temperature_factor(16.3) == pytest.approx(3.0)


def test_gates_at_rest_and_singularities():
    m, h, n = steady_state(-65.0)
    assert m == pytest.approx(0.0529, abs=5e-4)
    assert h == pytest.approx(0.5961, abs=5e-4)
    assert n == pytest.approx(0.3177, abs=5e-4)
    assert alpha_m(-40.0) == pytest.approx(1.0, rel=1e-6)
    assert alpha_n(-55.0) == pytest.approx(0.1, rel=1e-6)


def test_rest_is_stable():
    t, V = simulate_hh(0.0, t_stop_ms=100.0, dt=0.01)
    assert abs(V[-1] + 65.0) < 0.5
    assert len(detect_spikes(t, V, threshold=0.0)) == 0


def test_action_potential_shape():
    t, V = simulate_hh(10.0, t_stop_ms=200.0, dt=0.01)
    s = np.asarray(detect_spikes(t, V, threshold=0.0))
    assert len(s) >= 5
    late = t > 50.0
    vmax = V[late].max()
    vmin = V[late].min()
    assert 20.0 <= vmax <= 55.0
    assert -78.0 <= vmin <= -65.0
    w = (t > s[2] - 5.0) & (t < s[2] + 8.0)
    half = 0.5 * (vmax + vmin)
    above = t[w][V[w] > half]
    width = above.max() - above.min()
    assert 0.8 <= width <= 3.5


def test_fI_curve():
    f10 = _rate(10.0)
    assert 60.0 <= f10 <= 76.0
    f20 = _rate(20.0)
    f40 = _rate(40.0)
    assert f10 < f20 < f40


def test_type_II_onset():
    rates = [_rate(I) for I in (4.0, 5.0, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0)]
    assert rates[0] == 0.0
    first = next(r for r in rates if r > 0.0)
    assert first >= 40.0
