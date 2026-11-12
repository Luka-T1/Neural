import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
import numpy as np
import pytest
from neural.network.plasticity.tsodyks_markram import TsodyksMarkramSynapse
from neural.network.plasticity.stdp import PairSTDP, stdp_window

ALLOWED = {'established', 'strongly_supported', 'incomplete', 'hypothesis', 'speculative'}


def test_metadata():
    for cls in (TsodyksMarkramSynapse, PairSTDP):
        assert cls.status == 'implemented'
        assert cls.reliability_class in ALLOWED


def test_tm_depression_paired_pulse():
    s = TsodyksMarkramSynapse(U=0.5, tau_rec_ms=800.0, tau_facil_ms=0.0, A=1.0)
    s.reset()
    a1 = s.on_spike(0.0)
    a2 = s.on_spike(50.0)
    assert a1 == pytest.approx(0.5)
    assert a2 / a1 == pytest.approx(1.0 - 0.5 * np.exp(-50.0 / 800.0), rel=1e-9)


def test_tm_facilitation_paired_pulse():
    U, tr, tf, d = 0.1, 100.0, 1000.0, 20.0
    s = TsodyksMarkramSynapse(U=U, tau_rec_ms=tr, tau_facil_ms=tf, A=1.0)
    s.reset()
    a1 = s.on_spike(0.0)
    a2 = s.on_spike(d)
    ef = np.exp(-d / tf)
    u2 = U * ef + U * (1.0 - U * ef)
    R2 = 1.0 - U * np.exp(-d / tr)
    assert a1 == pytest.approx(U)
    assert a2 / a1 == pytest.approx(u2 * R2 / U, rel=1e-9)
    assert a2 / a1 > 1.5


def _train(rate_hz, n=200):
    s = TsodyksMarkramSynapse(U=0.5, tau_rec_ms=800.0, tau_facil_ms=0.0, A=1.0)
    s.reset()
    d = 1000.0 / rate_hz
    return [s.on_spike(k * d) for k in range(n)]


def test_tm_steady_state_and_1_over_f():
    for r in (20.0, 40.0):
        e = np.exp(-(1000.0 / r) / 800.0)
        Rss = (1.0 - e) / (1.0 - 0.5 * e)
        assert _train(r)[-1] == pytest.approx(0.5 * Rss, rel=1e-6)
    a20 = _train(20.0)[-1] * 20.0
    a40 = _train(40.0)[-1] * 40.0
    assert a40 == pytest.approx(a20, rel=0.10)


def test_stdp_window():
    Ap, Am, tp, tm = 0.005, 0.00525, 20.0, 20.0
    assert stdp_window(10.0, Ap, Am, tp, tm) == pytest.approx(Ap * np.exp(-0.5))
    assert stdp_window(-10.0, Ap, Am, tp, tm) == pytest.approx(-Am * np.exp(-0.5))
    assert stdp_window(0.0, Ap, Am, tp, tm) == 0.0
    x = np.linspace(-200.0, 200.0, 40001)
    integral = float(np.sum(np.asarray(stdp_window(x, Ap, Am, tp, tm))) * (x[1] - x[0]))
    assert integral < 0.0


def test_song2000_defaults_and_bi_poo_preset():
    s = PairSTDP()
    assert s.A_plus == pytest.approx(0.005)
    assert s.A_minus / s.A_plus == pytest.approx(1.05)
    assert s.tau_plus_ms == pytest.approx(20.0)
    assert s.tau_minus_ms == pytest.approx(20.0)
    b = PairSTDP.bi_poo_1998()
    assert b.tau_plus_ms == pytest.approx(16.8)
    assert b.tau_minus_ms == pytest.approx(33.7)


def test_pair_ltp_and_ltd():
    s = PairSTDP(w0=0.5, w_max=1.0)
    s.on_pre(0.0)
    s.on_post(10.0)
    assert s.w == pytest.approx(0.5 + 0.005 * np.exp(-0.5), rel=1e-9)
    s = PairSTDP(w0=0.5, w_max=1.0)
    s.on_post(0.0)
    s.on_pre(10.0)
    assert s.w == pytest.approx(0.5 - 0.00525 * np.exp(-0.5), rel=1e-9)


def test_hard_bounds():
    s = PairSTDP(A_plus=0.5, A_minus=0.5, w0=0.9, w_min=0.0, w_max=1.0)
    for k in range(10):
        s.on_pre(100.0 * k)
        s.on_post(100.0 * k + 1.0)
    assert s.w == pytest.approx(1.0)
    for k in range(10, 20):
        s.on_post(100.0 * k)
        s.on_pre(100.0 * k + 1.0)
    assert s.w == pytest.approx(0.0)
