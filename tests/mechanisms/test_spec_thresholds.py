"""Po jedan test za svaki prag iz odeljka '## 10.' specifikacija (docs/specs/*.md).

Pragovi su provere kvalitativne verodostojnosti, ne poređenje sa eksperimentalnim podacima.
Izmerene vrednosti se ispisuju (pytest -s). Redosled testova prati docs/specs/README.md.
"""
import numpy as np
import pytest

from neural.core.seed import subsystem_rng
from neural.mechanisms import hh
from neural.mechanisms.cpg import oscillation_period, simulate_cpg
from neural.mechanisms.da_stdp import DASTDP
from neural.mechanisms.homeostasis import SynapticScaling
from neural.mechanisms.hh import detect_spikes, simulate_hh
from neural.mechanisms.stdp import STDP_PARAMS, PairSTDP, stdp_window
from neural.mechanisms.synapses import SYNAPSE_TYPES, Synapse, mg_block

RNG = subsystem_rng(0, "spec_thresholds")


# =========================================================================== HH
def _pulse(amp=40.0, t0=5.0, dur=1.0):
    return lambda t: amp if t0 <= t < t0 + dur else 0.0


def _ap(dt=0.025):
    r = simulate_hh(_pulse(), 30.0, dt)
    return r["t"], r["V"]


def _n_spikes(I, T, dt=0.025):
    r = simulate_hh(I, T, dt)
    return len(detect_spikes(r["t"], r["V"]))


def test_hh_rest_minus65():
    r = simulate_hh(0.0, 100.0)
    v = r["V"][-1]
    print(f"\nHH rest V = {v:.4f} mV")
    assert abs(v + 65.0) <= 0.5


def test_hh_peak_above_20mV():
    _, V = _ap()
    print(f"\nHH peak = {V.max():.2f} mV")
    assert V.max() > 20.0


def test_hh_amplitude_above_80mV():
    _, V = _ap()
    amp = V.max() - (-65.0)
    print(f"\nHH amplitude = {amp:.2f} mV")
    assert amp > 80.0


def test_hh_half_width_0p5_to_2p5_ms():
    t, V = _ap()
    half = (V.max() + (-65.0)) / 2
    width = detect_spikes(t, -V, -half)[0] - detect_spikes(t, V, half)[0]
    print(f"\nHH half-width = {width:.3f} ms (puls 40 uA/cm2, 1 ms)")
    assert 0.5 <= width <= 2.5


def test_hh_I1_no_spike():
    n = _n_spikes(1.0, 200.0)
    print(f"\nHH spikes at I=1: {n}")
    assert n == 0


def test_hh_I12_at_least_8_spikes_in_200ms():
    n = _n_spikes(12.0, 200.0)
    print(f"\nHH spikes at I=12 in 200 ms: {n}")
    assert n >= 8


def test_hh_fI_nondecreasing():
    f = {i: _n_spikes(float(i), 1000.0) for i in (12, 20, 35, 50)}
    print(f"\nHH f-I (spikes/1000 ms): {f}")
    seq = list(f.values())
    assert all(b >= a for a, b in zip(seq, seq[1:])), f


def test_hh_dt_consistency():
    def meas(dt):
        r = simulate_hh(20.0, 200.0, dt)
        sp = detect_spikes(r["t"], r["V"])
        return len(sp), float(sp[0]), float(r["V"].max())

    ref = meas(0.01)
    for dt in (0.01, 0.025, 0.05):
        n, t1, pk = meas(dt)
        print(f"\nHH dt={dt}: n={n} (ref {ref[0]}), first spike={t1:.3f} ms (ref {ref[1]:.3f}), "
              f"peak={pk:.2f} mV (ref {ref[2]:.2f})")
        assert abs(n - ref[0]) <= 1
        assert abs(t1 - ref[1]) <= 0.5
        assert abs(pk - ref[2]) <= 3.0


# ===================================================================== synapses
def _trace(kind, T=600.0, dt=0.005):
    s = Synapse(kind)
    st = s.on_spike(s.init_state(1, RNG), 0, 1.0)
    g = [st["g"][0]]
    for _ in range(int(T / dt)):
        st, _ = s.step(0.0, dt, st, {})
        g.append(st["g"][0])
    return np.array(g), dt


def test_synapse_peak_equals_g_max():
    for kind in SYNAPSE_TYPES:
        s = Synapse(kind, g_max_nS=2.0)
        st = s.on_spike(s.init_state(1, RNG), 0, 1.0)
        peak = 0.0
        for _ in range(int(600 / 0.005)):
            st, _ = s.step(0.0, 0.005, st, {})
            peak = max(peak, st["g"][0])
        print(f"\nsynapse {kind}: peak g = {peak:.5f} nS (g_max = 2)")
        assert peak == pytest.approx(2.0, rel=0.02)


def test_synapse_current_zero_at_reversal():
    for kind, p in SYNAPSE_TYPES.items():
        s = Synapse(kind)
        st = s.on_spike(s.init_state(1, RNG), 0, 1.0)
        st, _ = s.step(0.0, 1.0, st, {})
        i = float(np.ravel(s.current(st, p["E"]))[0])
        print(f"\nsynapse {kind}: I(E_rev={p['E']}) = {i:.3e} pA (g = {st['g'][0]:.4f} nS)")
        assert st["g"][0] > 0 and i == 0.0


def test_synapse_nmda_mg_block_formula_at_minus65():
    expected = 1.0 / (1.0 + np.exp(4.03) / 3.57)
    got = float(mg_block(-65.0))
    print(f"\nB(-65) = {got:.6f}, formula 1/(1+e^4.03/3.57) = {expected:.6f}")
    assert got == pytest.approx(expected, rel=1e-3)


def test_synapse_kinetics_order_nmda_gaba_ampa():
    t_peak, t_decay = {}, {}
    for kind in ("nmda", "gaba_a", "ampa"):
        g, dt = _trace(kind)
        ip = int(np.argmax(g))
        below = np.nonzero(g[ip:] <= g[ip] / np.e)[0]
        t_peak[kind] = ip * dt
        t_decay[kind] = below[0] * dt  # time from peak to 1/e of peak
    print(f"\ntime to peak (ms): {t_peak}\n1/e decay after peak (ms): {t_decay}")
    assert t_peak["nmda"] > t_peak["gaba_a"] > t_peak["ampa"]
    assert t_decay["nmda"] > t_decay["gaba_a"] > t_decay["ampa"]


# ======================================================================== STDP
def _run_pair(pre_t, post_t, dt=0.1, T=300.0, w0=0.5):
    m = PairSTDP(1, 1, w0=w0)
    s = m.init_state(1, RNG)
    pre_i = {int(round(x / dt)) for x in pre_t}
    post_i = {int(round(x / dt)) for x in post_t}
    for i in range(int(round(T / dt))):
        s, _ = m.step(i * dt, dt, s, {"pre_spikes": np.array([i in pre_i]), "post_spikes": np.array([i in post_i])})
    return s["w"][0, 0]


def test_stdp_sign_of_dw():
    pot = _run_pair([50.0], [60.0]) - 0.5
    dep = _run_pair([60.0], [50.0]) - 0.5
    print(f"\nSTDP dw(+10 ms) = {pot:.3e}, dw(-10 ms) = {dep:.3e}")
    assert pot > 0 > dep
    assert float(stdp_window(10.0)) > 0 > float(stdp_window(-10.0))


def test_stdp_tau_fit_within_10_percent():
    deltas = np.arange(2.0, 82.0, 4.0)
    pos = np.array([_run_pair([50.0], [50.0 + d]) - 0.5 for d in deltas])
    neg = np.array([-(_run_pair([50.0 + d], [50.0]) - 0.5) for d in deltas])
    tau_p = -1.0 / np.polyfit(deltas, np.log(pos), 1)[0]
    tau_m = -1.0 / np.polyfit(deltas, np.log(neg), 1)[0]
    print(f"\nSTDP fit tau+ = {tau_p:.3f} ms (ref {STDP_PARAMS['tau_plus']}), "
          f"tau- = {tau_m:.3f} ms (ref {STDP_PARAMS['tau_minus']})")
    assert tau_p == pytest.approx(STDP_PARAMS["tau_plus"], rel=0.10)
    assert tau_m == pytest.approx(STDP_PARAMS["tau_minus"], rel=0.10)


def test_stdp_online_equals_analytic():
    worst = 0.0
    for d in (-60.0, -25.0, -5.0, 5.0, 25.0, 60.0):
        dw = _run_pair([100.0], [100.0 + d], T=400.0) - 0.5
        ref = float(stdp_window(d))
        worst = max(worst, abs(dw - ref) / abs(ref))
    print(f"\nSTDP online vs analytic, max relative error = {worst:.2e}")
    assert worst < 1e-6


# ===================================================================== DA-STDP
def _run_da(pre_t, post_t, da_t=None, da=0.5, T=1500.0, dt=0.5, keep=None, **kw):
    m = DASTDP(1, 1, w0=1.0, **kw)
    s = m.init_state(1, RNG)
    pre_i = {int(round(x / dt)) for x in pre_t}
    post_i = {int(round(x / dt)) for x in post_t}
    da_i = None if da_t is None else int(round(da_t / dt))
    snaps = {}
    for i in range(int(round(T / dt))):
        s, _ = m.step(i * dt, dt, s, {"pre_spikes": np.array([i in pre_i]), "post_spikes": np.array([i in post_i]),
                                      "da": da if i == da_i else 0.0})
        if keep and i in keep:
            snaps[i] = float(s["c"][0, 0])
    return s, snaps


def test_da_stdp_no_da_no_change():
    s, _ = _run_da([10.0], [15.0], None)
    print(f"\nDA-STDP w without DA = {s['w'][0, 0]!r}, c = {s['c'][0, 0]:.4f}")
    assert s["c"][0, 0] != 0.0 and s["w"][0, 0] == 1.0
    s2, _ = _run_da([15.0], [10.0], None)
    assert s2["w"][0, 0] == 1.0


def test_da_stdp_eligibility_decays_to_1_over_e():
    dt, t_post = 0.5, 15.0
    i0, i1 = int(t_post / dt), int((t_post + 1000.0) / dt)
    _, snaps = _run_da([10.0], [t_post], None, T=1100.0, dt=dt, keep={i0, i1})
    ratio = snaps[i1] / snaps[i0]
    print(f"\nDA-STDP c(tau_c)/c0 = {ratio:.5f} (e^-1 = {np.exp(-1):.5f})")
    assert ratio == pytest.approx(np.exp(-1), rel=0.02)


def test_da_stdp_reward_sign_follows_order():
    pp, _ = _run_da([10.0], [15.0], 100.0)
    qp, _ = _run_da([15.0], [10.0], 100.0)
    print(f"\nDA-STDP dw pre->post = {pp['w'][0, 0] - 1:+.4f}, post->pre = {qp['w'][0, 0] - 1:+.4f}")
    assert pp["w"][0, 0] > 1.0 and qp["w"][0, 0] < 1.0


def test_da_stdp_delay_5s_less_than_10_percent_of_0p89s():
    # slab DA impuls, da * tau_d << w_max - w0, da se ne zasićuje na w_max
    def effect(delay_ms):
        s, _ = _run_da([10.0], [15.0], 15.0 + delay_ms, da=0.002, T=delay_ms + 2000.0)
        assert s["w"][0, 0] < DASTDP().p["w_max"]
        return s["w"][0, 0] - 1.0

    e_short, e_long = effect(890.0), effect(5000.0)
    print(f"\nDA-STDP dw(0.89 s) = {e_short:.5f}, dw(5 s) = {e_long:.5f}, ratio = {e_long / e_short:.4f}")
    assert e_short > 0 and e_long < 0.1 * e_short


# ================================================================== homeostasis
def _converge(w0, k, T_ms=300000.0, dt=10.0):
    sc = SynapticScaling(nu_target=5.0, tau_s=10000.0)
    w = np.array([w0], dtype=float)  # (n_post=1, n_pre)
    for _ in range(int(T_ms / dt)):
        rate = np.array([k * w.sum()])
        w = sc.step(dt, w, rate)
    return k * w.sum(), w[0]


def test_homeostasis_rate_converges_to_target_within_5_percent():
    k = 1.0
    for w0 in ([0.2, 0.4, 0.6], [4.0, 8.0, 12.0]):  # start: ~1 Hz (below) and ~24 Hz (above)
        start = k * sum(w0)
        r, _ = _converge(w0, k)
        print(f"\nhomeostasis r: {start:.2f} Hz -> {r:.4f} Hz (target 5 Hz)")
        assert r == pytest.approx(5.0, rel=0.05)


def test_homeostasis_weight_ratios_preserved_to_1e6():
    w0 = [0.2, 0.4, 0.6]
    _, w = _converge(w0, 1.0)
    dev = max(abs(w[1] / w[0] - 2.0), abs(w[2] / w[0] - 3.0))
    print(f"\nhomeostasis max ratio deviation = {dev:.2e}")
    assert dev < 1e-6


# ========================================================================== CPG
def _cpg():
    return simulate_cpg(5000.0)


def test_cpg_anticorrelation_below_minus_0p3():
    c = _cpg()
    r = float(np.corrcoef(c["y"][:, 0], c["y"][:, 1])[0, 1])
    print(f"\nCPG corr(y1, y2) = {r:.4f}")
    assert r < -0.3


def test_cpg_coactivity_below_20_percent():
    c = _cpg()
    both = float(np.mean((c["y"][:, 0] > 1e-6) & (c["y"][:, 1] > 1e-6)))
    print(f"\nCPG coactivity = {100 * both:.2f} % of time")
    assert both < 0.20


def test_cpg_at_least_3_cycles_in_5s():
    c = _cpg()
    out = c["out"]
    n_up = int(np.sum((out[:-1] < 0) & (out[1:] >= 0)))
    print(f"\nCPG upward zero crossings in 5 s = {n_up}, period = {oscillation_period(c['t'], out):.1f} ms")
    assert n_up >= 3


def test_cpg_control_a0_no_rhythm_std_below_1e3():
    c = simulate_cpg(5000.0, params={"a": 0.0})
    tail = c["out"][len(c["out"]) // 2:]
    print(f"\nCPG a=0: std(out) over last 2.5 s = {tail.std():.3e}")
    assert tail.std() < 1e-3
