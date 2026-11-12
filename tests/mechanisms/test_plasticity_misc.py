import numpy as np
import pytest

from neural.core.interfaces import SpikeEvent
from neural.core.registry import REGISTRY, validate_mechanism
from neural.core.seed import subsystem_rng
from neural.mechanisms import cpg, da_stdp, homeostasis, hh, stdp, synapses
from neural.mechanisms.cpg import oscillation_period, simulate_cpg
from neural.mechanisms.da_stdp import DASTDP
from neural.mechanisms.homeostasis import RateEstimator, SynapticScaling
from neural.mechanisms.stdp import PairSTDP, stdp_window
from neural.mechanisms.synapses import AMPASynapse, NMDASynapse, GABAASynapse, mg_block, norm_factor

RNG = subsystem_rng(0, "t")


# ---------------------------------------------------------------- metadata
@pytest.mark.parametrize("mod", [hh, synapses, stdp, da_stdp, homeostasis, cpg])
def test_params_meta_sources(mod):
    import re
    assert mod.CONFIDENCE in ("established", "strongly_supported", "incomplete", "hypothesis", "speculative")
    keys = {"HH1952", "JahrStevens1990", "Destexhe1994", "Destexhe1998", "BiPoo1998", "BiPoo2001", "Song2000",
            "Izhikevich2007", "Schultz1997", "Turrigiano1998", "Turrigiano2008", "vanRossum2000", "Brown1911",
            "Matsuoka1985", "Matsuoka1987", "MarderCalabrese1996", "DayanAbbott2001", "RushLarsen1978"}
    for name, m in mod.PARAMS_META.items():
        assert {"value", "unit", "source", "confidence"} <= set(m), name
        for tok in (x.strip() for x in m["source"].split(";")):
            assert tok in keys or tok.startswith("ASSUMPTION"), (name, tok)


def test_registry_and_validation():
    for n in ["hh_neuron", "synapse_ampa", "synapse_nmda", "synapse_gaba_a", "pair_stdp", "da_stdp",
              "homeostasis", "cpg_matsuoka"]:
        m = REGISTRY.create(n)
        validate_mechanism(m)
        m.init_state(2, RNG)


# ----------------------------------------------------------------- synapses
def test_synapse_peak_normalisation():
    s = AMPASynapse(g_max_nS=2.0)
    st = s.on_spike(s.init_state(1, RNG), 0, 1.0)
    peak = 0.0
    for _ in range(4000):
        st, _ = s.step(0, 0.005, st, {})
        peak = max(peak, st["g"][0])
    assert peak == pytest.approx(2.0, rel=1e-3)
    assert norm_factor(0.2, 2.0) > 0


def test_synapse_linear_summation():
    s = AMPASynapse()
    one = s.on_spikes(0, s.init_state(1, RNG), [SpikeEvent(0, "x", 0)])
    two = s.on_spikes(0, s.init_state(1, RNG), [SpikeEvent(0, "x", 0)] * 2)
    for _ in range(100):
        one, _ = s.step(0, 0.025, one, {})
        two, _ = s.step(0, 0.025, two, {})
    assert two["g"][0] == pytest.approx(2 * one["g"][0])


def test_nmda_block_and_sign():
    n = NMDASynapse()
    st = n.on_spike(n.init_state(1, RNG), 0, 1.0)
    st, _ = n.step(0, 10.0, st, {})  # g(0) = 0 by construction; look at the rising/peak phase
    assert abs(n.current(st, -80.0)) < abs(n.current(st, -20.0))
    assert mg_block(-80) < mg_block(-20) < 1
    g = GABAASynapse()
    sg = g.on_spike(g.init_state(1, RNG), 0, 1.0)
    sg, _ = g.step(0, 2.0, sg, {})
    assert g.current(sg, -50.0)[0] > 0  # outward (hyperpolarising) when V > E_GABA


# --------------------------------------------------------------------- STDP
def run_pair(pre_t, post_t, dt, T=200.0, w0=0.5, **kw):
    m = PairSTDP(1, 1, w0=w0, **kw)
    s = m.init_state(1, RNG)
    for i in range(int(round(T / dt))):
        t = i * dt
        pre = np.array([any(abs(t - x) < dt / 2 for x in pre_t)])
        post = np.array([any(abs(t - x) < dt / 2 for x in post_t)])
        s, _ = m.step(t, dt, s, {"pre_spikes": pre, "post_spikes": post})
    return s["w"][0, 0]


def test_stdp_window_signs():
    assert stdp_window(10.0) > 0 > stdp_window(-10.0) and stdp_window(0.0) == 0


def test_stdp_online_matches_window():
    w = run_pair([50.0], [60.0], 0.05)
    assert w - 0.5 == pytest.approx(float(stdp_window(10.0)), rel=1e-6)
    w = run_pair([60.0], [50.0], 0.05)
    assert w - 0.5 == pytest.approx(float(stdp_window(-10.0)), rel=1e-6)
    assert run_pair([50.0], [50.0], 0.05) == pytest.approx(0.5)


def test_stdp_dt_independent_and_clip():
    assert run_pair([50, 70], [55, 80], 0.05) == pytest.approx(run_pair([50, 70], [55, 80], 0.1), abs=1e-12)
    assert run_pair([50.0] * 1, [55.0], 0.05, w0=1.0) == 1.0  # clipped at w_max
    assert run_pair([55.0], [50.0], 0.05, w0=0.0) == 0.0  # clipped at w_min


# ------------------------------------------------------------------ DA-STDP
def run_da(events, da_at, T=1500.0, dt=0.5, **kw):
    m = DASTDP(1, 1, w0=1.0, **kw)
    s = m.init_state(1, RNG)
    for i in range(int(round(T / dt))):
        t = i * dt
        inp = {"pre_spikes": np.array([any(abs(t - x) < dt / 2 for x in events["pre"])]),
               "post_spikes": np.array([any(abs(t - x) < dt / 2 for x in events["post"])]),
               "da": 0.5 if da_at is not None and abs(t - da_at) < dt / 2 else 0.0}
        s, _ = m.step(t, dt, s, inp)
    return s


def test_da_stdp_behaviour():
    pre_post = {"pre": [10.0], "post": [15.0]}
    post_pre = {"pre": [15.0], "post": [10.0]}
    assert run_da(pre_post, None)["w"][0, 0] == 1.0  # no DA -> no change
    assert run_da(pre_post, 100.0)["w"][0, 0] > 1.0
    assert run_da(post_pre, 100.0)["w"][0, 0] < 1.0
    assert run_da(pre_post, None, d_tonic=0.1)["w"][0, 0] != 1.0  # tonic d with c != 0


# -------------------------------------------------------------- homeostasis
def test_rate_estimator_10Hz():
    est, dt = RateEstimator(1, 1000.0), 0.1
    for i in range(int(20000 / dt)):
        est.step(dt, [i % 1000 == 0])
    assert est.r[0] == pytest.approx(10.0, rel=0.1)


def test_scaling_direction_and_ratios():
    sc = SynapticScaling(nu_target=5.0, tau_s=1000.0)
    w = np.array([[1.0, 2.0]])
    up = sc.step(10.0, w, np.array([1.0]))
    down = sc.step(10.0, w, np.array([20.0]))
    assert up.sum() > w.sum() > down.sum()
    assert up[0, 1] / up[0, 0] == pytest.approx(2.0) and down[0, 1] / down[0, 0] == pytest.approx(2.0)
    assert (SynapticScaling(5.0, 1.0).step(10.0, w, np.array([100.0])) >= 0).all()


# ---------------------------------------------------------------------- CPG
def test_cpg_oscillates_and_tau_a_lengthens_period():
    assert cpg.oscillation_condition()
    c = simulate_cpg(8000.0)
    p1 = oscillation_period(c["t"], c["out"])
    p2 = oscillation_period(*(lambda r: (r["t"], r["out"]))(simulate_cpg(8000.0, params={"tau_a": 400.0})))
    print(f"\nCPG period default={p1:.1f} ms, tau_a=400: {p2:.1f} ms")
    assert np.isfinite(p1) and p2 > p1
