"""ChannelWorld: determinizam, bekstvo (pozitivna i negativne kontrole), i.i.d. komande, validacije, chem invarijantnost."""

import numpy as np
import pytest

from neural.environment import ChannelWorld, WorldConfig


def _quadrature(k, f_hz=1.0):
    ph = 2 * np.pi * f_hz * k / 1000.0
    return np.array([
        0.5 + 0.5 * np.sin(ph),
        0.5 - 0.5 * np.sin(ph),
        0.5 + 0.5 * np.cos(ph),
        0.5 - 0.5 * np.cos(ph),
    ])


def _in_phase(k, f_hz=1.0):
    s = 0.5 + 0.5 * np.sin(2 * np.pi * f_hz * k / 1000.0)
    return np.array([s, 1 - s, s, 1 - s])


def _anti_phase(k, f_hz=1.0):
    s = 0.5 + 0.5 * np.sin(2 * np.pi * f_hz * k / 1000.0)
    return np.array([s, 1 - s, 1 - s, s])


def _static(k):
    return np.full(4, 0.7)


def _run(cmd_fn, seed, dims, steps, dt=1.0):
    w = ChannelWorld(WorldConfig(dims=dims), seed)
    w.reset()
    for k in range(steps):
        w.step(cmd_fn(k), dt)
        p = w.probe()
        if p.escaped:
            return p.t_escape_ms
    return None


# --- determinizam -----------------------------------------------------------

@pytest.mark.parametrize("dims", [2, 3])
def test_determinism_same_seed(dims):
    def trace(seed):
        w = ChannelWorld(WorldConfig(dims=dims), seed)
        out = [w.reset().as_array()]
        probes = [w.probe().x.copy()]
        rng = np.random.default_rng(5)
        for _ in range(200):
            out.append(w.step(rng.uniform(0, 1, 4), 1.0).as_array())
            probes.append(w.probe().x.copy())
        return np.stack(out), np.stack(probes)

    o1, p1 = trace(42)
    o2, p2 = trace(42)
    assert np.array_equal(o1, o2)
    assert np.array_equal(p1, p2)
    o3, p3 = trace(43)
    assert not np.array_equal(o1, o3)
    assert not np.array_equal(p1, p3)


# --- pozitivna kontrola -----------------------------------------------------

def test_positive_control_quadrature_escapes_within_3000ms():
    times = [_run(_quadrature, s, 2, 3000) for s in range(10)]
    assert all(t is not None for t in times), times
    assert np.median(times) <= 3000.0


def test_positive_control_quadrature_escapes_in_3d_too():
    assert _run(_quadrature, 0, 3, 3000) is not None


# --- negativne kontrole -----------------------------------------------------

@pytest.mark.parametrize("cmd_fn", [_in_phase, _anti_phase, _static], ids=["in_phase", "anti_phase", "static"])
def test_negative_controls_do_not_escape(cmd_fn):
    assert _run(cmd_fn, 0, 2, 6000) is None


# --- i.i.d. uniformne komande -----------------------------------------------

def test_iid_uniform_commands_false_escape_rate():
    escapes = 0
    for s in range(20):
        cmds = np.random.default_rng(1000 + s).uniform(0.0, 1.0, size=(5000, 4))
        if _run(lambda k, c=cmds: c[k], s, 2, 5000) is not None:
            escapes += 1
    assert escapes <= 2, f"lažna bekstva: {escapes}/20"


# --- validacije -------------------------------------------------------------

def test_dt_above_one_raises():
    w = ChannelWorld(WorldConfig(), 0)
    w.reset()
    with pytest.raises(ValueError):
        w.step(np.full(4, 0.5), 1.5)


def test_dt_nonpositive_raises():
    w = ChannelWorld(WorldConfig(), 0)
    w.reset()
    with pytest.raises(ValueError):
        w.step(np.full(4, 0.5), 0.0)


def test_dims_below_two_raises():
    with pytest.raises(ValueError):
        WorldConfig(dims=1)


def test_commands_wrong_shape_raises():
    w = ChannelWorld(WorldConfig(), 0)
    w.reset()
    with pytest.raises(ValueError):
        w.step(np.full(3, 0.5), 1.0)


# --- chem invarijantnost na q_B ---------------------------------------------

@pytest.mark.parametrize("dims", [2, 3])
def test_chem_and_front_back_touch_invariant_to_q_B(dims):
    # isti q_A = a0 - a1 (varira u vremenu), različit q_B = a2 - a3 sa faznim
    # pomakom u odnosu na q_A (kolinearni q_A/q_B bi dali nultu Lévyjevu površinu)
    n = 300
    t = np.arange(n)
    qa = 0.4 * np.sin(2 * np.pi * t / 60.0)

    def make_seq(amp):
        qb = amp * np.cos(2 * np.pi * t / 60.0)
        return [np.array([0.5 + q / 2, 0.5 - q / 2, 0.5 + b / 2, 0.5 - b / 2])
                for q, b in zip(qa, qb)]

    seq_a = make_seq(0.3)
    seq_b = make_seq(0.9)

    def run_seq(seq):
        w = ChannelWorld(WorldConfig(dims=dims), 17)
        w.reset()
        obs = [w.step(c, 1.0) for c in seq]
        return obs, w.probe()

    oa, pa = run_seq(seq_a)
    ob, pb = run_seq(seq_b)
    for x, y in zip(oa, ob):
        assert np.array_equal(x.chem_gradient, y.chem_gradient)
        assert np.array_equal(x.touch[:2], y.touch[:2])
    # kontrola: q_B zaista utiče na poprečni pomak (inače test ne bi ništa dokazivao)
    assert pa.x_perp_norm != pb.x_perp_norm
    assert not np.array_equal(pa.x[1:], pb.x[1:])
