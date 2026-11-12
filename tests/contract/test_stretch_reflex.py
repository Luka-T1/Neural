import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
import pytest

pytest.importorskip('mujoco')


def _run(**kw):
    from neural.systems.stretch_reflex import run_stretch_experiment
    return run_stretch_experiment(**kw)


def test_reproducible_with_seed():
    a = _run(seed=3); b = _run(seed=3)
    assert np.array_equal(np.asarray(a['emg']), np.asarray(b['emg']))
    assert a['latency_ms'] == b['latency_ms']


def test_reflex_emerges_from_loop():
    on = _run(seed=0, reflex_enabled=True)
    off = _run(seed=0, reflex_enabled=False)
    base = _run(seed=0, stretch=False)
    assert on['emg_peak'] > 1.5 * off['emg_peak']
    assert on['emg_peak'] > 1.5 * base['emg_peak']
    assert 0.0 < on['latency_ms'] <= 50.0
    assert on['force_peak_N'] > 0.0
    assert np.all(np.isfinite(np.asarray(on['angle'])))


def test_amplitude_scales_with_stretch():
    small = _run(seed=0, amplitude_rad=0.05)
    big = _run(seed=0, amplitude_rad=0.15)
    assert big['emg_peak'] >= small['emg_peak']
