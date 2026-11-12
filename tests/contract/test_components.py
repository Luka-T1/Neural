import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import numpy as np


def _ramp_end_ia(T):
    from neural.systems.spindle import MuscleSpindle
    s = MuscleSpindle(); s.reset()
    n = int(T / 1e-3)
    ia = 0.0
    for i in range(n):
        ia, _ = s.step(1.0 + 0.1 * (i + 1) / n, 1e-3)
    return ia


def test_spindle_stretch_and_velocity():
    from neural.systems.spindle import MuscleSpindle
    s = MuscleSpindle(); s.reset()
    for _ in range(200):
        ia0, ii0 = s.step(1.0, 1e-3)
    s.reset()
    for _ in range(200):
        ia1, ii1 = s.step(1.1, 1e-3)
    assert ia0 >= 0 and ii0 >= 0
    assert ia1 > ia0 and ii1 >= ii0
    assert _ramp_end_ia(0.02) > _ramp_end_ia(0.2)


def test_henneman_recruitment():
    from neural.systems.motor_pool import MotorPool

    def active(drive):
        p = MotorPool(n_units=50, seed=1); p.reset()
        a = np.zeros(50, bool)
        for _ in range(500):
            a |= np.asarray(p.step(drive, 1e-3), bool)
        return a

    th = np.asarray(MotorPool(n_units=50, seed=1).thresholds)
    assert np.all(np.diff(th) >= 0) and th[0] < 0.1
    lo, hi = active(0.2), active(0.8)
    assert 0 < lo.sum() <= hi.sum()
    assert np.all(hi[lo])


def test_hill_muscle():
    from neural.systems.muscle import HillMuscle

    def mk():
        return HillMuscle(f_max=100.0, l_opt=0.1, v_max=1.0, l_slack=0.09)

    def settle(l, v, e=1.0):
        m = mk(); f = 0.0
        for _ in range(500):
            f = m.step(e, l, v, 1e-3)
        return f

    assert settle(0.1, 0.0) > settle(0.06, 0.0)
    assert settle(0.1, -0.3) < settle(0.1, 0.0) < settle(0.1, 0.3)
    assert settle(0.08, 0.0, e=0.0) == 0.0
    assert settle(0.13, 0.0, e=0.0) > 0.0
    m = mk()
    assert m.step(1.0, 0.1, 0.0, 1e-3) < settle(0.1, 0.0)
