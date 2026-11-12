"""Monosinaptički refleks istezanja: povezuje LimbEnv, MuscleSpindle, MotorPool i HillMuscle.

Klasa pouzdanosti: I (petlja i parametri su nepotpuno razumljivi; struktura je S).

Karakteristike (15 tačaka, skraćeno):
1. Naziv: run_stretch_experiment (petlja refleksa).
2. Svrha: izranjanje refleksa istezanja iz petlje istezanje -> aferent -> motoneuron -> mišić -> MuJoCo. (E za petlju)
3. Osnova: Matthews 1972; Prochazka & Gorassini 1998; Heckman & Enoka 2012. Citate treba proveriti.
4. Matematika: geometrija l_ext = l0 + r*theta (bez ponašanja); aferenti (spindle); pogon po
   populaciji; EMG i ekscitacija su isti filtrirani spajkovi (tau_emg = 10 ms); moment tau = r*(F_flex - F_ext).
5. Implementacija: redosled po koraku (dt = 1 ms): LimbEnv -> geometrija -> MuscleSpindle.step ->
   aferentni pogon -> MotorPool.step -> EMG/ekscitacija -> HillMuscle.step -> LimbEnv.step.
   Nema pravila oblika "ako je istezanje > x, aktiviraj mišić".
6. Ulazi: seed, amplitude_rad, velocity_rad_s, duration_s, onset_s, reflex_enabled, stretch.
7. Izlazi: rečnik sa t, angle, emg, force, latency_ms, emg_peak, force_peak_N, angle_peak_rad.
8. Stanje: stanja spindle-a, motornog bazena, mišića i MuJoCo-a; filteri pogona i EMG-a.
9. Zavisnosti: LimbEnv (mujoco), MuscleSpindle, MotorPool, HillMuscle. Sinapse iz neural.network.synapses
   NISU u ovoj grani; vidi PRIVREMENO u docstring-u funkcije _afferent_drive.
10. Interakcije: reflex_enabled=False prekida Ia -> motoneuron (kontrola). stretch=False je bazna linija.
11. Konfiguracija: ReflexConfig (ia_norm_hz, tau_afferent_s, tau_emg_s, emg_sat_hz, n_motor_units,
    stretch_stiffness, stretch_damping) i parametri mišića u modulu.
12. Testovi: tests/contract/test_stretch_reflex.py (reproducibilnost, izranjanje, skaliranje amplitudom).
13. Validacija: kvalitativna; latencija je očekivanje 5-50 ms, a izmerena vrednost se prijavljuje posebno.
14. Ograničenja: jedan zglob, jedan ekstenzor sa refleksom, fleksor pasivan; bez Ib, Renshaw, II-puta;
    bez tetive; afferentni pogon je privremen (vidi docstring _afferent_drive).
15. Integracija: jedini ulaz je run_stretch_experiment; svi ostali moduli su bez globalnog stanja.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from neural.embodiment.limb_env import EXTENSOR, FLEXOR, LimbEnv, LimbParams
from neural.systems.motor_pool import MotorPool
from neural.systems.muscle import HillMuscle
from neural.systems.spindle import MuscleSpindle, SpindleParams


@dataclass(frozen=True)
class ReflexConfig:
    ia_norm_hz: float = 60.0          # stopa Ia koja daje pogon 1.0 (normalizacija)
    tau_afferent_s: float = 0.005     # PRIVREMENO: filter afferentnog pogona
    tau_emg_s: float = 0.01           # filter spajkova motoneurona (EMG i ekscitacija)
    emg_sat_hz: float = 40.0          # stopa (težinska) koja daje ekscitaciju 1.0
    n_motor_units: int = 50
    stretch_stiffness: float = 5.0    # N·m/rad, krutost stretch uređaja
    stretch_damping: float = 0.0      # N·m·s/rad


@dataclass
class StretchDevice:
    """Spoljašnja perturbacija: opruga ka rampi ugla. Ne menja mišićni ulaz."""

    amplitude_rad: float
    velocity_rad_s: float
    onset_s: float
    stiffness: float
    damping: float

    def __call__(self, t: float, theta: float, omega: float) -> float:
        ramp = max(0.0, (t - self.onset_s) * self.velocity_rad_s)
        ref = min(self.amplitude_rad, ramp)
        return self.stiffness * (ref - theta) - self.damping * omega


def _afferent_drive(pop_rate_hz: float, cfg: ReflexConfig) -> float:
    """PRIVREMENO: zamena za sinapsu Ia -> motoneuron.

    Sinapse (neural.network.synapses) nisu u ovoj grani. Ovde je samo linearni pogon
    proporcionalan filtriranoj populacionoj stopi Ia spajkova, normalizovan na [0, 1].
    Nema lokalne kopije kinetike sinapse. Zameniti sinapsama kad se spoje.
    """
    return float(np.clip(pop_rate_hz / cfg.ia_norm_hz, 0.0, 1.0))


def run_stretch_experiment(seed: int = 0, amplitude_rad: float = 0.1,
                           velocity_rad_s: float = 1.0, duration_s: float = 0.5,
                           onset_s: float = 0.1, reflex_enabled: bool = True,
                           stretch: bool = True) -> dict:
    cfg = ReflexConfig()
    device = (StretchDevice(amplitude_rad, velocity_rad_s, onset_s,
                            cfg.stretch_stiffness, cfg.stretch_damping)
              if stretch else None)
    env = LimbEnv(seed=seed, perturbation=device, params=LimbParams())
    dt = env.dt
    r = env.params.moment_arm
    l_opt = env.l_opt

    spindle = MuscleSpindle(SpindleParams(seed=seed))
    spindle.reset()
    pool = MotorPool(n_units=cfg.n_motor_units, seed=seed)
    pool.reset()
    ext = HillMuscle(f_max=100.0, l_opt=l_opt, v_max=1.0, l_slack=0.09)
    flex = HillMuscle(f_max=100.0, l_opt=l_opt, v_max=1.0, l_slack=0.09)

    alpha_aff = min(1.0, dt / cfg.tau_afferent_s)
    alpha_emg = min(1.0, dt / cfg.tau_emg_s)
    ia_pop = 0.0
    emg = 0.0

    n = int(round(duration_s / dt))
    t_arr = np.zeros(n)
    angle = np.zeros(n)
    emg_arr = np.zeros(n)
    force = np.zeros(n)

    obs = env.reset(seed)
    for i in range(n):
        t_arr[i] = i * dt
        angle[i] = obs["theta"]

        # 2. geometrija -> 3. aferenti
        ia_hz, _ = spindle.step(obs["muscle_len"][EXTENSOR] / l_opt, dt)
        # 4. aferentni pogon (privremen)
        pop_rate = float(spindle.spikes_ia.mean()) / dt
        ia_pop += alpha_aff * (pop_rate - ia_pop)
        drive = _afferent_drive(ia_pop, cfg) if reflex_enabled else 0.0
        # 5. motoneuroni
        spikes = pool.step(drive, dt)
        # 6. EMG i ekscitacija
        weighted_rate = float(spikes @ pool.twitch_peak) / dt
        emg += alpha_emg * (weighted_rate - emg)
        emg_arr[i] = emg
        excitation = float(np.clip(emg / cfg.emg_sat_hz, 0.0, 1.0))
        # 7. mišići
        f_ext = ext.step(excitation, obs["muscle_len"][EXTENSOR],
                         obs["muscle_vel"][EXTENSOR], dt)
        f_flex = flex.step(0.0, obs["muscle_len"][FLEXOR],
                           obs["muscle_vel"][FLEXOR], dt)
        force[i] = f_ext
        # 8. moment -> MuJoCo
        torque = r * (f_flex - f_ext)
        obs = env.step(torque)

    onset_idx = int(round(onset_s / dt))
    base = emg_arr[:onset_idx] if onset_idx > 0 else np.zeros(1)
    thr = float(base.mean() + 3.0 * base.std())
    post = emg_arr[onset_idx:]
    crossed = np.nonzero(post > thr)[0]
    latency_ms = float(crossed[0] * dt * 1e3) if crossed.size else float("nan")
    angle_dev = np.abs(angle - angle[0])

    return {
        "t": t_arr,
        "angle": angle,
        "emg": emg_arr,
        "force": force,
        "latency_ms": latency_ms,
        "emg_peak": float(post.max()) if post.size else 0.0,
        "force_peak_N": float(force.max()),
        "angle_peak_rad": float(angle_dev.max()),
    }
