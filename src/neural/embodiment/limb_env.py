"""LimbEnv: MuJoCo ud sa jednim zglobom i dva antagonistička mišića (geometrija).

Klasa pouzdanosti: E za fiziku (MuJoCo hinge zglob), I za parametre uda.

Karakteristike (15 tačaka, skraćeno):
1. Naziv: LimbEnv, model assets/mujoco/limb_1dof.xml.
2. Svrha: pasivna mehanika uda na koju deluju samo momenti mišića (+ opcioni spoljašnji perturbacioni moment).
3. Osnova: Todorov et al. 2012, MuJoCo (doi:10.1109/IROS.2012.6386109).
4. Matematika: hinge sa prigušenjem; moment tau = r*(F_flex - F_ext);
   dužine mišića l_flex = l0 - r*theta, l_ext = l0 + r*theta; brzine su izvodi po vremenu.
5. Implementacija: sile mišića idu u data.qfrc_applied; senzori jointpos, jointvel i touch na vrhu uda.
6. Ulazi: torque (N·m, mišićni moment) i opcioni perturbation(t, theta, omega) -> N·m (spoljašnji moment).
   Perturbacija se sabira sa mišićnim momentom i ne menja mišićni ulaz.
7. Izlazi: rečnik sa t, theta, omega, touch, muscle_len[2], muscle_vel[2]. Indeks 0 = fleksor, 1 = ekstenzor.
8. Stanje: MuJoCo MjData (qpos, qvel, time).
9. Zavisnosti: paket mujoco (4.x), numpy.
10. Interakcije: implementira čitanje senzora (read) i ulaz momenta (step). Ne zna ništa o neuronima.
11. Konfiguracija: LimbParams(moment_arm, l_opt, xml_path); ostali parametri su u XML-u.
12. Testovi: vidi tests/contract (indirektno preko refleksa); bez momenta ud miruje.
13. Validacija: bez NaN, energetska konzistentnost ograničena na pasivnu dinamiku.
14. Ograničenja: jedan stepen slobode, bez tetiva, bez gravitacije, bez kontakta osim dodira.
    Zamena: MuJoCo tendon/muscle model.
15. Integracija: reset(seed) i step(torque) su jedine ulazne tačke. Nema globalnog stanja.

Napomena o interfejsima: neural.core.interfaces (Sensor, Actuator) još nije u ovoj grani.
Dok ne postoji, LimbEnv implementira iste metode strukturno: read() i step(torque).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import mujoco
import numpy as np

FLEXOR = 0
EXTENSOR = 1

_DEFAULT_XML = Path(__file__).resolve().parents[3] / "assets" / "mujoco" / "limb_1dof.xml"

Perturbation = Callable[[float, float, float], float]


@dataclass(frozen=True)
class LimbParams:
    """Geometrija mišića koja mora biti usklađena sa XML modelom."""

    moment_arm: float = 0.03   # r, m
    l_opt: float = 0.10        # l0, m (optimalna dužina mišića pri theta = 0)
    xml_path: Optional[str] = None


class LimbEnv:
    """Jedan zglob, dva mišića, MuJoCo fizika. Implementira čitanje senzora i ulaz momenta."""

    def __init__(self, seed: int = 0, perturbation: Optional[Perturbation] = None,
                 params: Optional[LimbParams] = None):
        self.params = params or LimbParams()
        self.perturbation = perturbation
        path = self.params.xml_path or str(_DEFAULT_XML)
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._dof = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "touch_tip")
        self._touch_adr = int(self.model.sensor_adr[sid])
        self.seed = int(seed)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def l_opt(self) -> float:
        return self.params.l_opt

    def reset(self, seed: Optional[int] = None) -> dict:
        """Vraća uđe u početno stanje (theta = 0, omega = 0). Seed se čuva; ud nema slučajnost."""
        if seed is not None:
            self.seed = int(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.read()

    def read(self) -> dict:
        """Senzorski ulaz (Sensor): ugao, brzina, dodir i geometrija mišića."""
        r = self.params.moment_arm
        th = float(self.data.qpos[self._dof])
        om = float(self.data.qvel[self._dof])
        l0 = self.params.l_opt
        return {
            "t": float(self.data.time),
            "theta": th,
            "omega": om,
            "touch": float(self.data.sensordata[self._touch_adr]),
            "muscle_len": np.array([l0 - r * th, l0 + r * th], dtype=float),
            "muscle_vel": np.array([-r * om, r * om], dtype=float),
        }

    def step(self, torque: float) -> dict:
        """Jedan korak (model.opt.timestep). torque je mišićni moment u N·m (Actuator ulaz)."""
        tau = float(torque)
        if self.perturbation is not None:
            th = float(self.data.qpos[self._dof])
            om = float(self.data.qvel[self._dof])
            tau += float(self.perturbation(float(self.data.time), th, om))
        self.data.qfrc_applied[self._dof] = tau
        mujoco.mj_step(self.model, self.data)
        return self.read()
