# Refleks istezanja: implementacija i aproksimacije

Status: faza 1, izvršilac okruženja i senzora. Klasa pouzdanosti celog lanca: **I** (petlja je S,
parametri su I). Specifikacija: `Specijalista sistemske neuronauke i otelotvorenja`.

## Petlja po koraku (dt = 1 ms)

1. `LimbEnv.read()` daje ugao `theta`, brzinu `omega`, dodir i dužine mišića.
2. Dužina ekstenzora `l_ext = l0 + r·theta` (r = 0.03 m, l0 = 0.10 m). Ovo je geometrija, ne ponašanje.
3. `MuscleSpindle.step(L/L0)` daje Ia i II stope i Poisson spajkove (N_ia = 30, N_ii = 20).
4. Afferentni pogon ide ka `MotorPool` (**privremeno**, vidi dole).
5. `MotorPool.step(drive, dt)` daje spajkove motoneurona (N = 50, Henneman redosled).
6. Spajkovi se filtriraju (tau = 10 ms) u EMG (težinski zbir `twitch_peak`) i ekscitaciju `u = clip(EMG/40 Hz)`.
7. `HillMuscle.step(u, l, v, dt)` daje silu ekstenzora; fleksor je pasivan (`u = 0`).
8. Moment `tau = r·(F_flex − F_ext)` ide u `LimbEnv.step(tau)`; MuJoCo integriše.

Nijedno pravilo oblika „ako je istezanje > x, aktiviraj mišić" ne postoji. Refleks izranja iz ove petlje.
Kontrola `reflex_enabled=False` prekida afferentni pogon, a bazna linija `stretch=False` ne daje perturbaciju.

## Fajlovi

| Modul | Sadržaj |
|---|---|
| `src/neural/systems/spindle.py` | `MuscleSpindle`, `SpindleParams` |
| `src/neural/systems/motor_pool.py` | `MotorPool` (LIF, Henneman) |
| `src/neural/systems/muscle.py` | `HillMuscle` |
| `src/neural/systems/stretch_reflex.py` | `run_stretch_experiment`, `StretchDevice`, `ReflexConfig` |
| `src/neural/embodiment/limb_env.py` | `LimbEnv`, `LimbParams` |
| `assets/mujoco/limb_1dof.xml` | model uda (hinge, bez gravitacije, capsula 0.3 m, 1 kg, prigušenje 0.5) |
| `examples/stretch_reflex/run.py` | primer sa merama i grafikonom |

## Parametri i jedinice

- Potencijali i ugao: radijani, N·m, m, s, Hz (stope), N (sile). Dužine mišića u m.
- Spindle: `k_v = 20`, `k_L = 300`, `k_II = 150` (Hz po jedinici L/L0 − 1), `exp = 0.6`, `tau_v = 5 ms`, bazne stope 0.
  γ_d množi dinamički, a γ_s statički član Ia; II nije modulisan γ.
- MotorPool: `size_ratio = 50`, `tau_m = 20 ms`, `drive_gain = 1.25`, refraktarnost 25 ms (maks. ~40 Hz).
- HillMuscle: `F_max = 100 N`, `l_opt = 0.10 m`, `v_max = 1 m/s`, `l_slack = 0.09 m`, `tau_act = 10 ms`, `tau_deact = 40 ms`,
  `k_pe = 0.05`, `c = 4`, `w = 0.45`.
- Stretch uređaj: opruga ka rampi ugla, `K = 5 N·m/rad`, bez prigušenja.

## Jednačine (kratko)

- Ia = γ_d·k_v·sign(v)·|v|^0.6 + γ_s·k_L·(L/L0 − 1), ograničeno na ≥ 0. v se dobija iz razlike dužina i filtra.
- II = k_II·(L/L0 − 1), ≥ 0.
- F = F_max·[a·f_L(l)·f_V(v) + f_PE(l)], `da/dt = (u − a)/τ`, τ = τ_act ili τ_deact.
- f_V: skraćivanje `(1 − s)/(1 + 4s)`, s = −v/v_max; izduživanje `1.8 − 0.8·exp(−e/0.3)`, e = v/v_max.
- `tau = r·(F_flex − F_ext)`, `l_ext = l0 + r·theta`, `l_flex = l0 − r·theta`.

## Aproksimacije i ograničenja (klasa I–X po specifikaciji)

| Aproksimacija | Klasa | Šta se gubi | Zamena |
|---|---|---|---|
| Ia bez nelinearnosti bag/chain, bez "initial burst" | S/I | razlika nuklearna kesa/lanac, istorijski efekti | Mileusnic model (faza 2) |
| Bez γ-motoneurona (skalarni γ) | I | dinamičko podešavanje vretena | moduli `neuromod` i γ-motoneuroni |
| MotorPool bez PIC-a, bez dendritskih efekata, AHP pojednostavljen | S/I | kasnija nelinearnost | višekompartmentni model |
| HillMuscle bez tetive, bez zamora, bez istorije | S/I | elastičnost tetive, zamor | Millard model sa tetivom |
| LimbEnv jedan zglob, bez gravitacije, bez kontakta | E/I | složena dinamika uda | MuJoCo tendon/muscle |
| Fleksor pasivan (`u = 0`) | I | recipročna inhibicija | Ia inhibicija (kuka) |
| **Afferentni pogon bez sinapse (privremeno)** | I | sinaptička kinetika (τ, nelinearnost) | `neural.network.synapses` kad se spoji |

### Privremeni afferentni pogon (važno)

Specifikacija traži da Ia spajkovi prolaze kroz sinapse iz `neural.network.synapses`. Te sinapse
su na grani `aide/dbcb86-...` (ćelijski tim), a nisu na ovoj grani. Zato je `_afferent_drive`
linearni pogon proporcionalan filtriranoj populacionoj stopi Ia spajkova, bez lokalne kopije kinetike sinapse.
Kad se sinapse spoje, ovu funkciju treba zameniti i ponovo izmeriti sve brojke.

## Interfejsi

- `LimbEnv` nema još `neural.core.interfaces.Sensor/Actuator` (grana `aide/862bcd-...`). Trenutno implementira
  `read()` i `step(torque)` strukturno. Kad se jezgro spoji, uskladiti potpise.
- `run_stretch_experiment` vraća dict sa ključevima `t, angle, emg, force, latency_ms, emg_peak, force_peak_N, angle_peak_rad`.

## Validacija

- Kvalitativno: spindle i Hill testovi iz `tests/contract/test_components.py`.
- Refleks: `tests/contract/test_stretch_reflex.py` (reproducibilnost, izranjanje, kontrola bez refleksa, skaliranje amplitudom).
- Latencija se navodi kao očekivanje 5–50 ms. Izmerena vrednost se prijavljuje posebno u izveštaju, ne ovde.
