# VALIDATION

Datum: 2026-11-12. „Implementirano“ = kod postoji. „Testirano“ = komanda je pokrenuta i zelena. „Validirano“ = konkretan izmeren rezultat sa statistikom ili poređenje sa podacima. Ugovorni pragovi i testovi su provere kvalitativne verodostojnosti, ne validacija prema eksperimentalnim podacima.

## Organizam i eksperiment `dimension_escape`

| stavka | implementirano | testirano | validirano |
|---|---|---|---|
| Organizam (LIF, E/I 80/20, spiking half-center CPG, tri `DASTDP` pravila `sens_exc`, `exc_cpg`, `cross`) — `neural/organism/organism.py` | DA | DA (`tests/organism`) | NE |
| Izolacija organizma (ulaz samo (9,) nA, izlaz samo (4,) nA; zabranjene reči u izvoru) | DA | DA (`tests/organism`, `tests/contract/test_dimension_escape_integration.py`) | NE |
| Harness + CLI (stvarni `ChannelWorld` i `Body`, `body.reset()`, bekstvo iz probe snapshot-a, `derive_seed`) | DA | DA (`tests/experiments`, `tests/contract/test_dimension_escape_integration.py`, determinizam po seedu) | NE (metodološki) |
| Statistika (Wilson, Fisher, MWU, Holm, Spearman) | DA | DA (`tests/experiments/test_stats.py`; ručno poznate vrednosti) | DA samo za poznate vrednosti; poređenje sa scipy nije urađeno |
| **Bekstvo: CPG-pogonjeno ponašanje (full vs random_motor), dims=2** | DA | DA | **DA** (vidi dole) |
| **Bekstvo, dims=3** | DA | DA | **NE: nedovoljna snaga (3 seeda)** |
| **Doprinos DA-STDP učenja bekstvu** | DA | DA (frozen ne menja težine, full menja) | **NE — nije validirano** |
| DA leakage audit | DA | DA | izmereno, vidi dole; ne tumačiti kao dokaz odsustva curenja |

### Rezultati (git hash e66e958 (pun: e66e958bac69875937decde1148a38b2b90721e7), commit-ovano čisto stablo; seed‑ovi 0..n−1; `derive_seed(seed, ime)`)

dims=2, 10 seedova, 60 s, dt 0.1 ms (`python -m experiments.dimension_escape --seeds 10 --dims 2 --condition all --duration-ms 60000 --dt-ms 0.1 --out experiments/dimension_escape/results/dims2_seeds10`, 6 min 58 s):

| uslov | k/n | Wilson 95% | medijana t_esc (ms) |
|---|---|---|---|
| full | 10/10 | 0.72–1.00 | 865 |
| random_motor | 2/10 | 0.06–0.51 | 14266 |
| frozen | 10/10 | 0.72–1.00 | 1079 |
| no_da | 10/10 | 0.72–1.00 | 1020 |

| poređenje | Fisher p | MWU p | p_holm |
|---|---|---|---|
| full vs random_motor | 0.00071 | 0.00011 | **0.0021** |
| full vs frozen | 1 | 0.472 | 1 |
| full vs no_da | 1 | 0.290 | 1 |

dims=3, 3 seeda, 10 s (`python -m experiments.dimension_escape --seeds 3 --dims 3 --condition all --duration-ms 10000 --dt-ms 0.1 --out experiments/dimension_escape/results/dims3_short`, 35 s): full 3/3, frozen 3/3, no_da 3/3, random_motor 0/3; full vs random_motor Fisher p = 0.1, **p_holm = 0.3**. Sa 3 seeda test nema snagu; zaključak za dims=3 ne postoji.

**Promena u odnosu na run iz radnog stabla roditelja 28d9a7d:** brojevi se razlikuju (random_motor 3/10 → 2/10; medijana full 881 → 865 ms, frozen 886 → 1079, no_da 920 → 1020; p_holm full vs random_motor 0.0093 → 0.0021; w_cross kraj 0.0568 → 0.0564; ρ full dims=2 −0.083 → −0.192). Razlika nije samo od tipskih anotacija: između 28d9a7d i e66e958 menjani su `neural/environment/channel_world.py`, `neural/core/interfaces.py`, `neural/core/simulator.py` i `neural/organism/organism.py`. Doprinos pojedinačnih izmena nije izolovan; uzrok svake razlike je nepoznat. Kvalitativni zaključak je isti.

### Tumačenje (važno)
- **Frozen i no_da beže jednako kao full** (10/10, medijane 1079 i 1020 ms naspram 865 ms; MWU p 0.472 i 0.290; p_holm = 1). Efekat potiče od CPG dinamike organizma, a ne od DA-STDP učenja. **Hipoteza da DA-STDP učenje doprinosi bekstvu NIJE validirana** i ostaje „NE“ dok novi run (npr. duži, sa zadatkom koji zahteva učenje) ne pokaže razliku.
- Plastičnost je jedva pomerila težine: srednja `w_cross` full 0.0499 → 0.0564 (dims=2), no_da 0.0499 → 0.0484, frozen 0.0499 → 0.0499 (nepromenjeno po konstrukciji).
- DA leakage ρ(DA, |ẋ⊥|): dims=2 full −0.192, frozen −0.178; dims=3 full −0.215, frozen −0.128 (frozen nema pravo učenje, pa je korelacija u njemu čisto slučajna/novelty artefakt).
- Validirano je samo da organizam sa CPG-om beži bolje od nasumičnog OU motora u 2D (stopa 1.00 vs 0.20).
- Kandidat zadatka za test doprinosa DA-STDP specificiran je u `docs/experiment_design.md` (j); nije pokrenut.

## Mehanizmi (`neural/mechanisms/`, specifikacije u `docs/specs/`)

| mehanizam | implementirano | testirano | validirano |
|---|---|---|---|
| hh | DA | DA (`tests/mechanisms/test_hh.py`) | NE (kvalitativno vs HH1952) |
| synapses, stdp, da_stdp, homeostasis, cpg | DA | DA (`tests/mechanisms/test_plasticity_misc.py`) | NE |

Provere mehanizama pripadaju izvršiocu jezgra; ovde su samo prijavljene.

## Faza 3: roll (`roll_gain`)

| stavka | implementirano | testirano | validirano |
|---|---|---|---|
| svet: roll φ, osa n (samo dims=3) | DA | DA (`tests/contract/test_phase3_roll_world.py`, `tests/environment/test_channel_world_roll_attenuation.py`) | NE |
| harness: `--roll-gain`, φ iz probe snapshot-a | DA | DA (`tests/contract/test_phase3_harness_roll.py`; roll_gain>0 uz `NEURAL_REQUIRE_PHASE3=1`) | NE |

Pri `roll_gain=0` izlaz harness-a i report.json (bez ključa `roll`) ostaju isti kao u fazama 1 i 2. Nijedan eksperiment sa `roll_gain>0` nije pokrenut, pa nema zaključka o uticaju rolla na bekstvo.

## Faza 4: jednostrani zid, revizija organizma, CLI

| stavka | implementirano | testirano | validirano |
|---|---|---|---|
| Jednostrani zid `escape_sign` (svet; pri 0 bez novih ključeva u snapshot-u) | DA | DA (`tests/contract/test_phase4_world_sign.py`, `tests/environment/test_channel_world_one_sided.py`) | NE (nema eksperimenta na više seedova) |
| Revizija organizma (CPG ritam i ko-kontrakcija, d≡0 pri no_da, šum sa sqrt(0.1/dt), znaci E/I težina, autapse; `reset()`) — `docs/specs/organism_audit.md` | DA | DA (`tests/organism/test_organism_audit.py`) | NE (rezultati `dims2_seeds10` nisu ponovo izračunati) |
| Reset motornog RNG-a (`Body.reset` vraća `_rng_state0`, `neural/body/body.py:53-65`) | DA | DA (`tests/body/test_body_reset_rng.py`) | NE |
| Opseg touch[2]: min(1, exp(...) + tanh(dubina/_TOUCH_SCALE_MM)) ∈ [0,1], _TOUCH_SCALE_MM=0.05 ASSUMPTION | DA | DA (`tests/environment/test_channel_world_one_sided.py`) | NE |
| CLI preset (`neural run --preset quick, phase1, dims3, roll ili phase4`; pozicioni oblik `neural run <preset>`; roll_gain 0.0125; phase4 = escape_sign) | DA | DA (`tests/cli/test_cli.py::test_run_positional_preset_dry_run`, `tests/cli/test_cli.py`, `tests/cli/test_cli_contract.py`, `tests/contract/test_phase4_harness_sign.py`) | NE (preseti dims3, roll i phase4 nisu pokrenuti) |
| Harness escape_sign (`escape_side`, sekcija Strana bekstva u izveštaju) | DA | DA (`tests/contract/test_phase4_harness_sign.py`) | NE (nema eksperimenta na više seedova) |

Faza 4 i roll ostaju validirano=NE dok se ne pokrene i ne izvesti dimension_escape sa --dims 3 (i --roll-gain > 0, odnosno --escape-sign ≠ 0) na više seedova.

## Deo 2

Stanje po stavkama Dela 2 (`docs/plan_deo2.md`). Sve kolone „validirano“ ostaju NE dok se ne izvedu eksperimenti
sa brojkama; brojke se upisuju u sledećoj iteraciji.

| stavka | implementirano | testirano | validirano |
|---|---|---|---|
| Motorni šum sa dt (noise_scale) | DA (`neural/body/motor.py`) | DA (`tests/body/test_motor_noise_dt.py::test_noise_scale_reference_and_value`) | NE (ispravka nije ponovo izmerena u eksperimentu) |
| Snimanje (recording) | DA (`experiments/dimension_escape/recording.py`) | DA (`tests/experiments/test_recording.py::test_run_one_identical_with_and_without_record`) | NE (snimci nisu analizirani ni za jedan eksperiment na više seedova) |
| Vremenska linija (timeline) | DA (`experiments/dimension_escape/timeline.py`) | DA (`tests/experiments/test_timeline.py::test_timeline_matches_report`) | NE (linija nije izvedena za eksperimente na više seedova) |
| Poređenje sa bebama (`docs/infant_comparison.md`) | DA (dokument) | DA (`tests/contract/test_deo2_docs.py`) | NE (speculative: kvalitativno poređenje, nema podataka o bebama) |
| Rerun eksperimenata posle popravki faze 4 | NE (rerun nije urađen) | NE (nema testa koji meri rerun) | NE (`dims2_seeds10` je još iz e66e958, pre popravki faze 4) |

## Provere pokrenute u ovoj rundi
- `python -m pytest -q` (ceo skup): vidi izveštaj lead-u (broj prolazaka upisan u izveštaju).
- `python -m mypy experiments/dimension_escape` i `python -m mypy -p neural.organism`: bez grešaka.

## Poznata ograničenja i ASSUMPTION lista
- Organizam je lokalna realizacija (confidence `speculative`); CPG je spiking half-center jer `neural.mechanisms.cpg.MatsuokaCPG` je model brzine i nije kompatibilan sa DA-STDP-om na A↔B sinapsama.
- ASSUMPTION: parametri LIF ćelije, težine, bias struje, adaptacija (b = 8 pA, τ_w = 450 ms), E_GABA = −75 mV, A− = 1.05·A+, η = 0.02, ε = 0.05 nA, DA_phasic ≤ 10, w_max = 4·srednja početna težina. CPG je podešen samo na autonomni ritam.
- Holm se primenjuje na Fisherove p-vrednosti; MWU p je izveštajan.
- ASSUMPTION: šum organizma je definisan pri referentnom dt = 0.1 ms i skalira se sa sqrt(0.1/dt) (`noise_scale` u `neural/organism/organism.py`); pri dt = 0.1 rezultat je bit-identičan ranijem.
- Motorni šum tela je skaliran sa dt: `noise_scale` u `neural/body/motor.py` (sqrt(0.1/dt)), test `tests/body/test_motor_noise_dt.py`. Ispravka je u kodu (Deo 2), ali rezultati u ovom dokumentu nisu ponovo izračunati.
- Bekstvo definiše svet (probe snapshot); harness nema sopstveni prag.
- DOI faze 3 ShapereWilczek1989, Hogan1984 i LaugaPowers2009 su verified (Crossref). Ostali, uključujući Bilbao2018, su needs-verify.
- Runovi su ponovljeni na commit-ovanom kodu (e66e958). Posle toga je u `neural/core/config.py` menjan samo čitač konfiguracije; to ne menja parametre sveta ni organizma.
- Instalacija: `pip install -e .[dev]`; `[yaml]` se dodaje po potrebi (PyYAML je opcion).
