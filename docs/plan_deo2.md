# Deo 2: stanje implementacije

Stanje validacije je u `VALIDATION.md` (sekcija „Deo 2“). Sledljivost stavki je u `docs/traceability.md`.
Ovaj dokument prati šest tema Dela 2; svaka sekcija ima status implementacije i poseban status validacije.

## 1. Vizualizacija „šta organizam misli“

Status: delimično implementirano; validirano NE.

### 1.1 Snimanje (implementirano)
- `experiments/dimension_escape/recording.py` snima spajkove (`last_spikes`), fazni DA (`da_phasic`),
  `d`, `novelty_bar` i eligibility za pravila `sens_exc`, `exc_cpg` i `cross`.
- Snimanje je opciono (`--record`). Pri isključenom snimanju izlaz i report.json su bit-identični i bez
  novih ključeva (isto pravilo kao za roll_gain=0 i escape_sign=0). Uz snimanje report.json dobija samo
  ključ `recording_dir`.
- Testovi: `tests/experiments/test_recording.py::test_run_one_identical_with_and_without_record`,
  `tests/experiments/test_recording.py::test_main_record_report_identical_and_no_new_keys_without`.
- Organizam ne dobija uvid u svet: snimanje čita samo javna stanja organizma i ne menja ih
  (`tests/experiments/test_recording.py::test_recording_is_read_only`).

### 1.2 Crteži (nije implementirano)
- Raster spajkova (senzorni, E, I, CPG i motorni), dopaminski trag i eligibility nisu iscrtani.
  `experiments/dimension_escape/visualize.py` crta samo vremensku liniju iz podsekcije 3.
- Motorni izlaz (4 kanala premotorne struje i komande tela) nije iscrtan.

Kriterijum prihvatanja: dva run-a istog seeda, sa snimanjem i bez njega, daju identičan niz izlaza i
identičan report.json (osim ključa snimka kad je uključen). Taj kriterijum je ispunjen testovima iz 1.1.

## 2. Interakcija po dimenziji

Status: implementirano (dims 2 i 3, `roll_gain`, `escape_sign`, preseti CLI-ja); validirano NE.

### 2.1 CLI preseti
- Preseti `quick`, `phase1`, `dims3`, `roll` i `phase4` postoje u `neural/cli/__init__.py` (`PRESETS`).
  Testovi: `tests/cli/test_cli.py`, `tests/cli/test_cli_contract.py::test_presets_exact_argv`.
- Pozicioni oblik `neural run <preset>` i `--preset` zajedno daju izlaz 2 (`tests/cli/test_cli.py`).

### 2.2 Dimenzije i rotacija
- dims 2 i dims 3: implementirano; izveštaj po uslovu sa k/n i Wilson intervalom.
- `roll_gain` (preset `roll`, 0.0125): implementirano u `experiments/dimension_escape/harness.py` i u
  svetu; roll sekcija je prisutna samo pri roll_gain ≠ 0 (`tests/contract/test_phase3_harness_roll.py`).
  Eksperiment sa roll_gain > 0 na više seedova nije pokrenut.

### 2.3 Jednostrani zid
- `escape_sign`: implementirano u svetu (`neural/environment/channel_world.py`) i u harness-u
  (`tests/contract/test_phase4_harness_sign.py::test_report_escape_side_section`,
  `tests/contract/test_phase4_harness_sign.py::test_main_smoke_report`).
- Pri escape_sign = 0 snapshot i report.json su bit-identični sa ranijim. Eksperiment na više seedova nije
  pokrenut, pa uticaj nije validiran.

Kriterijum prihvatanja za validaciju (nije ispunjen): dims 3 na najmanje 10 seedova i roll_gain > 0 odnosno
escape_sign ≠ 0 na više seedova, sa izveštajem k/n i Wilson intervalom po uslovu.

## 3. Vremenska linija uspeha i neuspeha

Status: implementirano (`experiments/dimension_escape/timeline.py`, `experiments/dimension_escape/visualize.py`);
validirano NE.

- Po seedu i po uslovu (full, frozen, no_da, random_motor): t_escape, x_perp(t), događaji DA i faza CPG-a
  (iz q = o0 − o1).
- Testovi: `tests/experiments/test_timeline.py::test_timeline_matches_report` (t_escape u liniji je jednak
  t_escape u report.json), `tests/experiments/test_timeline.py::test_build_timeline_synthetic`.
- Crtanje zavisi od matplotlib-a; bez njega `visualize` vraća bez greške
  (`tests/experiments/test_timeline.py::test_visualize_without_matplotlib`).

## 4. Poređenje sa razvojem kod beba

Status: implementirano kao dokument (`docs/infant_comparison.md`); kvalitativno; validirano NE.

Poklapanje 1-1 sa ljudskim mozgom nije izvodljivo u organizmu ove veličine.

Organizam ima 205 neurona (9×5 senzornih, 80 E, 20 I, 4×10 CPG i 4×5 CPG interneurona; vidi `OrganismConfig` u `neural/organism/organism.py`), red veličine ~10^2, naspram ~10^11 neurona kod čoveka. Organizam nema anatomske regione; korteks, bazalne ganglije i kičmena moždina su samo analogije.

Poređenje je samo kvalitativno i zasnovano na izvorima; status: validirano = NE dok ne postoje podaci.

Izvori su u `docs/bibliography.md` pod ključevima, a poređenje je u `docs/infant_comparison.md`:
- [ThelenFisher1982] (nestajanje refleksa koračanja);
- [Forssberg1985] (koračanje novorođenčeta);
- [Dominici2011] (lokomotorni primitivi);
- [RoveeRovee1969] (kontingentno učenje udaranja nogom);
- [ThelenSmith1994] (dinamički sistemi u razvoju);
- [Azevedo2009] (broj neurona u ljudskom mozgu).

Novi izvori imaju doi_status: needs-verify dok se ne provere kroz Crossref. Nijedan izvor u ovom Delu
nije proveren, pa nijedan podatak o bebama nije uvršten u analizu kao validiran.

## 5. Otvorena pitanja i rizici

Status: rizici su popisani; nijedan nije rešen osim motornog šuma (vidi dole).

- Doprinos DA-STDP učenja bekstvu nije validiran: frozen i no_da beže kao full (`VALIDATION.md`).
- Popravke organizma u fazi 4 (autapse, d≡0 pri no_da, šum sa dt, reset) menjaju dinamiku, pa rezultati
  `dims2_seeds10` treba ponovo izračunati; rerun nije urađen (`VALIDATION.md`, stavka „Rerun eksperimenata
  posle popravki faze 4“).
- Dims 3, roll i escape_sign nemaju eksperiment na više seedova.
- Motorni šum u `neural/body/motor.py` je ispravljen: skalira se sa `noise_scale(dt) = sqrt(0.1/dt)`
  (`tests/body/test_motor_noise_dt.py`). Otvoreno ostaje samo to što ispravka nije ponovo merena u eksperimentu.
- Poređenje sa bebama zavisi od izvora koje tek treba proveriti kroz Crossref (doi_status: needs-verify).
- Snimanje za vizualizaciju sme da uspori run; trošak nije izmeren pre uključivanja u preset.

## 6. Realni obim biološke sličnosti u Delu 2

Status: nije implementirano za dopune; postojeći deo je implementiran (vidi „Već postoji“).

### Već postoji
- HH i LIF neuroni, E/I populacije, DA-STDP sa eligibility tragom, homeostaza, CPG (half-center) i organizam
  od 205 neurona (`neural/organism/organism.py`, `neural/mechanisms/`).

### Nije implementirano
- Ostali neuromodulatori (5-HT, ACh, NE), dendriti i višekompartmentni modeli, kortikalni slojevi, glija,
  razvojna plastičnost.

### Ograničenja
- Strukturno, hemijski i psihološki poklapanje sa ljudskim mozgom nije cilj. Organizam ostaje mali model
  (confidence: speculative), a ne model ljudskog mozga.
- Poređenja sa ljudskim mozgom su samo kvalitativna; validirano = NE dok ne postoje podaci.
