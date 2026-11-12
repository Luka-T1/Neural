# NEUral

Simulacioni motor za spajkujuće neuralne modele organizma sa telom u N-D okruženju i eksperimentima
o bekstvu iz dimenzije (`dimension_escape`). Projekat je istraživački: rezultati su kvalitativni, a status
svake tvrdnje je u `VALIDATION.md` i `docs/traceability.md`.

## Instalacija

```bash
pip install -e .
```

Opcione zavisnosti iz `pyproject.toml`:
- `pip install -e .[dev]`: pytest za testove;
- `pip install -e .[yaml]`: PyYAML za čitanje YAML konfiguracije;
- `pip install -e .[viz]`: matplotlib za crtanje vremenske linije (bez njega crtanje preskače).

## Pokretanje

Konzolna komanda `neural` (posle instalacije; isto i `python -m neural`):

```bash
neural run quick                         # pozicioni preset
neural run --preset roll --dry-run       # --preset, samo ispiše komandu
neural list                              # tabela rezultata iz experiments/dimension_escape/results
neural show experiments/dimension_escape/results/dims2_seeds10 [--json]
```

Presets: `quick` (2 dims, 2 seeda, 5 s), `phase1` (2 dims, 10 seedova, 60 s), `dims3` (3 dims, 10 seedova,
60 s), `roll` (3 dims, `roll_gain` 0.0125) i `phase4` (2 dims, `escape_sign` 1, zid 0.5 mm).
Preset se zadaje ili poziciono (`neural run <preset>`) ili kao `--preset`; kad se zada oba, izlaz 2.
Ostali argumenti (`--seeds`, `--dims`, `--condition`, `--duration-ms`, `--dt-ms`, `--roll-gain`, `--workers`,
`--out`) prosleđuju se harness-u, a poslednja vrednost pobeđuje.

Harness direktno, sa snimanjem spajkova i vremenske linije:

```bash
python -m experiments.dimension_escape --dims 2 --seeds 10 --duration-ms 60000 --out experiments/dimension_escape/results/moj --record
python -m experiments.dimension_escape.timeline experiments/dimension_escape/results/moj
```

Izlazni kod: 0 za uspeh, 2 za loše argumente ili nepostojeći direktorijum ili izveštaj. Pri `roll_gain=0`
report.json nema ključ `roll`. Bez `--record` report.json nema ključ `recording_dir`.

## Struktura

- `neural/core`: stanje, integratori (RK4, exp-Euler), raspoređivač događaja, registar mehanizama, konfiguracija, seme.
- `neural/mechanisms`: HH neuron, sinapse, STDP, DA-STDP, homeostaza, CPG (Matsuoka).
- `neural/organism`: organizam (LIF, E/I 80/20, spiking half-center CPG, tri DA-STDP pravila).
- `neural/body`: telo (senzorno-motorni interfejs, motorni neuroni sa šumom skaliranim sa dt).
- `neural/environment`: N-D okruženje (`ChannelWorld`), roll (`roll_gain`, samo dims=3), jednostrani zid (`escape_sign`).
- `neural/cli`: konzolna aplikacija `neural`.
- `experiments/`: harness `dimension_escape` (izveštaji, snimanje, vremenska linija, statistika).
- `src/neural`: ćelijski i sistemski modeli.
- `docs/`: specifikacije (`docs/specs`), bibliografija (`docs/bibliography.md`), plan Dela 2 (`docs/plan_deo2.md`),
  poređenje sa bebama (`docs/infant_comparison.md`), sledljivost (`docs/traceability.md`).
- `tests/`: testovi po komponentama; `tests/contract` sadrži ugovorne testove (specifikacije, bibliografija,
  dokumentacija, izolacija).
- `VALIDATION.md`: status svake tvrdnje (implementirano, testirano, validirano).

## Šta je validirano, a šta nije

Validirano je samo to da organizam sa CPG-om beži bolje od nasumičnog motora (`random_motor`) u 2D
(stopa 1.00 naspram 0.20, 10 seedova; `VALIDATION.md`).

Nije validirano:
- DA učenje (doprinos bekstvu): frozen i no_da beže jednako kao full;
- dims3 (dims=3, samo 3 seeda, nedovoljna snaga);
- roll (`roll_gain`) i jednostrani zid (`escape_sign`): implementirani, bez eksperimenta na više seedova;
- poređenje sa bebama (`docs/infant_comparison.md`): kvalitativno, bez podataka.

## Poznata ograničenja

- Motorni šum tela je skaliran sa dt (`noise_scale` u `neural/body/motor.py`), ali rezultati u `VALIDATION.md`
  nisu ponovo izračunati.
- Mali organizam (205 neurona) nije model ljudskog mozga; poređenja su kvalitativna.
- Izvori: većina DOI-ja je needs-verify (vidi `docs/bibliography.md`); samo tri su proverena kroz Crossref.
- Rerun posle popravki faze 4 nije urađen: `dims2_seeds10` je iz e66e958, pre popravki organizma.

## Testovi

```bash
python -m pytest -q                      # ceo skup
python -m pytest -q tests/cli            # konzola
python -m pytest -q tests/contract       # ugovori (specifikacije, bibliografija, dokumentacija)
```
