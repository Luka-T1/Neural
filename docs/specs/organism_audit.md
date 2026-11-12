# Revizija organizma (faza 4)

Predmet: `neural/organism/organism.py`. Mesta u kodu su linije pre popravke (commit 0ce5f58). Testovi: `tests/organism/test_organism_audit.py`.

| stavka | status | mesto pre popravke | popravka | test |
|---|---|---|---|---|
| a1 znaci i topologija (E→AMPA ≥ 0, I→GABA ≥ 0, senzorni pol +,+,+,−,−) | OK | `organism.py:132-168` | nema | `tests/organism/test_organism_audit.py::test_weight_matrices_are_copies_and_signs` |
| a2 autapse (dijagonale We[e,e], Wi[i,i] nenulte iz `rnd` maski) | GREŠKA | `organism.py:169` | `np.fill_diagonal(We, 0)` i `np.fill_diagonal(Wi, 0)` posle izvlačenja, pre `_WeT` | `tests/organism/test_organism_audit.py::test_autapses_zero_and_stay_zero` |
| b izolacija organizma (ulaz (9,), izlaz (4,), zabranjene reči) | OK | cela datoteka | nema | `tests/organism/test_organism.py::test_no_forbidden_content_in_source` |
| c CPG ritam, antifaza, ko-kontrakcija | OK po merenju | `organism.py:151-168` | nema | `tests/organism/test_organism_audit.py::test_cpg_rhythm_antiphase_cocontraction` |
| d `no_da` ne sme imati tonični DA (d≡0) | GREŠKA | `organism.py:192` | `d_tonic=(g_da*da_tonic) if self.dopamine else 0.0` | `tests/organism/test_organism_audit.py::test_no_da_weights_frozen_without_phasic_or_tonic` |
| e šum ne skalira sa dt | GREŠKA | `organism.py:285-287` | `noise_scale(dt)=sqrt(0.1/dt)`, `_noise_k`; bafer i RNG redosled nepromenjeni | `tests/organism/test_organism_audit.py::test_noise_scale_values_and_errors`, `tests/organism/test_organism_audit.py::test_noise_digest_unchanged` |
| f nema `reset()` organizma | GREŠKA | (ne postoji) | `Organism.reset()` | `tests/organism/test_organism_audit.py::test_reset_restores_dynamic_state_keeps_weights` |

## Napomene

- (a2) Plastični blokovi `sens_exc`, `exc_cpg` i `cross` ne sadrže dijagonalu pune matrice (E→E i I→I nisu plastični), pa STDP ne može da vrati autapsu.
- (c) Izmereno (plasticity=False, 30000 koraka, dt 0.1 ms, bez ulaza; dominantna frekvencija iz FFT od q = out0 − out1, uzorak 1 ms):

| seed | f (Hz) | corr(o0,o1) | corr(o2,o3) | CCI |
|---|---|---|---|---|
| 0 | 1.0 | −0.918 | −0.912 | 0.099 |
| 1 | 1.0 | −0.928 | −0.931 | 0.155 |
| 2 | 1.0 | −0.930 | −0.931 | 0.110 |

  Isto pre i posle popravki. Pragovi iz `docs/specs/cpg.md` važe za Matsuoka model, ne za spiking half-center; za organizam su ASSUMPTION.
- (d) Referenca je ASSUMPTION ablacije (`docs/specs/da_stdp.md`), a ne Izhikevich2007.
- (e) ASSUMPTION: `noise_nA` je definisan pri referentnom dt = 0.1 ms. Pri tom dt je skala tačno 1.0, pa je rezultat bit-identičan ranijem (digest bafera šuma za seed 0 je isti).

## Šta reset namerno zadržava

- naučene težine `w` svih pravila, `_WeT` i `_WiT`;
- `detune`;
- tok RNG-a šuma (bafer i `_noise_i` se ne resetuju; ASSUMPTION: šum se ne reseeduje).

Resetuju se: V (na početno `_V0`), refraktar, adaptacija, `ge`, `gi`, brzine, novelty stanje, `da_phasic`, `last_spikes`, brojač koraka, a po pravilu `c`, `x_pre`, `x_post` (nule) i `d` (= `d_tonic`).

## Posledica

Popravke a2 i d menjaju dinamiku, pa `results/dims2_seeds10` treba ponovo izračunati (paket B). Do tada brojevi u `VALIDATION.md` važe za staru dinamiku.
