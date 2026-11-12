# Specifikacija: cpg

## 1. Naziv
Matsuoka polucentar (half-center oscillator)

## 2. Biološka svrha i fenomen
Ritmična naizmenična aktivnost bez ritmičnog ulaza (lokomotorni CPG).

## 3. Vremenska skala
Period ≈ 754 ms za τr = 50 ms, τa = 250 ms (izmereno, nije iz izvora; test_cpg_oscillates_and_tau_a_lengthens_period; za τa = 400 ms ≈ 1056 ms).

## 4. Jednačine
τr x_i' = −x_i − β v_i − a y_j + s; τa v_i' = −v_i + y_i; y_i = max(0, x_i) [Matsuoka1985; Matsuoka1987; Brown1911; MarderCalabrese1996].

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| a | 2.5 | 1 | ASSUMPTION: zadovoljava uslov oscilovanja iz Matsuoka1985 | speculative |
| beta | 2.5 | 1 | ASSUMPTION: oblik modela iz Matsuoka1985, vrednost izabrana | speculative |
| tau_r | 50 | ms | ASSUMPTION: skala vremena, oblik modela iz Matsuoka1985 | speculative |
| tau_a | 250 | ms | ASSUMPTION: skala vremena, oblik modela iz Matsuoka1985 | speculative |
| s | 1 | 1 | ASSUMPTION: tonski ulaz | speculative |

## 6. Klasa pouzdanosti celog modela
**strongly_supported** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Princip half-centra je dobro potkrepljen; konkretan rate model je apstrakcija.

## 7. Ulazi, izlazi i stanje
Ulaz: s (tonski), opciono povratna sprega. Izlaz: y1, y2. Stanje: x, v po polucentru.

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: spajkovi, stvarne ćelijske struje (Ih, perzistentni Na⁺), senzorna povratna sprega. Pogađa robustnost ritma i fazno resetovanje.

## 9. Kako se kasnije zamenjuje
Spajkujući HH polucentar sa GABA_A.

## 10. Validacioni pragovi
- korelacija y1 i y2 < −0.3
- istovremena aktivnost < 20% vremena
- ≥ 3 ciklusa u 5 s
- kontrola a = 0 nema ritam (std < 1e-3)

## 11. Implementacija
Putanja i test: `neural/mechanisms/cpg.py`, `tests/mechanisms/test_plasticity_misc.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Uslov oscilovanja 1 + τr/τa < a < 1 + β: TBD-verify ([Matsuoka1985]). Mapiranje rate modela na spajkujući CPG organizma (`neural/organism` koristi spajkujuću varijantu: LIF + adaptacija + GABA_A, E_GABA = −75 mV).
