# Specifikacija: hh

## 1. Naziv
Hodgkin–Huxley membrana (jedan kompartment)

## 2. Biološka svrha i fenomen
Generisanje akcionog potencijala naponski zavisnim Na⁺ i K⁺ provodnostima.

## 3. Vremenska skala
Spajk ~1 ms; kapije m ~0.1–1 ms, h i n ~1–10 ms.

## 4. Jednačine
C_m dV/dt = I_ext − ḡ_Na m³h (V−E_Na) − ḡ_K n⁴ (V−E_K) − ḡ_L (V−E_L); dx/dt = α_x(1−x) − β_x x, x∈{m,h,n}; α/β izrazi iz [HH1952] pomereni za −65 mV (moderna konvencija).

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| E_Na | 50 | mV | HH1952 | established |
| E_K | −77 | mV | HH1952 | established |
| E_L | −54.387 | mV | HH1952 | established |
| g_Na | 120 | mS/cm² | HH1952 | established |
| g_K | 36 | mS/cm² | HH1952 | established |
| g_L | 0.3 | mS/cm² | HH1952 | established |
| C_m | 1 | µF/cm² | HH1952 | established |
| temperature | 6.3 | °C | HH1952 | established |
| Q10 | 1.0 | 1 | ASSUMPTION: bez temperaturnog skaliranja po defaultu | speculative |
| area_cm2 | 1e-3 | cm² | ASSUMPTION: površina za konverziju 1 nA → 1 µA/cm² | speculative |

V_rest ≈ −65 mV je izvedena veličina (posledica E_Na, E_K, E_L i provodnosti), a ne parametar; proverava je tests/mechanisms/test_spec_thresholds.py::test_hh_rest_minus65.

## 6. Klasa pouzdanosti celog modela
**established** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Za akson lignje na 6.3 °C; za sisare je samo kvalitativno.

## 7. Ulazi, izlazi i stanje
Ulaz: I_ext (µA/cm²). Izlaz: V (mV), spajkovi. Stanje: V, m, h, n.

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: 6.3 °C bez Q10, akson lignje, samo Na⁺ i K⁺ kanali, jedan kompartment, bez dendrita. Pogađa f-I krivu, adaptaciju, dendritsku integraciju.

## 9. Kako se kasnije zamenjuje
Markov modeli kanala, višekompartmentni model, kanali sisara.

## 10. Validacioni pragovi
- mirovanje −65 ± 0.5 mV
- vrh > +20 mV
- amplituda > 80 mV od mirovanja
- poluširina 0.5–2.5 ms (pulsni stimulus 40 µA/cm², 1 ms)
- I = 1 µA/cm² ne daje spajk
- I = 12 µA/cm² daje ≥ 8 spajkova u 200 ms
- f-I za I ∈ {12, 20, 35, 50} neopadajuća
- dt ∈ {0.01, 0.025, 0.05}: broj spajkova ±1, prvi spajk ±0.5 ms, vrh ±3 mV naspram dt = 0.01

## 11. Implementacija
Putanja i test: `neural/mechanisms/hh.py`, `tests/mechanisms/test_hh.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Kvantitativno poređenje sa slikama iz [HH1952] nije urađeno. Q10 korekcija.
