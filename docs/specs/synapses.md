# Specifikacija: synapses

## 1. Naziv
Dvoeksponencijalne provodne sinapse (AMPA, NMDA, GABA_A)

## 2. Biološka svrha i fenomen
Hemijska sinaptička transmisija preko provodnosti sa konačnim usponom i opadanjem.

## 3. Vremenska skala
AMPA ~2 ms, GABA_A ~6 ms, NMDA ~100 ms.

## 4. Jednačine
g(t) = g_max · (e^{−t/τ_d} − e^{−t/τ_r}) / norm, norm = vrednost u vrhu; I = g·(V − E) [nS·mV = pA], pozitivno = izlazna. NMDA: B(V) = 1/(1 + e^{−0.062V}[Mg]/3.57) [JahrStevens1990].

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| ampa.tau_r | 0.2 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| ampa.tau_d | 2.0 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| ampa.E | 0 | mV | Destexhe1998 | strongly_supported |
| nmda.tau_r | 2.0 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| nmda.tau_d | 100 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| nmda.E | 0 | mV | Destexhe1998 | strongly_supported |
| gaba_a.tau_r | 0.5 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| gaba_a.tau_d | 6.0 | ms | Destexhe1998; ASSUMPTION: tačna vrednost zavisi od preparata | strongly_supported |
| gaba_a.E | −70 | mV | Destexhe1998; ASSUMPTION: tipična vrednost, organizam koristi −75 | strongly_supported |
| Mg | 1.0 | mM | JahrStevens1990 | established |
| g_max | 1.0 | nS | ASSUMPTION: jedinična težina | speculative |

## 6. Klasa pouzdanosti celog modela
**strongly_supported** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Kinetika receptora; konkretne vrednosti zavise od preparata.

## 7. Ulazi, izlazi i stanje
Ulaz: spajk presinaptičkog neurona, V postsinaptičkog. Izlaz: I (pA). Stanje: g (po sinapsi, ili dve pomoćne promenljive).

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: stohastičnost oslobađanja, kratkoročna plastičnost, desenzitizacija, prostorna difuzija, dinamika Cl⁻. Pogađa varijabilnost odgovora i dinamiku pri visokim frekvencijama.

## 9. Kako se kasnije zamenjuje
Markovljevi modeli receptora, stohastičko oslobađanje, Tsodyks–Markram kratkoročna plastičnost.

## 10. Validacioni pragovi
- vrh = g_max ± 2%
- I(E_rev) = 0
- B(−65) prema formuli
- NMDA sporije od GABA_A, a GABA_A sporije od AMPA

## 11. Implementacija
Putanja i test: `neural/mechanisms/synapses.py`, `tests/mechanisms/test_plasticity_misc.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Tačne τ vrednosti iz [Destexhe1998]. E_GABA = −70 vs −75 mV.
