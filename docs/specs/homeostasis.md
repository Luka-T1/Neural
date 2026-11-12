# Specifikacija: homeostasis

## 1. Naziv
Multiplikativno sinaptičko skaliranje

## 2. Biološka svrha i fenomen
Održavanje ciljne frekvencije paljenja uz očuvanje relativnih težina.

## 3. Vremenska skala
Biološki sati–dani; u simulaciji ubrzano (ASSUMPTION).

## 4. Jednačine
r' = −r/τ_rate (pri spajku r += 1000/τ_rate, r u Hz); w_i ← w_i·(1 + dt·(ν_cilj − ν)/(τ_s·ν_cilj)), w ≥ w_min (multiplikativno, isti faktor za sve ulaze neurona) [Turrigiano1998; Turrigiano2008; vanRossum2000].

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| tau_rate_ms | 1000 | ms | ASSUMPTION: prozor procenjivača frekvencije | speculative |
| nu_target_Hz | 5 | Hz | ASSUMPTION: ciljna frekvencija | speculative |
| tau_s_ms | 10000 | ms | ASSUMPTION: vremensko ubrzanje; Turrigiano1998; Turrigiano2008 | incomplete |
| w_min | 0 | 1 | ASSUMPTION: donja granica | speculative |

## 6. Klasa pouzdanosti celog modela
**strongly_supported** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Fenomen je dobro potkrepljen; kvantitativni model i vremenske skale nisu.

## 7. Ulazi, izlazi i stanje
Ulaz: izmerena frekvencija, težine. Izlaz: skalirane težine. Stanje: prosek frekvencije.

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: heterogenost po sinapsi, inhibitorno skaliranje, vremenska skala. Pogađa dinamiku nakon poremećaja.

## 9. Kako se kasnije zamenjuje
Model zasnovan na kalcijumu i BDNF/TNFα signalizaciji.

## 10. Validacioni pragovi
- frekvencija konvergira ka cilju ± 5% odozdo i odozgo
- odnosi težina sačuvani do 1e-6

## 11. Implementacija
Putanja i test: `neural/mechanisms/homeostasis.py`, `tests/mechanisms/test_plasticity_misc.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Biološke vrednosti τ_s i ν_cilj (u kodu su pretpostavke, vremenski ubrzane).
