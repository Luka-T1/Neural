# Specifikacija: da_stdp

## 1. Naziv
Dopaminom modulisan STDP sa eligibility tragom

## 2. Biološka svrha i fenomen
Rešavanje problema odložene nagrade: STDP ostavlja trag, dopamin ga pretvara u promenu težine.

## 3. Vremenska skala
τc = 1 s, τd = 200 ms; nagrada u opsegu sekundi.

## 4. Jednačine
c' = −c/τc + STDP(Δt)·δ(t − t_spike); d' = −(d − d_tonic)/τd + DA·δ; w' = c·d [Izhikevich2007].

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| tau_c | 1000 | ms | Izhikevich2007 | hypothesis |
| tau_d | 200 | ms | Izhikevich2007 | hypothesis |
| A_plus | 1.0 | 1 | Izhikevich2007 | hypothesis |
| A_minus | 1.5 | 1 | Izhikevich2007 | hypothesis |
| tau_plus | 20 | ms | Izhikevich2007 | hypothesis |
| tau_minus | 20 | ms | Izhikevich2007 | hypothesis |
| w_max | 4 | 1 | Izhikevich2007 | hypothesis |
| w_min | 0 | 1 | ASSUMPTION: tvrda donja granica | speculative |
| d_tonic | 0 | 1 | ASSUMPTION: izabrano tako da bez DA nema promene | speculative |

## 6. Klasa pouzdanosti celog modela
**hypothesis** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Veza sa biologijom je hipoteza; [Schultz1997] je samo kontekst RPE.

## 7. Ulazi, izlazi i stanje
Ulaz: spajkovi, DA(t). Izlaz: w. Stanje: c, d, w.

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: molekularni identitet traga (CaMKII, cAMP/PKA), D1/D2 razlike. Pogađa vremenski prozor nagrade.

Kontrolni uslov „bez DA“ u organizmu znači d ≡ 0, tj. d_tonic = 0 i fazni DA = 0; tada se w ne menja, dok c i dalje radi (ponašanje dokazuje `tests/mechanisms/test_da_stdp_no_da.py`). Ovo je ASSUMPTION ablacije, a ne tvrdnja iz Izhikevich2007: taj rad ne daje vrednost toničnog DA, pa mu se ona ne pripisuje.

## 9. Kako se kasnije zamenjuje
Biohemijski model signalnih kaskada.

## 10. Validacioni pragovi
- bez DA Δw = 0 tačno
- c(τc)/c0 = e⁻¹ ± 2%
- nagrada posle pre→post daje Δw > 0, posle post→pre Δw < 0
- nagrada odložena 5 s daje < 10% efekta nagrade od 0.89 s

## 11. Implementacija
Putanja i test: `neural/mechanisms/da_stdp.py`, `tests/mechanisms/test_plasticity_misc.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Veza sa biološkim eligibility tragovima. Napomena: organizam u `neural/organism` koristi DASTDP (pravila sens_exc, exc_cpg, cross), g_da = eta·1e-3, w normalizovan na w_max_nS (`neural/organism/organism.py:9-13`).
