# Specifikacija: stdp

## 1. Naziv
Spike-timing-dependent plasticity (parna, eksponencijalni prozor)

## 2. Biološka svrha i fenomen
Promena sinaptičke jačine u zavisnosti od vremenskog odnosa pre- i postsinaptičkog spajka.

## 3. Vremenska skala
Prozor ~10–100 ms; promena težine traje minute–sate.

## 4. Jednačine
Δt = t_post − t_pre; Δw = A₊ e^{−Δt/τ₊} za Δt > 0; Δw = −A₋ e^{Δt/τ₋} za Δt < 0; Δt = 0 → Δw = 0. Online: tragovi x_pre, y_post sa raspadom, redosled: raspad, ažuriranje zbog spajkova, inkrement traga.

## 5. Tabela parametara
| ime | vrednost | jedinica | izvor | klasa pouzdanosti |
|---|---|---|---|---|
| tau_plus | 16.8 | ms | BiPoo2001; BiPoo1998 | strongly_supported |
| tau_minus | 33.7 | ms | BiPoo2001; BiPoo1998 | strongly_supported |
| A_plus | 0.005 | 1 | Song2000; ASSUMPTION: red veličine | hypothesis |
| A_minus | 0.00525 | 1 | Song2000; ASSUMPTION: 1.05·A_plus | hypothesis |
| w_min | 0 | 1 | ASSUMPTION: tvrda donja granica | speculative |
| w_max | 1 | 1 | ASSUMPTION: tvrda gornja granica | speculative |

## 6. Klasa pouzdanosti celog modela
**strongly_supported** (skala: established / strongly_supported / incomplete / hypothesis / speculative). Prozor u kulturi hipokampusa; in vivo je složenije.

## 7. Ulazi, izlazi i stanje
Ulaz: vremena spajkova. Izlaz: Δw. Stanje: tragovi x_pre, y_post.

## 8. Šta aproksimacija gubi i koja ponašanja to pogađa
Gubi se: zavisnost od frekvencije, tripleti, zavisnost od težine, dendritska lokacija. Pogađa učenje pri visokim brzinama paljenja.

## 9. Kako se kasnije zamenjuje
Triplet pravilo, pravila zavisna od kalcijuma.

## 10. Validacioni pragovi
- znak Δw prema Δt
- fit τ ± 10%
- online pravilo = analitičko

## 11. Implementacija
Putanja i test: `neural/mechanisms/stdp.py`, `tests/mechanisms/test_plasticity_misc.py`. Prag-provere su kvalitativne; nisu poređenje sa eksperimentalnim podacima.

## 12. Otvorena pitanja (TBD-verify)
Tačne vrednosti τ± i A± iz izvora.
