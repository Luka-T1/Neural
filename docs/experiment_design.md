# Dizajn eksperimenta: „1D organizam otkriva višu dimenziju“

## (a) Pitanje i hipoteze
Pitanje: može li organizam čije su sve senzacije lokalni 1D skalari, bez ikakve informacije o pravcima ili koordinatama, kroz DA-modulisanu plastičnost naučiti motorni obrazac koji ga izvodi iz kanala u ortogonalni pravac?

- **H1** stopa bekstva u uslovu `full` je veća nego u `random_motor`.
- **H2** stopa bekstva u `full` je veća nego u `frozen`.
- **H3** stopa bekstva u `full` je veća nego u `no_da`.

## (b) Svet
Osa 0 je kanal sa mekim zidovima; ortogonalni deo x_⊥ drži žleb −κx_⊥ (dinamika je u `WorldConfig`, vlasnik ag-exec_world; ovde se ne prepisuje numerički). Ortogonalni pogon je Lévyjeva površina

Ȧ = ½ (q_A q̇_B − q_B q̇_A),

pa je potrebna **trajna kvadratura** (fazna razlika ≈ ±π/2 uz istu frekvenciju) parova A i B; jedan aktuator ne može da je proizvede. Ovo je analogon Purcell 1977 ([Purcell1977], „Life at low Reynolds number“, Am J Phys 45:3, DOI 10.1119/1.10903, doi_status='needs-verify'): nereciprocitetan ciklus u prostoru oblika.

## (c) Senzori
9 lokalnih 1D skalara. Chem zavisi samo od x0; ventralni dodir je bez znaka. Nijedan senzor ne daje smer ni koordinatu ortogonalnog dela.

## (d) Definicija bekstva (samo iz probe-a)
‖x_⊥‖ > 1.0 mm neprekidno ≥ 500 ms. `t_escape` je početak tog neprekidnog intervala. Probe vidi samo harness (`experiments/dimension_escape/harness.py`); `neural/organism` ne sme da ga spomene (ugovorni test skenira izvor). Provera se radi na koraku uzorkovanja (`--trace-every-ms`, podrazumevano 10 ms); rano zaustavljanje je dozvoljeno posle bekstva.

## (e) DA i zašto ne otkriva osu
Novost: n_t = (1/9) Σ_i |s_i − μ_i| / (σ_i + ε), gde su μ_i i σ_i eksponencijalni prosek i SD (τ = 2 s), **isti za sve kanale, bez težina po kanalu**. ε = 0.05 nA (ASSUMPTION: sprečava deljenje malim σ).
DA_phasic = g · max(0, n_t − n̄_t), uz habituaciju (n̄ je eksponencijalni prosek n_t, τ = 10 s); DA_tonic = 0.1. Ograničenje implementacije: DA_phasic ≤ 10 (ASSUMPTION, zaštita).

Signal je skalaran: nema znak ni pravac, ne zna koji je motorni obrazac bio aktivan, a pale ga i aksijalni zidovi i gradijent. Zbog habituacije stanje posle bekstva nije trajno „nagrađeno“.
**Ograničenje:** ventralni kanal je u korelaciji sa ‖x_⊥‖, pa novost može indirektno da prati kretanje u ortogonalnom pravcu. Zato je obavezan **DA leakage audit**: Spearman korelacija DA_phasic sa |ẋ_⊥| i sa |ẋ0| (uslovi `full` i `frozen`; ẋ je konačna razlika nad trace uzorcima). Rezultat je u `report.json::da_leakage_audit`. Korelacija sa |ẋ_⊥| koja je znatno veća od one sa |ẋ0| je znak curenja informacije o dimenziji.

## (f) Kontrole
- **random_motor**: premotorni izlaz je OU šum (τ = 200 ms), nezavisan po aktuatoru, mean = 1.0 nA, sd = 0.5 nA, odsečen na ≥0. Izabrana je fiksna varijanta (jednostavnija) umesto izjednačavanja sa izlazom `frozen` organizma; ova odluka je dokumentovana ovde i u `harness.py`.
- **frozen**: η = 0 (eligibility se računa, težine se ne menjaju), ostalo isto, DA i dalje računat.
- **no_da**: DA_phasic ≡ 0, tonična ostaje, dakle nemodulisani STDP (η·c·0.1). Razlikuje se od `frozen` jer se težine i dalje menjaju (kroz tonični DA), samo bez novosti kao modulatora; `frozen` odvaja efekat samog učenja, `no_da` efekat modulacije.

## (g) Statistika
Seedovi 0..N−1 (podrazumevano 20, N ≥ 10), upareni po uslovima (isti seed sveta). Računa se: Wilson 95% interval stope; medijana i IQR `t_escape` (samo kod bekstava); Fisher exact two-sided `full` vs svaka kontrola (tabela [[esc_full, ne_full],[esc_ctrl, ne_ctrl]]); Mann–Whitney two-sided na `t_escape` gde je necenzurisano = T+1; Holm korekcija za 3 poređenja pri α = 0.05. **Napomena:** Holm se u implementaciji primenjuje na Fisherove p-vrednosti (primarna mera); MWU p je izveštajna. Sekundarno: PLV i srednji sin(Δφ) između q_A i q_B (Hilbert), srednja |Ȧ| i trajektorija sprežnih težina A↔B.
Napomena o „uparenosti“: seed određuje svet, telo i organizam sa fiksnim ofsetima (0, +10 000, +20 000), ali testovi su nezavisni uzorci (Fisher/MWU, ne upareni testovi).

## (h) Proširenje 2D → 3D
Faza 1: `dims=2`. Faza 2: `dims=3`, x_⊥ je 2-D, smer n izvučen iz seeda, kriterijum ostaje ‖x_⊥‖. Opciona faza 3: roll φ rotira n, sa φ̇ ∝ tonična razlika ko-kontrakcija, pa i smer mora da se otkrije. Za dims > 3, n leži na (dims−1)-sferi. Oblici senzora (9) i motora (4) se nikad ne menjaju, pa se kod organizma ne menja. Faze 2 i 3 su zavisne od implementacije sveta (ag-exec_world).

## (i) Pre-registrovano
Neuspeh bekstva (0 bekstava u svim uslovima) je validan rezultat i izveštava se kao takav. Nema podešavanja parametara organizma posle gledanja rezultata bekstva; parametri CPG-a su podešeni samo prema autonomnom ritmu (≈1 Hz) bez sveta.

---
## (j) Faza 2 kandidat: test doprinosa DA-STDP
Status: samo specifikacija, nije implementirano i nije pokrenuto. Dinamiku sveta menja ag-exec_world tek kad se ova sekcija odobri.

**Zašto faza 1 ne razlikuje učenje.** Kriterijum ‖x_⊥‖ > 1.0 mm ne zavisi od znaka Lévyjeve površine, pa svaka trajna kvadratura urođenog CPG-a izvodi telo, bez obzira na to da li A vodi B ili B vodi A. Zato su frozen i no_da u fazi 1 jednaki full (VALIDATION.md, e66e958).

**j.1 Zadatak: bekstvo samo na jednoj strani, uz ponovljene epizode.**
- Skriveni znak s ∈ {+1, −1} zavisi od seeda: s = +1 za paran seed, −1 za neparan, pa je skup uravnotežen. s je fiksan kroz sve epizode jednog seeda. Organizam ne vidi ni seed ni s.
- Bekstvo važi samo kad je s·(x_⊥·n) > 1.0 mm neprekidno ≥ 500 ms (isti prag i isto zadržavanje kao u (d)). Ako je s·(x_⊥·n) < −0.5 mm, meki jednostrani zid iste forme i krutosti kao aksijalni (k_wall) vraća telo. Kontakt sa zidom ide samo kroz postojeće dodirne senzore; oblik je i dalje 9 senzora i 4 motora.
- Posledica: CPG sa fiksnim znakom fazne razlike Δφ(A,B) beži u ≈50% seedova i ne popravlja se. Da bi bekstvo bilo pouzdano, organizam mora da obrne znak Δφ, a to traži promenu sprežnih težina (`exc_cpg`, `cross`).
- Sesija: E = 20 epizoda × T_ep = 30 s. Između epizoda se svet resetuje novim seedom `derive_seed(seed, "world.ep<k>")`: nova početna x0 i novi izvor, a isti s i n. Takođe se poziva `body.reset()`, a brze promenljive organizma (V, adaptacija, eligibility, μ_i, σ_i, n̄) vraćaju se na početne vrednosti. Između epizoda se prenose **samo težine**. Pauza između epizoda je 2 s bez senzornog ulaza (ASSUMPTION). Posle bekstva epizoda se prekida.

**j.2 Nagrada i DA.** Nagrade nema. DA ostaje novost iz (e), bez ijedne izmene, i ne kodira rešenje: nijedan signal ne zavisi od s, a i pogrešni zid i uspešno bekstvo proizvode novost bez znaka. Obavezni audit znaka: Spearman ρ(DA_phasic, s·(ẋ_⊥·n)) u uslovima random_motor i frozen, sa 95% bootstrap CI po seedu. Ako je |ρ| ≥ 0.1 i CI ne obuhvata 0, put svet→DA nosi znak. U tom slučaju se pozitivan rezultat prijavljuje kao „učenje vođeno posledicom kroz DA koji curi znak“, a ne kao validacija H4.

**j.3 Uslovi.** full; frozen (η = 0); no_da (DA_phasic ≡ 0); da_shuffled (DA_phasic se pušta kao snimljena vremenska serija iz full runa seeda (seed+1) mod N, pa statistika ostaje ista, a kontingencija je prekinuta); random_motor (provera ispravnosti). Uparenost: isti seed znači isti s, n i iste početne položaje u svim uslovima.

**j.4 Metrika.** Primarna metrika: Δ_s = medijana log t_esc u epizodama 16–20 − medijana log t_esc u epizodama 1–5. Epizoda bez bekstva se cenzuriše kao t = T_ep + 1 s. Negativna Δ_s znači učenje. Sekundarno: Theil–Sen nagib log t_esc po epizodi; stopa bekstva u poslednjem bloku (Wilson 95%); znak i srednji sin Δφ(q_A, q_B) po epizodi (Hilbert); trajektorije w_cross i exc_cpg.
Test: Wilcoxon signed-rank, dvostrani, uparen po seedu, za full vs frozen, full vs no_da i full vs da_shuffled. Holm korekcija za 3 poređenja, α = 0.05. Izveštavaju se d_z i Hodges–Lehmann procena sa 95% CI.

**j.5 Pre-uslov dizajna** (proverava se pre H4, samo na podacima iz frozen). Stopa bekstva u frozen u epizodama 1–5 mora biti između 0.2 i 0.8 (Wilson CI). Ako je ≥ 0.9, CPG sam rešava zadatak i rezultat je „zadatak nevalidan“, a ne „H4 opovrgnuta“. Ako je 0 bekstava u svim uslovima, to se prijavljuje kao validan neuspeh, kao u (i). Parametri sveta smeju da se podese samo na pilotu sa seedovima 1000–1009, samo u uslovu frozen, i ti seedovi ne ulaze u potvrdni run.

**j.6 Broj seedova (ASSUMPTION).** Pretpostavlja se d_z = 0.8 za razliku Δ (full − frozen). Za α = 0.05/3 (najgori slučaj Holma), dvostrano, snagu 0.8 i ARE Wilcoxona 0.955 potrebno je N = 20. Ako je d_z = 0.5, potrebno je N = 48. Pilot daje procenu d_z, a potvrdni N = max(20, n(d_z)) upisuje se u ovu sekciju pre potvrdnog runa. Seedovi su 0..N−1. Procena trajanja: 5 uslova × N = 24 × 20 epizoda × 30 s ≈ 72 000 s simulacije. Faza 1 je trajala ≈ 0.17 s po simuliranoj sekundi, pa je ovo ≈ 3.5 h, kao gornja granica jer se epizode prekidaju posle bekstva.

**j.7 Hipoteza i opovrgavanje.** H4: Δ_full < Δ_frozen, Δ_full < Δ_no_da i Δ_full < Δ_da_shuffled, svaka sa p_holm < 0.05. Hipotezu opovrgava sledeće: (1) uz N iz j.6, Δ_full nije manja od Δ_frozen (p_holm ≥ 0.05 i HL CI obuhvata 0 ili je pozitivan); (2) full ≈ da_shuffled, pa poboljšanje dolazi od nespecifične plastičnosti, a ne od kontingencije DA; (3) full ≈ no_da, pa modulacija nije potrebna; (4) full se poboljšava, ali seedovi koji počinju sa pogrešnim znakom Δφ ne obrću znak, pa mehanizam nije onaj koji se tvrdi. Audit iz j.2 je veto na svaki pozitivan nalaz. Sekundarno (faza 2b): obrtanje s u epizodi 11. Ako postoji učenje, očekuje se skok t_esc i oporavak u epizodama 12–20, a u frozen nema oporavka.

Razmotrene alternative: promenljiva aksijalna pozicija otvora meša se sa hemotaksijom; zahtev za određenim faznim odnosom je upravo znak Δφ, sadržan u j.1.

---
## (k) Deo 2 run (e): doprinos DA-STDP na jednostranom zidu (pre-registrovano)

**Zadatak.** Preset `phase4` (dims 2, escape_sign +1, side_wall 0.5 mm) sa `--duration-ms 120000`, seedovi 0..9, četiri uslova: full, random_motor, frozen, no_da. Komanda: `python -m neural run phase4 --duration-ms 120000 --out experiments/dimension_escape/results/deo2_phase4_learn_seeds10`. Nema nagrade. DA ostaje novost iz (e), bez ikakve izmene. Ako run ne staje u vreme, trajanje se smanjuje na 90000 ms (broj seedova se ne menja) i to se upisuje u rezultate.

**H5.** U uslovu full broj bekstava ka strani znaka (`escaped_toward_sign`) veći je nego u frozen i nego u no_da, uz Fisher p_holm < 0.05 iz `report.json::comparisons`. Sekundarno: medijana t_escape i `escape_side.escaped_toward_sign` po uslovu.

**Pre-uslov** (iz run-a (d), uslov frozen, 60 s): stopa bekstva frozen mora biti u [0.2, 0.8]. Ako je ≥ 0.9, zadatak je „nevalidan kao test učenja“. Ako je 0 u svim uslovima, to je validan neuspeh, kao u (i).

**Veto.** Potpisani audit Spearman ρ(DA_phasic, Δx_perp_n) u uslovima frozen i random_motor, iz `traces/*.npz` run-a (e) (ključevi: `da`, `x_perp_n`, `t`; Δx_perp_n je prva razlika po koraku; funkcija `experiments.dimension_escape.stats.spearman`; pooled po seedovima). Ako je |ρ| ≥ 0.1, pozitivan nalaz se prijavljuje kao „DA curi znak“, a ne kao validacija DA-STDP.

**Uparena analiza.** Wilcoxon po seedu sa Hodges–Lehmann CI sledi kad funkcije iz `stats.py` budu spojene. Do tada se rezultati prijavljuju deskriptivno, sa Fisher p_holm iz `report.json`.

## (l) Rezultati Dela 2 (10 seedova po uslovu, seedovi 0..9)

Izvor: `experiments/dimension_escape/results/deo2_*_seeds10/report.json` (n=10 po uslovu u svih pet run-ova). Trajanje epizode je 60000 ms za run-ove (a)–(d) i 120000 ms za (e), kako je pre-registrovano u (k); trajanje nije menjano ni smanjivano. t_escape je medijana među onima koji su pobegli (ms). Uparena statistika je `full` naspram uslova, Fisher p (stopa) i Mann-Whitney p (t_escape), p_holm je Holm korekcija nad tri poređenja.

| run | uslov | pobegli/10 | t_escape medijana | p_holm vs full |
|---|---|---|---|---|
| (a) deo2_dims2 (2D) | full | 10 | 865 | – |
| | random_motor | 2 | 14266 | 0.0021 |
| | frozen | 10 | 1045 | 1.0 |
| | no_da | 10 | 1045 | 1.0 |
| (b) deo2_dims3 (3D) | full | 10 | 899 | – |
| | random_motor | 3 | 10371 | 0.0093 |
| | frozen | 10 | 891 | 1.0 |
| | no_da | 10 | 891 | 1.0 |
| (c) deo2_roll (3D, roll) | full | 10 | 899 | – |
| | random_motor | 2 | 18068 | 0.0021 |
| | frozen | 10 | 891 | 1.0 |
| | no_da | 10 | 891 | 1.0 |
| (d) deo2_phase4 (escape_sign) | full | 10 | 1890 | – |
| | random_motor | 0 | – | 3.2e-5 |
| | frozen | 10 | 1969 | 1.0 |
| | no_da | 10 | 1969 | 1.0 |
| (e) deo2_phase4_learn (120000 ms) | full | 10 | 1890 | – |
| | random_motor | 4 | 76481 | 0.0325 |
| | frozen | 10 | 1969 | 1.0 |
| | no_da | 10 | 1969 | 1.0 |

Tumačenje:
- Koordinisana (CPG) lokomocija je potrebna: `random_motor` je značajno gori od `full` u svih pet run-ova (p_holm 3e-5 do 0.033), i po stopi bekstva i po vremenu.
- Doprinos DA-STDP nije pokazan: `full`, `frozen` i `no_da` pobegnu 10/10, a razlike u t_escape nisu značajne (p_holm = 1.0). Čak i u run-u (e), osmišljenom da to testira, nema razlike. Rezultat je nulti, ne dokaz da DA-STDP ne doprinosi; zid je verovatno prelak da bi plastičnost bila potrebna.
- `frozen` i `no_da` daju identične brojeve (iste putanje); to je očekivano jer pri d≡0 i zamrznutim težinama dinamika ista (vidi `tests/mechanisms/test_da_stdp_no_da.py`).
- Bekstvo radi isto u 2D i 3D (a vs b), pa skrivena osa nije prepreka u ovom okruženju.
