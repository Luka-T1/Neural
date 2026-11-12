# Poređenje organizma sa razvojem kod beba (kvalitativno)

Status: confidence `speculative`; validirano = NE. Ovo je kvalitativna analogija zasnovana na izvorima iz
`docs/bibliography.md`. Nema kvantitativnog poređenja, a podataka o bebama u projektu nema.

## Svrha

Dokument objašnjava u kojoj meri se ponašanje organizma iz `neural/organism/organism.py` može
opisati jezikom razvojne neuronauke i gde analogija prestaje da važi. Cilj nije dokazati sličnost
sa ljudskim mozgom, nego zabeležiti šta je pozajmljeno iz literature i šta nije proveravano.

## Organizam

Organizam (`neural/organism/organism.py`) ima 205 neurona:
- 9×5 = 45 senzornih neurona (on/off po kanalu);
- 80 ekscitatornih (E) i 20 inhibitornih (I) neurona u mreži sa retkim vezama;
- 4×10 = 40 neurona CPG populacija (A1, A2, B1, B2) i 4×5 = 20 CPG interneurona, koji tvore
  spajkujući half-center oscilator.

Izmereni CPG (bez ulaza, bez plastičnosti, 3 seeda, 30000 koraka pri dt = 0.1 ms):
- frekvencija ritma ≈ 1.0 Hz;
- antifaza: corr(o0, o1) ≈ −0.92 i corr(o2, o3) ≈ −0.91 do −0.93;
- indeks ko-kontrakcije CCI = 0.10–0.16.

Izvor brojki je tabela u `docs/specs/organism_audit.md` (sekcija (c)). Test
`tests/organism/test_organism_audit.py::test_cpg_rhythm_antiphase_cocontraction` proverava samo
znake korelacija i opseg frekvencije, a ne ove konkretne vrednosti.

## Bebe

Iz bibliografije (samo ono što stoji u „Nalaz“ podstavkama):
- Koračanje novorođenčeta je ritmično i alternirajuće; tumači se kao urođeni spinalni generator
  ritma [Forssberg1985].
- Nestajanje refleksa koračanja objašnjeno je biomehanikom (masa nogu raste u odnosu na snagu),
  a ne samo kortikalnom inhibicijom [ThelenFisher1982].
- Novorođenčad imaju dva osnovna obrasca motorne aktivacije, a kasnije se dodaju novi [Dominici2011].
- Stopa udaranja nogom raste kad udarac pokreće mobil, što je kontingentno potkrepljenje
  [RoveeRovee1969].
- Razvoj motorike je samoorganizacija tela, okruženja i nervnog sistema; okvir je dinamički
  sistem [ThelenSmith1994].
- Ljudski mozak ima oko 86·10^9 neurona [Azevedo2009].

## Tabela analogija

| aspekt | beba | organizam | snaga analogije |
|---|---|---|---|
| ritam i alternacija | alternirajuće koračanje, urođeni generator ritma [Forssberg1985] | CPG half-center sa antifazom (corr ≈ −0.92) | umerena: oba su alternirajući ritam, ali organizam nema telo sa realnom masom |
| ko-kontrakcija | nije obrađena u izvorima o bebama; u bibliografiji je ko-aktivacija antagonista iz [Hogan1984] (nije studija o bebama) | CCI 0.10–0.16, mala ko-aktivacija | slaba: mereno je samo u CPG-u bez tela, bez validacije |
| učenje iz posledica | kontingentno potkrepljenje udarca [RoveeRovee1969] | DA-STDP sa dopaminskim signalom novelty | slaba do nikakva: vidi ograničenje ispod |
| razvoj ili sazrevanje | promene u toku odrastanja [Dominici2011], [ThelenSmith1994] | nema: težine i broj neurona su fiksni, bez razvojne faze | nema analogije u modelu |
| razmera | ~86·10^9 neurona [Azevedo2009] | 205 neurona (~10^2) | nema: razlika je ~10^9 puta |

## Iskreno ograničenje

- Doprinos DA učenja bekstvu NIJE validiran. U `VALIDATION.md` frozen i no_da beže jednako kao full
  (10/10, medijane 1079 i 1020 ms naspram 865 ms; p_holm = 1). Zato analogija sa učenjem iz
  posledica [RoveeRovee1969] nije podržana podacima, nego samo pretpostavljena.
- Nema anatomije: korteks, bazalne ganglije i kičmena moždina su u modelu samo imena za populacije.
- Nema razvojne plastičnosti: organizam ne prolazi kroz faze sazrevanja.
- Brojevi iz tabele CPG-a su merenja na jednom modelu sa 3 seeda; nisu poređenje sa podacima o bebama.

## Šta bi validiralo poređenje

Poređenje bi postalo kvantitativno tek kad postoje podaci o bebama za iste mere (npr. stopa
ritma pod kontingentnim potkrepljenjem) i eksperiment sa više seedova koji meri istu meru u modelu.
Do tada ostaje status validirano = NE.
