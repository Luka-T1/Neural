# Dimension escape: dims=3, T=60000.0 ms, dt=0.1 ms, N seedova=10

| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |
|---|---|---|---|---|---|---|
| full | 10 | 10 | 1.00 | 0.72–1.00 | 899 | 862–1144 |
| random_motor | 10 | 2 | 0.20 | 0.06–0.51 | 1.81e+04 | 18033–18102 |
| frozen | 10 | 10 | 1.00 | 0.72–1.00 | 891 | 849–2022 |
| no_da | 10 | 10 | 1.00 | 0.72–1.00 | 891 | 849–2022 |

## Poređenja (full vs kontrola)

| kontrola | Fisher p | MWU p | Holm p |
|---|---|---|---|
| random_motor | 0.0007145 | 0.0001107 | 0.002143 |
| frozen | 1 | 1 | 1 |
| no_da | 1 | 1 | 1 |

## DA leakage audit (Spearman)

| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |
|---|---|---|---|
| full | 1385 | -0.0846 | 0.393 |
| frozen | 1561 | -0.122 | 0.407 |

## Sekundarne metrike (srednje po seedu)

| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |
|---|---|---|---|---|---|
| full | 0.779 | -0.164 | 1.55 | 0.0499 | 0.0582 |
| random_motor | 0.0969 | -0.00385 | 0.532 | 0 | 0 |
| frozen | 0.727 | -0.106 | 1.64 | 0.0499 | 0.0499 |
| no_da | 0.727 | -0.106 | 1.64 | 0.0499 | 0.0499 |

## Roll φ (iz probe snapshot-a; nije u Observation)

| uslov | n | srednje φ(T) (rad) | srednje |φ(T)| (rad) | max |φ(T)| (rad) |
|---|---|---|---|---|
| full | 10 | -0.0593 | 0.352 | 1.25 |
| random_motor | 10 | -1.66 | 12.2 | 28.1 |
| frozen | 10 | 0.0732 | 0.37 | 0.987 |
| no_da | 10 | 0.0732 | 0.37 | 0.987 |
