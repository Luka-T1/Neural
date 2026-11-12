# Dimension escape: dims=3, T=10000.0 ms, dt=0.1 ms, N seedova=3

| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |
|---|---|---|---|---|---|---|
| full | 3 | 3 | 1.00 | 0.44–1.00 | 902 | 879–1245 |
| random_motor | 3 | 0 | 0.00 | 0.00–0.56 | - | - |
| frozen | 3 | 3 | 1.00 | 0.44–1.00 | 902 | 891–1584 |
| no_da | 3 | 3 | 1.00 | 0.44–1.00 | 1.51e+03 | 1193–1673 |

## Poređenja (full vs kontrola)

| kontrola | Fisher p | MWU p | Holm p |
|---|---|---|---|
| random_motor | 0.1 | 0.0636 | 0.3 |
| frozen | 1 | 0.7 | 1 |
| no_da | 1 | 0.7 | 1 |

## DA leakage audit (Spearman)

| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |
|---|---|---|---|
| full | 339 | -0.215 | 0.295 |
| frozen | 409 | -0.128 | 0.366 |

## Sekundarne metrike (srednje po seedu)

| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |
|---|---|---|---|---|---|
| full | 0.671 | -0.128 | 1.47 | 0.0503 | 0.0532 |
| random_motor | 0.175 | -0.00817 | 0.506 | 0 | 0 |
| frozen | 0.893 | -0.545 | 1.46 | 0.0503 | 0.0503 |
| no_da | 0.83 | -0.595 | 1.73 | 0.0503 | 0.0494 |
