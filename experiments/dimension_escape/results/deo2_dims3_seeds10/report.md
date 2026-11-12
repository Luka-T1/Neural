# Dimension escape: dims=3, T=60000.0 ms, dt=0.1 ms, N seedova=10

| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |
|---|---|---|---|---|---|---|
| full | 10 | 10 | 1.00 | 0.72–1.00 | 899 | 862–1144 |
| random_motor | 10 | 3 | 0.30 | 0.11–0.60 | 1.04e+04 | 9851–29800 |
| frozen | 10 | 10 | 1.00 | 0.72–1.00 | 891 | 849–2037 |
| no_da | 10 | 10 | 1.00 | 0.72–1.00 | 891 | 849–2037 |

## Poređenja (full vs kontrola)

| kontrola | Fisher p | MWU p | Holm p |
|---|---|---|---|
| random_motor | 0.003096 | 0.0001317 | 0.009288 |
| frozen | 1 | 0.7623 | 1 |
| no_da | 1 | 0.7623 | 1 |

## DA leakage audit (Spearman)

| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |
|---|---|---|---|
| full | 1113 | -0.143 | 0.295 |
| frozen | 1452 | -0.0806 | 0.348 |

## Sekundarne metrike (srednje po seedu)

| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |
|---|---|---|---|---|---|
| full | 0.787 | -0.15 | 1.61 | 0.0499 | 0.057 |
| random_motor | 0.128 | 0.0323 | 0.521 | 0 | 0 |
| frozen | 0.77 | -0.174 | 1.57 | 0.0499 | 0.0499 |
| no_da | 0.77 | -0.174 | 1.57 | 0.0499 | 0.0499 |
