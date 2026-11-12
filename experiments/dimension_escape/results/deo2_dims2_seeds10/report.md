# Dimension escape: dims=2, T=60000.0 ms, dt=0.1 ms, N seedova=10

| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |
|---|---|---|---|---|---|---|
| full | 10 | 10 | 1.00 | 0.72–1.00 | 865 | 843–1073 |
| random_motor | 10 | 2 | 0.20 | 0.06–0.51 | 1.43e+04 | 12327–16205 |
| frozen | 10 | 10 | 1.00 | 0.72–1.00 | 1.04e+03 | 847–1937 |
| no_da | 10 | 10 | 1.00 | 0.72–1.00 | 1.04e+03 | 847–1937 |

## Poređenja (full vs kontrola)

| kontrola | Fisher p | MWU p | Holm p |
|---|---|---|---|
| random_motor | 0.0007145 | 0.0001107 | 0.002143 |
| frozen | 1 | 0.4055 | 1 |
| no_da | 1 | 0.4055 | 1 |

## DA leakage audit (Spearman)

| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |
|---|---|---|---|
| full | 1208 | -0.148 | 0.32 |
| frozen | 1586 | -0.106 | 0.383 |

## Sekundarne metrike (srednje po seedu)

| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |
|---|---|---|---|---|---|
| full | 0.825 | -0.101 | 1.53 | 0.0499 | 0.0563 |
| random_motor | 0.108 | 0.0181 | 0.529 | 0 | 0 |
| frozen | 0.791 | 0.0424 | 1.45 | 0.0499 | 0.0499 |
| no_da | 0.791 | 0.0424 | 1.45 | 0.0499 | 0.0499 |
