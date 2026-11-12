# Dimension escape: dims=2, T=120000.0 ms, dt=0.1 ms, N seedova=10

| uslov | n | bekstava | stopa | Wilson 95% | medijana t_esc (ms) | IQR (ms) |
|---|---|---|---|---|---|---|
| full | 10 | 10 | 1.00 | 0.72–1.00 | 1.89e+03 | 934–2960 |
| random_motor | 10 | 4 | 0.40 | 0.17–0.69 | 7.65e+04 | 72034–80537 |
| frozen | 10 | 10 | 1.00 | 0.72–1.00 | 1.97e+03 | 1207–2605 |
| no_da | 10 | 10 | 1.00 | 0.72–1.00 | 1.97e+03 | 1207–2605 |

## Poređenja (full vs kontrola)

| kontrola | Fisher p | MWU p | Holm p |
|---|---|---|---|
| random_motor | 0.01084 | 0.0001494 | 0.03251 |
| frozen | 1 | 0.9118 | 1 |
| no_da | 1 | 0.9118 | 1 |

## DA leakage audit (Spearman)

| uslov | uzoraka | ρ(DA, |ẋ⊥|) | ρ(DA, |ẋ0|) |
|---|---|---|---|
| full | 2646 | -0.0485 | 0.343 |
| frozen | 1995 | -0.134 | 0.319 |

## Sekundarne metrike (srednje po seedu)

| uslov | PLV | srednji sin Δφ | srednja |Ȧ| | w_cross početak | w_cross kraj |
|---|---|---|---|---|---|
| full | 0.538 | 0.0943 | 1.55 | 0.0499 | 0.059 |
| random_motor | 0.0648 | 3.63e-05 | 0.523 | 0 | 0 |
| frozen | 0.596 | 0.224 | 1.62 | 0.0499 | 0.0499 |
| no_da | 0.596 | 0.224 | 1.62 | 0.0499 | 0.0499 |

## Strana bekstva (escape_sign=±1, zid na −s strani)

| uslov | n | bekstava | ka s | protiv s | srednje x⊥·n(T) |
|---|---|---|---|---|---|
| full | 10 | 10 | 10 | 0 | 2.18 |
| random_motor | 10 | 4 | 4 | 0 | 0.508 |
| frozen | 10 | 10 | 10 | 0 | 3.05 |
| no_da | 10 | 10 | 10 | 0 | 3.05 |
