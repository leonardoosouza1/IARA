# Resultados (log vivo)

Cada experimento: setup → números → interpretação. Custo 0.14% round-trip,
thr 0.6, consenso + non-overlapping, salvo nota. net/trade e net-win são as
métricas honestas; o composto full-capital é irreal (ver METHODOLOGY §6).

---

## Phase 1 — descoberta da falha do label (trend, NÃO tradeable)
Modelo rotulado em slope de EMA (`compute_trend`): 82-89% "acurácia" vs trend-label,
mas só **47%** vs direção real do preço → P&L OOS **net -97%** (vs buy&hold +12.86%).
Causa: EMA defasada. **Lição: acurácia em alvo suavizado ≠ tradeabilidade.**

## Phase 2 — relabel no retorno real (fix)
`--label-mode return`. Treino → val AUC 0.61, acc 57%. Backtest realista
(non-overlapping + consenso): OOS net-win 82.9%, +0.28%/trade.
**Lição: gate de consenso + non-overlapping são obrigatórios** (sem eles: 50% dir, net -65%).

## Phase 3 — robustez por regime (in-sample, por ano)
net-win por ano: 2021 84.9%, 2022(bear) 84.4%, 2023 78.5%, 2024 82.8%, 2025 79.8%,
2026 78.9%. Edge positivo em bull E bear. (Caveat: in-sample.)

## Phase 4 — split limpo (treino ≤2025) + cross-coin
- BTC-only → BTC 2026 (holdout real): net-win 82.3%, +0.29%/trade, NET vs buy&hold -12%.
- BTC-only → ETH 2026 (moeda nunca treinada): net-win 82.2%, +0.42%/trade, ETH buy&hold -33%.
**Lição: edge generaliza entre moedas (features scale-invariant).**

## Phase 5 — modelo multi-coin (BTC+ETH)
3-classe, 137k samples, AUC 0.62. → BTC 2026 net-win 75.7% (dir 56%, +108%),
ETH 2026 net-win 79.9%. Multi-coin **subiu acurácia direcional** (56% vs 52-53%).

## Phase 6 — held-out coin (CONCLUSIVO)
3-coin model (BTC+ETH+SOL 2021-25), **BNB segurado fora**. Backtest 2026 (todos caíram):

| Coin | treinado? | dir | net-win | net/trade | buy&hold |
|---|---|---|---|---|---|
| **BNB** | **NÃO** | 56.5% | **87.0%** | +0.34% | -23.8% |
| BTC | sim | 56.6% | 83.0% | +0.29% | -12.0% |
| ETH | sim | 56.3% | 87.8% | +0.58% | -33.4% |
| SOL | sim | 55.3% | 88.7% | +0.58% | -30.4% |

**BNB nunca treinada em linha com treinados.** Padrão é coin-agnóstico.

## Phase 7 — walk-forward (gold-standard) + position sizing
Janela expandindo, cada ano um holdout real (modelo só vê anos anteriores).
M5 base, risk_frac 0.1, custo 0.14% round-trip, consenso + non-overlap.

| Fold | train | net-win | net/trade | Sharpe | maxDD | buy&hold | trades |
|---|---|---|---|---|---|---|---|
| 2024 | ≤2023 | 79.05% | +0.304% | 14.97 | -0.28% | +118.89% (bull) | 754 |
| 2025 | ≤2024 | 75.52% | +0.219% | 12.27 | -0.41% | -6.33% (queda) | 825 |
| 2026 | ≤2025 | 81.25% | +0.281% | 15.69 | -0.14% | -12.04% (queda) | 240 |

**net-win 75-81% em TODO fold out-of-sample, bull E queda.** net/trade +0.22-0.30%
consistente. Plots: `plots/BTCUSD_walkforward_{equity,bars}.png`.

⚠️ Sharpe 12-16 e maxDD <0.5% são **idealizados** (fills no close exato, sem
slippage proporcional nem latência). Número de produção será menor — ver Phase 8.

## Phase 8 — baseline momentum (honesty check) + custos realistas

**Modelo vs momentum puro** (sign de ROC), mesmo 2026 holdout, mesmas barras/custos/gate:

| estratégia | trades | net-win | net/trade | Sharpe | equity (f=0.1) |
|---|---|---|---|---|---|
| **modelo** | 240 | 81.25% | +0.281% | 15.69 | **+6.97%** |
| momentum puro | 1988 | 42.25% | -0.096% | -12.62 | **-17.33%** |

**Momentum sozinho PERDE dinheiro** (net-win 42%, -17%). Modelo lucra. **Edge NÃO é
momentum reembrulhado.** Modelo agrega (a) direção melhor que ROC e (b) seletividade
(opera 240 barras vs 1988 — sabe quais valem). `--baseline momentum` em pnl_backtest.

> Nuance: momentum opera toda barra golden (sem filtro de confiança), modelo filtra.
> Comparação não é trade-count-matched, mas momentum negativo vs modelo positivo é claro.

### Slippage realista (∝ volatilidade) — 2026 holdout
`--slippage_atr_mult` adiciona slippage = mult × ATR-fraction por lado.

| slippage (×ATR/lado) | net-win | net/trade | equity (f=0.1) | buy&hold |
|---|---|---|---|---|
| 0 (só fixo 2bps) | 81.25% | +0.281% | +6.97% | -12.04% |
| 0.25 | 65.42% | +0.171% | +4.18% | -12.04% |
| 0.50 | 55.42% | +0.060% | +1.46% | -12.04% |

**Edge real mas fino e slippage-sensível.** Sobrevive slippage moderado, mas a
0.5×ATR/lado fica quase break-even (+0.06%/trade). Bate buy&hold em todos os casos.
Conclusão honesta: edge genuíno (bate momentum, walk-forward, buy&hold) porém pequeno
— captura exige execução de baixo slippage. Não é máquina de dinheiro.

⏳ Pendente: Sharpe deflacionado por nº de configs; rede cross-asset.
