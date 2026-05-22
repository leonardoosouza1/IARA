# Edge finder: onde o edge concentra (relações acionáveis)

`scripts/edge_finder.py` fatia os trades do modelo (789 trades, 2026 holdout,
BTC+ETH+SOL+BNB, 3-coin model, thr 0.6) por condição. Overall: net-win 86.4%,
+0.422%/trade. As fatias revelam estrutura forte e explorável.

## 1. Volatilidade (ATR quartil) — MONOTÔNICO, maior lever
| ATR quartil | n | net-win | net/trade |
|---|---|---|---|
| baixo | 198 | 78.8% | +0.145% |
| 2 | 197 | 83.8% | +0.265% |
| 3 | 197 | 91.4% | +0.459% |
| **alto** | 197 | **91.9%** | **+0.820%** |

Edge **escala com volatilidade**: 5.6× mais net/trade no quartil alto. O modelo
ganha onde há movimento. Bars de baixa-vol (+0.145%) mal cobrem slippage.

## 2. Hora do dia (UTC) — sessão US domina
| janela | net/trade típico |
|---|---|
| 13-16h UTC (abertura US) | +0.43 a **+0.78%** (hora 15-16 picos) |
| madrugada (0,5,21,23h) | ~+0.24% |

Edge 2-3× maior na sessão americana. Liquidez/volatilidade do horário US.

## 3. Confiança do modelo (quartil) — MONOTÔNICO, bem calibrada
| conf quartil | net-win | net/trade |
|---|---|---|
| 0.60-0.626 | 80.3% | +0.338% |
| 0.626-0.651 | 81.7% | +0.291% |
| 0.651-0.68 | 90.9% | +0.462% |
| 0.68-0.727 | **92.9%** | **+0.597%** |

Subir o threshold de confiança melhora a qualidade — a confiança do modelo é
calibrada (mais conf = mais acerto).

## 4. Direção — sem viés
LONG +0.427% (net-win 85.8%) ≈ SHORT +0.417% (87.1%). Saudável, simétrico.

## Implicação acionável
O edge não é uniforme — concentra em **alta-volatilidade + sessão-US + alta-confiança**.
Combinar esses filtros deve elevar bem o net/trade (do +0.42% médio p/ provável
>+0.6%) e tornar a estratégia **mais robusta a slippage** (Phase 8 mostrou que o
edge médio é fino). Próximo: backtest do filtro combinado (vol-quartil-alto +
conf≥0.66 + horas 13-20 UTC) com walk-forward, e checar se o trade-count restante
ainda é suficiente.

> Cuidado com overfit: são 4 condições; validar o filtro combinado em holdout
> separado / outras moedas antes de confiar.
