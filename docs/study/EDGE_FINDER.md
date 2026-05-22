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

## 5. Backtest do filtro combinado + guarda anti-overfit — `scripts/combined_filter.py`
Filtro: `atr ≥ p75` (por moeda, **único limiar ajustado**) **&** `hora ∈ [13,20) UTC`
**&** `conf ≥ 0.66`. 789 trades 2026 (BTC+ETH+SOL+BNB, modelo 3-coin, thr 0.6).

| corte | n | net-win | net/trade |
|---|---|---|---|
| baseline (todos) | 789 | 86.4% | +0.422% |
| **filtro combinado** | **58** | **91.4%** | **+0.902%** |

Combinado **2.1× o net/trade**, mas guarda só **7.4%** dos trades.

### Guarda anti-overfit (corte ATR ajustado em moedas DISJUNTAS)
O único parâmetro fitado (corte p75 do ATR) é estimado num conjunto de moedas e
**aplicado em moedas que não entraram no fit** → testa se o filtro generaliza ou
decora.

| fit → teste | baseline teste | filtrado teste | lift |
|---|---|---|---|
| BNB+BTC → ETH+SOL | +0.580% (319) | **+1.103%** (35) | +0.522pp, 97.1% net-win |
| ETH+SOL → BNB+BTC | +0.314% (470) | **+0.670%** (26) | +0.356pp |

O corte **transfere entre moedas** (lift positivo nos dois sentidos) → o edge de
volatilidade é estrutural, **não overfit**. (No 2º caso o net-win não sobe, 85→85;
o ganho vem de selecionar vencedores maiores, não mais acertos.)

### Ablação: combinar vale a pena vs cada condição sozinha?
| filtro | n | net-win | net/trade | total% |
|---|---|---|---|---|
| **alta-vol só** | 199 | 92.0% | +0.755% | **+150.2** |
| US-session só | 258 | 88.4% | +0.568% | +146.6 |
| alta-conf só | 327 | 91.7% | +0.540% | +176.5 |
| **3 combinados** | 58 | 91.4% | +0.902% | +52.3 |

**Veredito honesto:** combinar as 3 dá o **melhor net/trade (+0.902%)** e generaliza
out-of-sample, mas é agressivo demais — 58 trades em 4 moedas / ~5 meses ≈ 3/moeda/mês.
Frequência baixa demais pra confiar, e o **lucro total cai** (+52 vs +150 do filtro só-vol)
porque corta trades bons. **Alta-volatilidade sozinha é o sweet spot**: +0.755%/trade
(1.8× baseline), mantém 199 trades (25%), captura quase todo o edge com frequência usável.
Volatilidade era o lever monotônico no §1 — confirmado como o filtro principal.
Empilhar conf/sessão só quando quiser seletividade máxima e tolerar baixa frequência.
