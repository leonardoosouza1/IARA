# Estudo: Direção de Cripto com SwarmBrain v9 (saci-trader)

Projeto de estudo **reproduzível e validado** sobre se um único modelo neural
(SwarmBrain v9) consegue prever direção de preço de criptomoedas com edge real
**após custos**, e se esse edge **generaliza entre moedas** (um cérebro, várias
cryptos).

> Status: edge validado em dados reais nunca vistos (2026) e em moeda nunca
> treinada (BNB). Em validação formal walk-forward + position sizing realista.

## A tese

1. Um modelo treinado em direção de **retorno real** (não em indicador
   suavizado) tem edge positivo após custos.
2. Features TA **normalizadas são scale-invariant** → o mesmo modelo opera
   qualquer cripto líquida sem retreinar por moeda.
3. O edge sobrevive **out-of-sample no tempo** (2026) e **out-of-sample na
   moeda** (coin nunca treinada).

## Resultado-resumo (até agora)

Backtest 2026 real (modelo treinado só 2021-2025), thr 0.6, custo 0.14% round-trip,
trading non-overlapping em barras de consenso:

| Modelo → Coin 2026 | treinado? | net-win | net/trade | buy&hold |
|---|---|---|---|---|
| 3-coin → **BNB** | **NÃO (held-out)** | 87.0% | +0.34% | -23.8% |
| 3-coin → BTC | sim | 83.0% | +0.29% | -12.0% |
| 3-coin → ETH | sim | 87.8% | +0.58% | -33.4% |
| 3-coin → SOL | sim | 88.7% | +0.58% | -30.4% |

Todos os coins **caíram** em 2026; o modelo lucrou em todos, incl. a moeda
nunca treinada. Detalhes e todos os experimentos em [RESULTS.md](RESULTS.md).

## Como funciona (resumo)

Pipeline: M1/M5 OHLCV → resample multi-timeframe → 12 features TA core
normalizadas → SwarmBrain v9 (workers CfC por timeframe + coordinator) →
3-classe Long/Short/Wait. Decisão tradeable = só barras de **consenso fractal**
(H1 trend == M5 trend) com confiança ≥ threshold, **uma posição por vez**.
Método completo em [METHODOLOGY.md](METHODOLOGY.md).

## Reproduzir

Tudo roda dentro do container `saci-brain` (PyTorch ROCm). Ver
[METHODOLOGY.md § Reprodução](METHODOLOGY.md#reprodução). Scripts:
`backend/scripts/{split_temporal,feature_factory,pnl_backtest,walk_forward,plot_results}.py`.

## Documentos do estudo

- [METHODOLOGY.md](METHODOLOGY.md) — método, labels, consenso, custos, reprodução
- [RESULTS.md](RESULTS.md) — todos os experimentos (walk-forward, cross-coin, sizing)
- [CROSS_ASSET.md](CROSS_ASSET.md) — relações entre moedas (lead-lag, choque, USD/FX)
- [EDGE_FINDER.md](EDGE_FINDER.md) — onde o edge concentra (vol, sessão, confiança) + filtro combinado
- [MARKET_INTELLIGENCE.md](MARKET_INTELLIGENCE.md) — estrutura de mercado: rede de correlação, fator PCA, regimes
- [ENGINEERING.md](ENGINEERING.md) — pipeline, contrato de input, performance

## Caveats honestos (o que ainda NÃO está provado)

- Retorno composto bruto é full-capital (irreal). Métrica crível = net/trade +
  net-win + Sharpe com **position sizing fixo** (em implementação).
- 2026 testado = ~4.7 meses, um regime de queda. Walk-forward multi-ano em curso.
- Fills idealizados (entra/sai no close da barra); sem funding/latência real.
- Edge direcional fino (~56%); o lucro vem da seleção por confiança.
