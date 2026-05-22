# Metodologia

Documento de método reproduzível. Toda execução roda **dentro do container**
`saci-brain` (PyTorch tunado p/ ROCm RX 6750 XT) — nunca no host OS.

## 1. Dados

| Coin | Origem | Granularidade | Cobertura |
|---|---|---|---|
| BTCUSD | histórico local | M1 | 2021-03 → 2026-05 |
| ETHUSD | histórico local | M1 | 2021-02 → 2026-02 |
| SOLUSDT | Binance klines | M5 | 2021-03 → 2026-05 |
| BNBUSDT | Binance klines | M5 | 2021-03 → 2026-05 |

Colunas: `date, open, high, low, close, volume`. Split temporal por
`scripts/split_temporal.py` → `{SYM}_TR` (≤ cutoff) / `{SYM}_BT` (> cutoff).
Convenção do estudo: **treino 2021-2025, backtest 2026** como holdout real.

## 2. Features (12 core, scale-invariant)

`resample_data` (M1→M5/M15/M30/H1) + `compute_features` (indicators.py). O modelo
consome só as **12 core**: RSI, MACD, MACD-signal, MACD-hist, BB-width, BB-%b,
ATR, Stoch-K, Stoch-D, Velocity(ROC), ADX, Sentiment(EMA-dev). Todas em unidade
relativa/% → **invariantes a escala de preço** (chave da generalização entre
moedas). MACD normalizado por preço (`macd/close`) — bug histórico que estourava
em BTC (~60k) foi corrigido. OHLC bruto guardado no fim só p/ P&L.

## 3. Label = sinal do retorno real (tradeable)

`feature_factory --label-mode return`. Para cada barra de decisão, horizonte
`lead_steps=12` (1h em M5): label = **0 (LONG) se preço sobe, 1 (SHORT) se cai**,
nas barras que passam consenso + |retorno| ≥ threshold ("golden bars").

> NÃO usar `--label-mode trend` (legado): rotula sinal de slope de EMA, que é
> defasado/autocorrelacionado → 89% de "acurácia" mas P&L -97%. Ver RESULTS § Phase 1.

## 4. Modelo: SwarmBrain v9

`src/brain/train_v9.py`. Workers CfC por timeframe (M5/M15/M30/H1/H4, downsample
de x[:, :12]) + self-attention + coordinator (fusão + policy head 3-classe).
input_dim auto-detect = 12. seq_len 128, batch 1024. Métrica de seleção =
macro/binary AUC (corrigida; a antiga estava quebrada e congelava no epoch 1).

Multi-coin: merge dos shards `.pt` de vários símbolos em um cache dir, treina nele.

## 5. Regra de trading (backtest realista)

`scripts/pnl_backtest.py`. Reconstrói a representação EXATA do treino e:
- **Gate de consenso** (`--consensus`): só opera barras com H1 trend == M5 trend
  e |retorno| ≥ move_thr. SEM esse gate o edge some (50% dir, P&L negativo).
- **Não-sobreposto**: 1 posição por vez. Entra no signal, segura `horizon`
  barras, sai, só então re-entra. (Operar toda janela paga custo milhares de
  vezes → ruína mesmo com direção break-even.)
- **Custos**: `(fee_bps + slippage_bps)` por lado, nas duas pontas. Default
  cripto: 5 + 2 bps → 0.14% round-trip.
- **Gate de fidelidade**: imprime acurácia direcional; tem que bater os números
  do eval, senão a reconstrução de feature divergiu e o P&L é inválido.

## 6. Métricas

- **net-win**: fração de trades com retorno líquido > 0 (após custo).
- **net/trade**: retorno líquido médio por trade (métrica de edge mais honesta).
- **directional acc**: acerto de direção nas chamadas L/S (piso ~50%).
- **Sharpe / maxDD / equity**: com position sizing fixo (em implementação) — o
  retorno composto full-capital é irreal e NÃO deve ser citado como performance.
- Baseline obrigatório: **buy & hold** no mesmo período.

## 7. Validação (níveis, do fraco ao forte)

1. Val temporal in-sample (holdout dentro do range de treino).
2. OOS no tempo (2026, modelo treinado só ≤2025). ✅
3. Robustez por regime (net-win por ano, incl. bear 2022). ✅
4. OOS na moeda (coin nunca treinada — BNB). ✅
5. **Walk-forward** (retreina em janelas deslizantes, cada ano um holdout). ⏳
6. Position sizing realista + Sharpe. ⏳

## Reprodução

```bash
# tudo via: docker exec saci-brain python -m <módulo>
# 1. split treino/backtest
python -m scripts.split_temporal --src data/processed/BTCUSD.parquet \
  --extra data/processed/BTCUSDFRESHM1.parquet --train-end 2025-12-31 --out BTCUSD2125
# 2. shard com label tradeable
python -m src.data.feature_factory --symbol BTCUSD2125_TR --market-type crypto \
  --seq-len 128 --stride 2 --label-mode return
# 3. treina
python -m src.brain.train_v9 --symbol BTCUSD2125_TR --epochs 80 --patience 25 \
  --batch_size 1024 --seq_len 128
# 4. backtest com custos
python -m scripts.pnl_backtest --model logs/models/best_model_v9_BTCUSD2125_TR.pth \
  --data data/processed/BTCUSD2125_BT.parquet --threshold 0.6 --consensus
```
