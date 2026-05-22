# Engenharia: performance, qualidade, prioridades

Registro das melhorias de engenharia do estudo (separado dos resultados de
trading). Objetivo: rápido, reproduzível, sem bug silencioso.

## Feito (2026-05-21)

### 1. Pipeline de feature unificada (mata a classe de bug mais cara)
Antes: 3-4 caminhos independentes construíam o input do modelo
(`train_v9.MultiTimeframeDataset`, `backtester._simulate_timeframes`,
`pnl_backtest.build_m5`, `backtest_v9`). Qualquer divergência = bug silencioso
(foi a causa do "85% que perdia -97%").

Agora: **`src/data/model_input.py` é fonte única da verdade**:
- `compute_core12` — as 12 features que o modelo consome (RSI, MACD×3, BB×2,
  ATR, Stoch×2, Velocity, ADX, Sentiment). **Bit-identical** ao legado
  `compute_features[:, :12]` (diff 0.00e+00, travado por teste).
- `build_features(df, base_tf)` — OHLCV → (feats12, close, dates, consensus).
- `make_timeframe_dict(x_core, seq_len)` — downsample M5/M15/M30/H1/H4.

`backtester`, `pnl_backtest` e `train_v9` agora chamam essas funções. Treino e
backtest **não podem mais divergir**.

### 2. Teste de contrato (trava regressão)
`tests/test_model_input_contract.py`: core12 == legado, downsample == legado
(batched e não-batched), build_features sane. Roda em <10s. Pega feature-drift
e quebra de downsample antes de virar resultado falso.

### 3. Performance
- **Hurst 10× mais rápido** (`step=1→10` em indicators.py): era o gargalo do
  shard (~13min em 450k barras). Hurst vive em col >12, **modelo nunca usa** →
  zero impacto. Shard de M5 caiu de ~13min p/ ~1-2min.
- **core12 em vez de 26-33 features no backtest**: pula Hurst/FFT/CVD/OI/SMA/
  wicks/time. Backtest BNB completo (load + features em 528k barras + inferência)
  = **5.5s** (antes: dezenas de segundos).
- **M5 base** (M1 removido do fluxo): 1/5 das barras, mesmo sinal (cérebro usa M5).

### 4. Bug de mount corrigido
Container montava só `./backend:/app`; scripts gravavam resultados em
`/app/docs/...` (não montado) → artefatos do estudo perdidos no restart.
Adicionado `./docs:/app/docs` ao compose. Resultados/plots agora persistem.

## Prioridades ajustadas

| # | item | status |
|---|---|---|
| 1 | Unificar pipeline de feature | ✅ feito + testado |
| 2 | core12 + Hurst 10× (perf) | ✅ feito |
| 3 | Teste de contrato | ✅ feito |
| 4 | Mount docs (persistência) | ✅ feito |
| 5 | Walk-forward 3 folds (M5) | ✅ net-win 75-81% todo fold OOS |
| 6 | Baseline momentum honesto (edge é momentum?) | ✅ momentum perde -17%, modelo +7% |
| 7 | Custos realistas (slippage ∝ volatilidade) | ✅ `--slippage_atr_mult`; edge fino mas positivo |
| 8 | Slim shard layout (12+4 cols, re-shard) | ⏳ Phase 2 |
| 9 | Vetorizar loop de janelas no shard | ⏳ |
| 10 | Features incrementais O(1)/tick (só p/ live) | ⏳ live |
| 11 | Rede cross-asset (USD/BTC cai → alerta) | ⏳ próximo grande |
| 12 | Sharpe deflacionado por nº de configs | ⏳ |

## Notas de integridade
- Modelos treinados antes da unificação seguem válidos: só consumiam `[:, :12]`,
  que é bit-identical ao novo core12. Sem re-treino necessário.
- Sharpe/maxDD dos backtests são **idealizados** (fills no close exato). Não citar
  como performance de produção até ter custos/slippage realistas (item 6).
