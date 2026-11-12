# saci-trader: a crypto-direction research study, with its mistakes left in

saci-trader is a research codebase that asks one question: can a single neural model predict the direction of
crypto prices well enough to beat **trading costs**, out of sample in time and on coins it never saw? The code
has a PyTorch-ROCm backend (feature pipeline, the "SwarmBrain" model, backtests, 15 market-structure studies),
a FastAPI server and a Next.js dashboard.

**Bottom line, up front.** The early "85–89 % accuracy" headline was an artifact: the model was learning a
lagged trend label. After fixing the label, the backtests show a **thin edge that only appears under idealized
fills**. It shrinks to near break-even under realistic slippage. A later, independent re-examination found
**no price-only signal that pays its costs**. Nothing here has been traded live, and none of it is trading
advice.

## Highlights (what the study actually established)

- **Found and documented a label leak that faked 85–89 %.** The model scored 82–89 % against an EMA-slope trend
  label but only **47 %** against realized price direction. Out-of-sample P&L was **−97 % net** while buy & hold
  made +12.86 %. The EMA lags, so the label is predictable only after the move has already happened.
- **Five feature-contract bugs behind the "85 %" were fixed**: non-deterministic feature pruning, leakage
  through a symbol-substring match, backtest timeframes that differed from training, MACD in raw price units
  (NaN on BTC), and a 3-class metric whose AUC was stuck at 0.5. A single feature function now feeds both
  training and backtest. It is bit-identical to the legacy path (diff 0.0) and locked by a contract test.
- **After relabeling on realized returns: walk-forward net-win 75–81 %, +0.22 % to +0.30 % per trade** at
  0.14 % round-trip cost. Each year is a true holdout (BTC M5; 2024 / 2025 / 2026 folds: 79.05 % / 75.52 % /
  81.25 %, +0.304 / +0.219 / +0.281 % per trade). These assume fills at the close.
- **Realistic slippage nearly erases it.** On the 2026 holdout the edge is +0.281 % per trade with fixed costs,
  +0.171 % at 0.25×ATR per side, and **+0.060 %** at 0.5×ATR per side (net-win 55.4 %).
- **It is not momentum in disguise.** On the same 2026 bars and costs: model +6.97 % (240 trades, net-win 81.25 %)
  versus pure momentum −17.33 % (1,988 trades, net-win 42.25 %). Trade counts are not matched; the model's skill
  is **selectivity** rather than direction (directional accuracy is about 55–56 %).
- **It generalizes across coins.** A model trained on BTC+ETH+SOL (2021–25) and tested on held-out **BNB** in 2026
  gave net-win 87.0 %, +0.34 % per trade, while BNB buy & hold was −23.8 %.

## How it works

```
OHLCV M1/M5 ──► resample to multiple timeframes ──► 12 scale-invariant TA features (single model_input path)
            ──► SwarmBrain v9: per-timeframe CfC workers + coordinator ──► Long / Short / Wait
            ──► trade only on "consensus" bars (H1 trend == M5 trend) with confidence ≥ 0.6,
                one position at a time, hold 12 bars (1 h), fees + slippage
```

Method details are in [`docs/study/METHODOLOGY.md`](docs/study/METHODOLOGY.md), every experiment is in
[`docs/study/RESULTS.md`](docs/study/RESULTS.md), and the market-structure lenses (correlation/PCA, HMM
regimes, volatility forecasting and spillover, tail risk, calibration, term structure of the edge, opportunity
scan, risk sizing) are in [`docs/study/MARKET_INTELLIGENCE.md`](docs/study/MARKET_INTELLIGENCE.md). Docs are
in Portuguese.

## Results

All numbers come from backtests in `docs/study/`. Cost is 0.14 % round trip unless noted. They have **not**
been re-run independently for this README. Sharpe ratios of 12–16 in those docs come from idealized fills and
are not credible as production numbers.

| Experiment | Result | Comparison |
|---|---|---|
| Trend (EMA-slope) label, OOS | 47 % real direction, **−97 %** net | buy & hold +12.86 % |
| Return label, OOS, consensus gate | net-win 82.9 %, +0.28 % per trade | without gate: 50 % direction, −65 % net |
| Walk-forward BTC 2024 / 2025 / 2026 | net-win 79.1 / 75.5 / 81.3 %; +0.30 / +0.22 / +0.28 % per trade | buy & hold +118.9 / −6.3 / −12.0 % |
| Held-out coin BNB 2026 | net-win 87.0 %, +0.34 % per trade | BNB buy & hold −23.8 % |
| Slippage 0 / 0.25 / 0.5 × ATR per side | +0.281 / +0.171 / **+0.060 %** per trade | — |
| Causal volatility gate (prior-24 h vol > median) | +0.10 to +0.17 pp per trade across 3 yearly holdouts | cuts 60–76 % of trades, so total profit does not rise |
| Confidence calibration | ρ = 0.99, no alpha decay over 29 months | — |

## What didn't work (and why)

- **The "liquid" core was mostly dead weight.** A gradient audit found **198,658 of 691,608 parameters (28.7 %)
  with zero gradient**. The coordinator's CfC runs with seq_len = 1 and h0 = 0, which makes it an MLP. The
  long/short experts never entered the loss, one projection was never wired, and dt was always 1.
- **Multi-horizon (MTP) head: no gain.** Validation accuracy went 0.564 → 0.563 and OOS net per trade went
  +0.059 → +0.037 %. Direction at 2–4 h horizons is close to a random walk, so the extra targets add noise,
  not signal.
- **Cross-asset lead-lag: nothing to exploit.** Across 505k M5 returns, contemporaneous correlation is 0.62–0.84,
  but lead-lag at k ≥ 1 is about 0 at 5 m, 1 h and 4 h. Co-movement is instantaneous.
- **Risk-based sizing is a risk dial, not alpha.** Volatility targeting raised Sharpe slightly (+1.8) but cut
  CAGR by about 40 %.
- **Independent re-test (Aug 2026, separate code, not in this repo).** EMA features + ridge + an ensemble on
  BTC 5 m / 1 h: all three arms lost −0.6 to −1.4 bps per trade while buy & hold made +75 %. A feature search
  with an economic judge across 12 feature families: nothing crossed zero on an untouched block. Funding rate:
  BTC showed +15.3 bps walk-forward, but only 1 of 7 assets was positive, so it did not replicate.
  Conclusion: **5-minute OHLCV alone does not pay the toll; the bottleneck is data, not architecture.**

## Stack

Python 3.10+, PyTorch on ROCm (AMD RX 6750 XT, gfx1031), polars/pandas, TA-Lib, FastAPI, Next.js
(`frontend/`), TimescaleDB/Postgres, optional Ollama for the "slow system" LLM. Everything runs in Docker.

## Build & run

```bash
cp .env.example .env            # fill in your own values; never commit .env
docker compose up -d --build    # API on :8000, Postgres, Ollama

# backtest a trained checkpoint inside the container (see METHODOLOGY.md for the full pipeline)
docker compose exec saci-brain python -m scripts.pnl_backtest \
  --model logs/models/<checkpoint>.pth --data data/processed/<SYMBOL>_BT.parquet \
  --threshold 0.6 --consensus --risk_frac 0.1 --slippage_atr_mult 0.5

cd frontend && yarn install && yarn dev   # dashboard
```

Market data is not included. `backend/scripts/download_fresh_data.py` fetches OHLCV. The full reproduction
(temporal split → labeled shards → training → backtest) is in
[`docs/study/METHODOLOGY.md`](docs/study/METHODOLOGY.md#reprodução).

## Status

Research study, closed as a **negative on tradeable alpha from price data alone**. Kept public because the
method is reusable: label audit, walk-forward, held-out coin, momentum baseline, slippage stress and gradient
audit. Next step, if any: non-price data (order book, funding/open interest across many assets), tested with
the same harness.

License: TBD (no LICENSE file yet).
