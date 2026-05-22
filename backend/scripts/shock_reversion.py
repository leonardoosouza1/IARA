"""Post-shock mean-reversion backtest (cross-asset signal #2).

cross_asset.py found: after a big BTC move, the alts mean-revert (bounce after a
crash) rather than continue. This tests whether that is tradeable after costs.

Signal: when the leader (BTC) prints |return| over a rolling window > a quantile
threshold, take a REVERSION position on the target alt (down-shock → LONG alt,
up-shock → SHORT alt), hold `horizon` bars, exit. Non-overlapping, cost-aware,
realistic sizing. Reports per target coin vs buy&hold.

Run in container:
  docker exec saci-brain python -m scripts.shock_reversion --leader BTC \
    --targets ETH,SOL,BNB --shock_q 0.99 --horizon 12
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

from src.data.feature_factory import resample_data

STEM = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}


def load_closes(base_tf="5m"):
    out = {}
    for name, stem in STEM.items():
        df = pl.read_parquet(f"data/processed/{stem}.parquet")
        df = df.rename({c: c.lower() for c in df.columns})
        m = resample_data(df, base_tf).to_pandas()[["date", "close"]].set_index("date")
        m.index = pd.to_datetime(m.index, utc=True)
        out[name] = m["close"]
    return pd.DataFrame(out).dropna()


def backtest(close_leader, close_tgt, dates, shock_win, shock_q, horizon, cost, both_sides):
    lret = np.log(close_leader).diff().fillna(0).values
    roll = pd.Series(lret).rolling(shock_win).sum().values  # cumulative leader move
    thr = np.nanquantile(np.abs(roll), shock_q)
    tgt = close_tgt.values
    n = len(tgt)
    rets, dts, wins = [], [], 0
    k = shock_win
    while k < n - horizon:
        s = roll[k]
        sig = 0
        if s < -thr:
            sig = +1                      # leader crashed → LONG alt (revert)
        elif s > thr and both_sides:
            sig = -1                      # leader spiked → SHORT alt (revert)
        if sig != 0:
            entry, exit_ = tgt[k], tgt[k + horizon]
            r = sig * (exit_ - entry) / entry - cost
            rets.append(r); dts.append(dates[k]); wins += int(r > 0)
            k += horizon
        else:
            k += 1
    rets = np.array(rets)
    if len(rets) == 0:
        return None
    f = 0.1
    eq = np.cumprod(1 + f * rets)
    span_yr = max((pd.Timestamp(dts[-1]) - pd.Timestamp(dts[0])).days / 365.25, 1e-6)
    sharpe = rets.mean() / (rets.std() + 1e-12) * np.sqrt(len(rets) / span_yr)
    bh = (close_tgt.iloc[-1] / close_tgt.iloc[0] - 1) * 100
    return dict(trades=len(rets), net_win=100 * wins / len(rets),
                net_per_trade=100 * rets.mean(), total=(eq[-1] - 1) * 100,
                sharpe=sharpe, buyhold=bh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader", default="BTC")
    ap.add_argument("--targets", default="ETH,SOL,BNB")
    ap.add_argument("--base_tf", default="5m")
    ap.add_argument("--shock_win", type=int, default=3, help="bars to accumulate the leader move")
    ap.add_argument("--shock_q", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--slippage_bps", type=float, default=2.0)
    ap.add_argument("--both_sides", action="store_true", help="also short alts after an up-shock")
    args = ap.parse_args()

    closes = load_closes(args.base_tf)
    cost = 2.0 * (args.fee_bps + args.slippage_bps) / 1e4
    leader = args.leader.upper()
    print(f"bars={len(closes)} leader={leader} shock_win={args.shock_win} q={args.shock_q} "
          f"horizon={args.horizon} cost={cost*100:.3f}% both_sides={args.both_sides}")
    print(f"{'target':6} {'trades':>7} {'net-win':>8} {'net/trade':>10} {'total':>8} {'sharpe':>7} {'buyhold':>9}")
    for t in [x.strip().upper() for x in args.targets.split(",")]:
        r = backtest(closes[leader], closes[t], closes.index.values,
                     args.shock_win, args.shock_q, args.horizon, cost, args.both_sides)
        if r is None:
            print(f"{t:6}  no trades"); continue
        print(f"{t:6} {r['trades']:>7} {r['net_win']:>7.1f}% {r['net_per_trade']:>9.3f}% "
              f"{r['total']:>+7.1f}% {r['sharpe']:>7.2f} {r['buyhold']:>+8.1f}%")


if __name__ == "__main__":
    main()
