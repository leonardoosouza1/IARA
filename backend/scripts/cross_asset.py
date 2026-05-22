"""Cross-asset relationship analysis — first step of the "one network, alerts
across coins" idea: does a move in one asset (e.g. BTC) PREDICT moves in others?

Measures, on aligned M5 returns across BTC/ETH/SOL/BNB:
  1. contemporaneous return correlation matrix (how coupled they are);
  2. lead-lag: corr(LEADER_ret[t], asset_ret[t+k]) for k=0..K — a positive corr
     at k>=1 means the leader's move is an exploitable ALERT (it precedes the
     others), which is what makes a cross-asset signal tradeable vs merely
     coincident;
  3. shock response: after a big leader move (|ret| > q), the mean forward
     return of each asset over the next H bars, split by shock sign.

Honest framing: contemporaneous correlation alone is NOT tradeable (everything
moves together at once). Only a lead-lag edge (k>=1) or an asymmetric shock
response is.

Run in container:
  docker exec saci-brain python -m scripts.cross_asset --leader BTCUSDM5
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

from src.data.feature_factory import resample_data

COINS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}


def load_returns(base_tf: str = "5m") -> pd.DataFrame:
    """Aligned M5 log-returns for all coins on common timestamps."""
    series = {}
    for name, stem in COINS.items():
        df = pl.read_parquet(f"data/processed/{stem}.parquet")
        df = df.rename({c: c.lower() for c in df.columns})
        m = resample_data(df, base_tf).to_pandas()[["date", "close"]].set_index("date")
        m.index = pd.to_datetime(m.index, utc=True)
        series[name] = np.log(m["close"]).diff()
    rets = pd.DataFrame(series).dropna()
    return rets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader", default="BTC", help="leader coin (BTC/ETH/SOL/BNB)")
    ap.add_argument("--max_lag", type=int, default=6, help="bars of lead-lag to scan")
    ap.add_argument("--shock_q", type=float, default=0.99, help="quantile defining a 'shock' move")
    ap.add_argument("--horizon", type=int, default=12, help="forward bars for shock response")
    ap.add_argument("--base_tf", default="5m", help="bar size for the analysis (5m,1h,4h,1d)")
    args = ap.parse_args()
    leader = args.leader.upper()

    rets = load_returns(args.base_tf)
    print(f"aligned bars: {len(rets)}  ({rets.index[0]} → {rets.index[-1]})  leader={leader}")

    print("\n=== contemporaneous return correlation ===")
    print(rets.corr().round(3).to_string())

    print(f"\n=== lead-lag: corr({leader}_ret[t], asset_ret[t+k]) — k>=1 = {leader} LEADS ===")
    print(f"{'asset':6}" + "".join(f"  k={k:<2}" for k in range(args.max_lag + 1)))
    L = rets[leader]
    for name in rets.columns:
        if name == leader:
            continue
        row = []
        for k in range(args.max_lag + 1):
            c = L.corr(rets[name].shift(-k))   # leader[t] vs asset[t+k]
            row.append(c)
        print(f"{name:6}" + "".join(f"  {c:+.3f}" for c in row))

    print(f"\n=== shock response: after |{leader} ret| > {args.shock_q:.0%}-quantile, "
          f"mean forward {args.horizon}-bar return ===")
    thr = L.abs().quantile(args.shock_q)
    up = L > thr
    dn = L < -thr
    print(f"shock threshold |ret| > {thr*100:.3f}%   (up shocks {up.sum()}, down shocks {dn.sum()})")
    print(f"{'asset':6} {'after UP shock':>16} {'after DOWN shock':>18}")
    for name in rets.columns:
        fwd = rets[name].shift(-1).rolling(args.horizon).sum().shift(-(args.horizon - 1))
        mu_up = fwd[up].mean() * 100
        mu_dn = fwd[dn].mean() * 100
        print(f"{name:6} {mu_up:>15.4f}% {mu_dn:>17.4f}%")

    print("\nRead: lead-lag k>=1 positive + asymmetric shock response = exploitable alert.")


if __name__ == "__main__":
    main()
