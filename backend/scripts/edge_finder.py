"""Edge finder: where does the model's edge concentrate?

Reads one or more per-trade CSVs (from pnl_backtest --trades_csv, which now dumps
entry,net,hour,atr,conf,pred) and reports net-win + net/trade by:
  - hour of day (UTC session effects)
  - ATR (volatility) quartile
  - confidence quartile
  - direction (long/short)

If edge concentrates in a slice (e.g. high-vol hours, high-conf trades), that's a
new actionable filter. Run in container:
  docker exec saci-brain python -m scripts.edge_finder logs/trades_*.csv
"""
from __future__ import annotations

import sys
import glob

import numpy as np
import pandas as pd


def bucket(df, col, label, qcut=False, nq=4):
    print(f"\n=== by {label} ===")
    print(f"{'bucket':>14} {'n':>6} {'net-win':>8} {'net/trade':>10} {'total%':>8}")
    if qcut:
        try:
            grp = pd.qcut(df[col], nq, duplicates="drop")
        except Exception:
            grp = df[col]
    else:
        grp = df[col]
    for key, sub in df.groupby(grp, observed=True):
        nw = 100 * (sub["net"] > 0).mean()
        npt = 100 * sub["net"].mean()
        tot = 100 * sub["net"].sum()  # additive (not compounded) contribution
        ks = f"{key}"
        print(f"{ks:>14} {len(sub):>6} {nw:>7.1f}% {npt:>+9.3f}% {tot:>+7.2f}")


def main():
    pats = sys.argv[1:] or ["logs/trades_*.csv"]
    files = []
    for p in pats:
        files += glob.glob(p)
    if not files:
        print("no trade CSVs found"); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["entry"] = pd.to_datetime(df["entry"], utc=True, errors="coerce")
    if "hour" not in df:
        df["hour"] = df["entry"].dt.hour
    print(f"loaded {len(df)} trades from {len(files)} file(s)")
    print(f"overall: net-win {100*(df.net>0).mean():.1f}%  net/trade {100*df.net.mean():+.3f}%")

    bucket(df, "hour", "hour of day (UTC)")
    if "atr" in df:
        bucket(df, "atr", "ATR (volatility) quartile", qcut=True)
    if "conf" in df:
        bucket(df, "conf", "confidence quartile", qcut=True)
    if "pred" in df:
        df["dir"] = df["pred"].map({0: "LONG", 1: "SHORT"})
        bucket(df, "dir", "direction")


if __name__ == "__main__":
    main()
