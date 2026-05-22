"""USD strength vs crypto — tests the original thesis "dollar up → crypto down",
and whether USD strength PREDICTS crypto (a tradeable lead) or just co-moves.

USD strength proxy = -mean(log-returns of EUR/GBP/AUD vs USDT). When EUR/GBP/AUD
fall against USDT, the dollar is strengthening (DXY is ~60% EUR-weighted, so this
tracks DXY reasonably). Crypto is USD-quoted, so a stronger USD is a headwind.

Reports on aligned returns:
  1. correlation USD_STR vs BTC/ETH (expect NEGATIVE if thesis holds);
  2. lead-lag corr(USD_STR[t], crypto[t+k]) — k>=1 negative = USD strength
     PREDICTS crypto down = tradeable macro alert;
  3. shock response: after a big USD_STR move, crypto forward return.

FX from Binance USDT proxies has gaps/low early-history volume; the script uses
whatever overlapping bars exist. Run in container:
  docker exec saci-brain python -m scripts.fx_crypto --base_tf 1h
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

from src.data.feature_factory import resample_data

CRYPTO = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5"}
FX = {"EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD"}  # USD-proxy pairs


def _ret(stem, base_tf):
    df = pl.read_parquet(f"data/processed/{stem}.parquet")
    df = df.rename({c: c.lower() for c in df.columns})
    m = resample_data(df, base_tf).to_pandas()[["date", "close"]].set_index("date")
    m.index = pd.to_datetime(m.index, utc=True)
    return np.log(m["close"]).diff()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_tf", default="1h")
    ap.add_argument("--max_lag", type=int, default=6)
    ap.add_argument("--shock_q", type=float, default=0.99)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--fx", default="EUR,GBP,AUD", help="which FX legs for the USD proxy (EUR has full history)")
    args = ap.parse_args()

    want_fx = {k: FX[k] for k in [x.strip().upper() for x in args.fx.split(",")] if k in FX}
    cols = {}
    for n, s in {**CRYPTO, **want_fx}.items():
        try:
            cols[n] = _ret(s, args.base_tf)
        except Exception as e:
            print(f"skip {n} ({s}): {e}")
    df = pd.DataFrame(cols).dropna()
    if len(df) < 100:
        print(f"too few aligned bars ({len(df)}) — FX/crypto overlap insufficient")
        return
    # USD strength: dollar up when EUR/GBP/AUD fall vs USDT
    fx_present = [n for n in FX if n in df.columns]
    df["USD_STR"] = -df[fx_present].mean(axis=1)
    print(f"aligned bars: {len(df)}  ({df.index[0]} → {df.index[-1]})  base_tf={args.base_tf}")
    print(f"USD strength = -mean({fx_present})")

    print("\n=== correlation (expect USD_STR vs crypto NEGATIVE) ===")
    print(df[["USD_STR", "BTC", "ETH"] if "ETH" in df else ["USD_STR", "BTC"]].corr().round(3).to_string())

    print(f"\n=== lead-lag corr(USD_STR[t], crypto[t+k]) — k>=1 NEG = USD leads crypto down ===")
    print(f"{'crypto':6}" + "".join(f"  k={k:<2}" for k in range(args.max_lag + 1)))
    for c in [x for x in ("BTC", "ETH") if x in df.columns]:
        row = [df["USD_STR"].corr(df[c].shift(-k)) for k in range(args.max_lag + 1)]
        print(f"{c:6}" + "".join(f"  {v:+.3f}" for v in row))

    print(f"\n=== shock: after |USD_STR| > {args.shock_q:.0%}-q, mean fwd {args.horizon}-bar crypto ret ===")
    thr = df["USD_STR"].abs().quantile(args.shock_q)
    up = df["USD_STR"] > thr   # USD spikes up
    dn = df["USD_STR"] < -thr  # USD drops
    print(f"USD-up shocks {up.sum()}, USD-down shocks {dn.sum()} (thr {thr*100:.3f}%)")
    print(f"{'crypto':6} {'after USD UP':>14} {'after USD DOWN':>16}")
    for c in [x for x in ("BTC", "ETH") if x in df.columns]:
        fwd = df[c].shift(-1).rolling(args.horizon).sum().shift(-(args.horizon - 1))
        print(f"{c:6} {fwd[up].mean()*100:>13.4f}% {fwd[dn].mean()*100:>15.4f}%")
    print("\nThesis 'USD up → crypto down' holds iff USD_STR↔crypto corr NEG and after-USD-UP fwd ret NEG.")


if __name__ == "__main__":
    main()
