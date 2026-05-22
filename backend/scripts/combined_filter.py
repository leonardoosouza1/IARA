"""Combined-filter backtest with overfit guard.

EDGE_FINDER showed the model's edge concentrates in high-volatility + US-session +
high-confidence trades. This script tests the COMBINED filter and — critically —
guards against overfit by deriving the only fitted threshold (the ATR quartile
cutoff) on one set of coins and applying it to a DISJOINT set.

Filter conditions:
  - high vol : atr >= q-th percentile (q from --atr_q, default 75) — FITTED
  - US sess  : hour in [--us_start, --us_end) UTC                  — fixed
  - high conf: conf >= --conf_thr                                  — fixed

Run in container:
  docker exec saci-brain python -m scripts.combined_filter docs/study/results/trades_*_2026.csv
"""
from __future__ import annotations

import sys
import glob

import numpy as np
import pandas as pd


def stats(df):
    if len(df) == 0:
        return 0, float("nan"), float("nan"), float("nan")
    nw = 100 * (df["net"] > 0).mean()
    npt = 100 * df["net"].mean()
    tot = 100 * df["net"].sum()
    return len(df), nw, npt, tot


def line(label, df):
    n, nw, npt, tot = stats(df)
    print(f"{label:32s} n={n:5d}  net-win={nw:5.1f}%  net/trade={npt:+.3f}%  total={tot:+.2f}")


def apply_filter(df, atr_cut, conf_thr, us_start, us_end):
    m = (df["atr"] >= atr_cut) & (df["hour"] >= us_start) & (df["hour"] < us_end) & (df["conf"] >= conf_thr)
    return df[m]


def main():
    pats = sys.argv[1:] or ["docs/study/results/trades_*_2026.csv"]
    files = []
    for p in pats:
        files += glob.glob(p)
    if not files:
        print("no trade CSVs found"); return

    # tag each trade with its coin (from filename) so we can do coin-split holdout
    frames = []
    for f in files:
        d = pd.read_csv(f)
        coin = f.split("trades_")[-1].split("_")[0] if "trades_" in f else f
        d["coin"] = coin
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["entry"] = pd.to_datetime(df["entry"], utc=True, errors="coerce")
    if "hour" not in df:
        df["hour"] = df["entry"].dt.hour

    ATR_Q, CONF, US0, US1 = 75, 0.66, 13, 20  # filter spec
    coins = sorted(df["coin"].unique())
    print(f"loaded {len(df)} trades, coins={coins}")
    print(f"filter: atr>=p{ATR_Q} (per-coin, fitted) & hour in [{US0},{US1}) UTC & conf>={CONF}\n")

    # --- 1. pooled baseline vs combined filter (in-sample atr cutoff) ---
    print("=== POOLED (in-sample atr cutoff per coin) ===")
    line("baseline (all trades)", df)
    fdf = []
    for c, sub in df.groupby("coin"):
        cut = np.percentile(sub["atr"], ATR_Q)
        fdf.append(apply_filter(sub, cut, CONF, US0, US1))
    fdf = pd.concat(fdf, ignore_index=True)
    line("combined filter", fdf)
    print(f"  kept {100*len(fdf)/len(df):.1f}% of trades\n")

    # --- 2. per-coin survival (count sufficiency) ---
    print("=== per-coin under combined filter ===")
    for c, sub in df.groupby("coin"):
        cut = np.percentile(sub["atr"], ATR_Q)
        line(f"  {c}", apply_filter(sub, cut, CONF, US0, US1))
    print()

    # --- 3. OVERFIT GUARD: fit atr cutoff on one coin set, apply to disjoint set ---
    print("=== OVERFIT GUARD (atr cutoff fit on FIT coins, applied to TEST coins) ===")
    for fit_coins, test_coins in [(coins[:2], coins[2:]), (coins[2:], coins[:2])]:
        fit = df[df["coin"].isin(fit_coins)]
        test = df[df["coin"].isin(test_coins)]
        atr_cut = np.percentile(fit["atr"], ATR_Q)  # FITTED on fit set only
        out = apply_filter(test, atr_cut, CONF, US0, US1)
        base = test
        nb = stats(base); no = stats(out)
        print(f"fit={fit_coins} → test={test_coins}  (atr_cut={atr_cut:.5f})")
        print(f"   test baseline : n={nb[0]:4d} net-win={nb[1]:5.1f}% net/trade={nb[2]:+.3f}%")
        print(f"   test filtered : n={no[0]:4d} net-win={no[1]:5.1f}% net/trade={no[2]:+.3f}%   "
              f"lift={no[2]-nb[2]:+.3f}pp, kept {100*no[0]/max(nb[0],1):.0f}%")
    print()

    # --- 4. single-condition ablation (is COMBINING better than each alone?) ---
    print("=== single-condition ablation (pooled, per-coin atr cutoff) ===")
    def per_coin_mask(fn):
        out = []
        for c, sub in df.groupby("coin"):
            cut = np.percentile(sub["atr"], ATR_Q)
            out.append(sub[fn(sub, cut)])
        return pd.concat(out, ignore_index=True)
    line("high-vol only", per_coin_mask(lambda s, cut: s["atr"] >= cut))
    line("US-session only", per_coin_mask(lambda s, cut: (s["hour"] >= US0) & (s["hour"] < US1)))
    line("high-conf only", per_coin_mask(lambda s, cut: s["conf"] >= CONF))
    line("ALL three combined", per_coin_mask(
        lambda s, cut: (s["atr"] >= cut) & (s["hour"] >= US0) & (s["hour"] < US1) & (s["conf"] >= CONF)))


if __name__ == "__main__":
    main()
