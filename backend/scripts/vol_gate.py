"""Lens 5 payoff: a forward-looking, causal volatility gate.

The intelligence (Lens 4+5): edge scales with volatility, and volatility is
predictable (persists). So gate trades by a CAUSAL volatility forecast — trailing
realized vol known strictly BEFORE entry — and check if it lifts realized edge
without lookahead. Contrast with EDGE_FINDER's ATR slice, which used the trade's
OWN-bar ATR (contemporaneous, not a forecast).

Gates (all causal — use only info available at entry):
  - vol_fc : trailing-24h mean |1h ret| ENDING before entry > its expanding median
  - us     : entry hour in [13,20) UTC
  - both   : vol_fc AND us

Run in container:
  docker exec saci-brain python -m scripts.vol_gate
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl

from scripts.market_intel import hourly_logret

ASSETS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
RESULTS = "/app/docs/study/results"


def vol_forecast(stem: str) -> pl.DataFrame:
    """Trailing-24h mean |ret| shifted by 1 bar (known strictly before the bar),
    plus an expanding-median flag (also causal: median of vols seen so far)."""
    s = hourly_logret(stem).sort("date")
    a = np.abs(s["r"].to_numpy())
    n = len(a)
    trail = np.full(n, np.nan)
    W = 24
    csum = np.concatenate([[0], np.cumsum(a)])
    for t in range(W, n):
        trail[t] = (csum[t] - csum[t - W]) / W  # mean |ret| over [t-W, t)
    # shift by 1 so the value is known BEFORE the bar's return is realized
    trail = np.concatenate([[np.nan], trail[:-1]])
    # expanding median (causal): median of all trail values up to and incl. t-1
    exp_med = np.full(n, np.nan)
    seen = []
    for t in range(n):
        if not np.isnan(trail[t]):
            exp_med[t] = np.median(seen) if seen else trail[t]
            seen.append(trail[t])
    df = pl.DataFrame({"date": s["date"].to_numpy(), "vfc": trail, "vmed": exp_med})
    return df.with_columns((pl.col("vfc") > pl.col("vmed")).alias("hi_vol_fc")).drop_nulls()


def report(name, base, gated):
    nb, ng = base.height, gated.height
    if ng == 0:
        print(f"  {name:18s} n=0"); return
    wb = 100 * (base["net"] > 0).mean(); pb = 100 * base["net"].mean(); tb = 100 * base["net"].sum()
    wg = 100 * (gated["net"] > 0).mean(); pg = 100 * gated["net"].mean(); tg = 100 * gated["net"].sum()
    print(f"  {name:18s} n={ng:4d}/{nb:<4d} ({100*ng/nb:3.0f}%)  "
          f"net-win {wg:5.1f}% (Δ{wg-wb:+.1f})  net/trade {pg:+.3f}% (Δ{pg-pb:+.3f})  total {tg:+.1f} vs {tb:+.1f}")


def main():
    import sys
    pat = sys.argv[1] if len(sys.argv) > 1 else f"{RESULTS}/trades_*_2026.csv"
    files = sorted(glob.glob(pat))
    vf = {c: vol_forecast(v) for c, v in ASSETS.items()}
    name_map = {"BTC": "BTC", "ETH": "ETH", "SOL": "SOL", "BNB": "BNB"}

    pooled = []
    print("=== causal volatility gate (per coin, 2026) ===")
    for f in files:
        coin = f.split("trades_")[-1].split("_")[0]
        d = pl.read_csv(f, try_parse_dates=True).with_columns(
            pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        d = d.select(["entry", "net", "hour"]).sort("entry")
        d = d.join_asof(vf[name_map[coin]].sort("date"), left_on="entry", right_on="date",
                        strategy="backward").drop_nulls("hi_vol_fc")
        d = d.with_columns(coin=pl.lit(coin))
        pooled.append(d)
        print(f"\n{coin}:")
        report("baseline", d, d)
        report("vol_fc gate", d, d.filter(pl.col("hi_vol_fc")))
        report("US gate", d, d.filter((pl.col("hour") >= 13) & (pl.col("hour") < 20)))
        report("both", d, d.filter(pl.col("hi_vol_fc") & (pl.col("hour") >= 13) & (pl.col("hour") < 20)))

    P = pl.concat(pooled)
    print("\n=== POOLED (all coins) ===")
    report("baseline", P, P)
    report("vol_fc gate", P, P.filter(pl.col("hi_vol_fc")))
    report("US gate", P, P.filter((pl.col("hour") >= 13) & (pl.col("hour") < 20)))
    report("both", P, P.filter(pl.col("hi_vol_fc") & (pl.col("hour") >= 13) & (pl.col("hour") < 20)))
    print("\nNote: vol_fc uses ONLY trailing-24h vol known before entry (causal forecast),"
          "\nnot the trade's own-bar ATR. This is a deployable filter, not hindsight.")


if __name__ == "__main__":
    main()
