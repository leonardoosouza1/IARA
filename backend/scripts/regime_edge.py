"""Lens 4: does the model's edge depend on the market regime?

Ties market structure (market_intel) to the directional model. For every 2026
trade, label the prevailing crypto-correlation regime at entry (rolling 30d mean
corr: low = coins decoupled / calm, high = risk-off / everything-as-one) and the
realized-vol regime, then measure net/trade per regime.

Run in container:
  docker exec saci-brain python -m scripts.regime_edge
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl

from scripts.market_intel import hourly_logret

CRYPTO = ["BTC", "ETH", "SOL", "BNB"]
WIN = 720  # 30d of 1h bars
RESULTS = "/app/docs/study/results"


def rolling_corr_series() -> pl.DataFrame:
    s = {c: hourly_logret({"BTC": "BTCUSDM5", "ETH": "ETHUSDM5",
                           "SOL": "SOLUSDT", "BNB": "BNBUSDT"}[c]) for c in CRYPTO}
    df = s[CRYPTO[0]].rename({"r": CRYPTO[0]})
    for c in CRYPTO[1:]:
        df = df.join(s[c].rename({"r": c}), on="date", how="inner")
    X = df.select(CRYPTO).to_numpy()
    dates = df["date"].to_numpy()
    out = np.full(X.shape[0], np.nan)
    iu = np.triu_indices(len(CRYPTO), 1)
    for t in range(WIN, X.shape[0]):
        out[t] = np.corrcoef(X[t - WIN:t].T)[iu].mean()
    return pl.DataFrame({"date": dates, "mkt_corr": out}).drop_nulls()


def bucket(df, col, label, edges=None):
    print(f"\n=== edge by {label} ===")
    if edges is None:
        df = df.with_columns(
            pl.when(pl.col(col) <= pl.col(col).quantile(0.33)).then(pl.lit("low"))
            .when(pl.col(col) <= pl.col(col).quantile(0.66)).then(pl.lit("mid"))
            .otherwise(pl.lit("high")).alias("bk"))
        keys = ["low", "mid", "high"]
    else:
        df = df.with_columns(pl.col(col).alias("bk"))
        keys = None
    g = df.group_by("bk").agg(
        pl.len().alias("n"),
        (100 * (pl.col("net") > 0).mean()).alias("netwin"),
        (100 * pl.col("net").mean()).alias("npt"))
    g = g.to_dicts()
    order = {k: i for i, k in enumerate(keys)} if keys else {}
    for row in sorted(g, key=lambda r: order.get(r["bk"], r["bk"])):
        print(f"  {str(row['bk']):>6s}  n={row['n']:4d}  net-win={row['netwin']:5.1f}%  net/trade={row['npt']:+.3f}%")


def main():
    files = sorted(glob.glob(f"{RESULTS}/trades_*_2026.csv"))
    frames = []
    for f in files:
        d = pl.read_csv(f, try_parse_dates=True)
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        frames.append(d.select(["entry", "net", "atr", "conf", "hour"]))
    tr = pl.concat(frames).sort("entry")
    print(f"loaded {tr.height} trades; overall net/trade {100*tr['net'].mean():+.3f}%")

    rc = rolling_corr_series().sort("date")
    tr = tr.join_asof(rc, left_on="entry", right_on="date", strategy="backward").drop_nulls("mkt_corr")
    print(f"{tr.height} trades labeled with market-corr regime "
          f"(corr range {tr['mkt_corr'].min():.2f}-{tr['mkt_corr'].max():.2f})")

    bucket(tr, "mkt_corr", "crypto-correlation regime (low=decoupled, high=risk-off)")
    bucket(tr, "atr", "volatility (ATR) tercile")
    # cross: does high-corr stress amplify the high-vol edge?
    tr = tr.with_columns([
        (pl.col("mkt_corr") > pl.col("mkt_corr").median()).alias("hi_corr"),
        (pl.col("atr") > pl.col("atr").median()).alias("hi_vol")])
    print("\n=== corr × vol interaction (median splits) ===")
    for hc in [False, True]:
        for hv in [False, True]:
            sub = tr.filter((pl.col("hi_corr") == hc) & (pl.col("hi_vol") == hv))
            if sub.height:
                print(f"  corr={'hi' if hc else 'lo'} vol={'hi' if hv else 'lo'}  "
                      f"n={sub.height:4d}  net-win={100*(sub['net']>0).mean():5.1f}%  "
                      f"net/trade={100*sub['net'].mean():+.3f}%")


if __name__ == "__main__":
    main()
