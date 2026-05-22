"""Lens 9: edge intelligence — is the model's confidence calibrated, and is the
edge decaying over time?

The whole strategy rests on CONFIDENCE SELECTION (profit comes from trading only
high-conf bars, RESULTS). Two health checks on the pooled per-trade CSVs
(2024/2025/2026, all coins):

  1. Calibration — does higher confidence => higher realized win-rate / net?
     (a reliability check; if flat, confidence is meaningless)
  2. Alpha decay — is net/trade shrinking month over month? (alpha decay = the
     edge being arbitraged away or the model going stale)

Run in container:
  docker exec saci-brain python -m scripts.edge_health
"""
from __future__ import annotations

import glob

import numpy as np
import polars as pl

RESULTS = "/app/docs/study/results"


def main():
    files = sorted(glob.glob(f"{RESULTS}/trades_*.csv"))
    frames = []
    for f in files:
        d = pl.read_csv(f, try_parse_dates=True)
        if "conf" not in d.columns:
            continue
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        frames.append(d.select(["entry", "net", "conf"]))
    tr = pl.concat(frames).sort("entry")
    print(f"pooled {tr.height} trades from {len(frames)} files "
          f"({tr['entry'].min()} -> {tr['entry'].max()})")
    print(f"overall: net-win {100*(tr['net']>0).mean():.1f}%  net/trade {100*tr['net'].mean():+.3f}%")

    # --- 1. calibration by confidence decile ---
    print("\n=== calibration: realized outcome by confidence decile ===")
    print(f"{'conf bin':>14s} {'n':>5s} {'mean conf':>10s} {'net-win':>8s} {'net/trade':>10s}")
    tr2 = tr.with_columns(
        ((pl.col("conf").rank("ordinal") - 1) * 10 // pl.len()).alias("dec"))
    rows = (tr2.group_by("dec").agg(
        pl.len().alias("n"),
        pl.col("conf").mean().alias("mc"),
        pl.col("conf").min().alias("lo"), pl.col("conf").max().alias("hi"),
        (100 * (pl.col("net") > 0).mean()).alias("nw"),
        (100 * pl.col("net").mean()).alias("npt"))
        .sort("dec")).to_dicts()
    nws = []
    for r in rows:
        print(f"  {r['lo']:.3f}-{r['hi']:.3f} {r['n']:>5d} {r['mc']:>10.3f} "
              f"{r['nw']:>7.1f}% {r['npt']:>+9.3f}%")
        nws.append(r["nw"])
    # monotonicity / Spearman conf vs net-win across deciles
    mc = [r["mc"] for r in rows]
    rho = np.corrcoef(np.argsort(np.argsort(mc)), np.argsort(np.argsort(nws)))[0, 1]
    print(f"  Spearman(conf decile, net-win) = {rho:+.2f}  "
          f"({'calibrada' if rho > 0.5 else 'fraca'}: mais conf → mais acerto)")

    # --- 2. alpha decay by month ---
    print("\n=== alpha decay: net/trade by month ===")
    bym = (tr.with_columns(pl.col("entry").dt.strftime("%Y-%m").alias("ym"))
             .group_by("ym").agg(pl.len().alias("n"),
                                 (100 * (pl.col("net") > 0).mean()).alias("nw"),
                                 (100 * pl.col("net").mean()).alias("npt"))
             .sort("ym")).to_dicts()
    for r in bym:
        bar = "#" * int(max(r["npt"], 0) * 20)
        print(f"  {r['ym']}  n={r['n']:4d}  net-win={r['nw']:5.1f}%  net/trade={r['npt']:+.3f}%  {bar}")
    # linear trend of net/trade over month index
    y = np.array([r["npt"] for r in bym])
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    print(f"\n  trend: {slope:+.4f}pp net/trade per month  "
          f"({'DECAINDO' if slope < -0.01 else ('estável' if abs(slope) <= 0.01 else 'melhorando')})")
    print(f"  first-half avg {y[:len(y)//2].mean():+.3f}%  vs  second-half avg {y[len(y)//2:].mean():+.3f}%")


if __name__ == "__main__":
    main()
