"""Generate demonstration plots for the study from saved results.

  - equity curves per walk-forward fold (sized, after costs)
  - bar chart: net-win vs buy&hold per fold/coin

Reads docs/study/results/*.json and *_equity.csv, writes PNGs to
docs/study/plots/. Run in container:
  docker exec saci-brain python -m scripts.plot_results --symbol BTCUSD
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("docs/study/results")
PLOTS = Path("docs/study/plots")


def plot_equity(symbol: str):
    files = sorted(glob.glob(str(RESULTS / f"wf_{symbol}_*_equity.csv")))
    if not files:
        print("no equity CSVs yet")
        return
    plt.figure(figsize=(10, 6))
    for f in files:
        tag = Path(f).stem.replace(f"wf_{symbol}_", "").replace("_equity", "")
        df = pd.read_csv(f, parse_dates=["entry"])
        plt.plot(df["entry"], df["equity"], label=f"fold {tag}")
    plt.axhline(1.0, color="gray", ls="--", lw=0.8)
    plt.title(f"{symbol} walk-forward equity (sized, after costs)")
    plt.ylabel("equity (×)")
    plt.legend()
    plt.tight_layout()
    out = PLOTS / f"{symbol}_walkforward_equity.png"
    plt.savefig(out, dpi=110)
    print(f"wrote {out}")


def plot_bars(symbol: str):
    wf = RESULTS / f"walkforward_{symbol}.json"
    if not wf.exists():
        print("no walkforward summary yet")
        return
    data = json.loads(wf.read_text())
    if not data:
        return
    folds = [m.get("fold", "?") for m in data]
    netwin = [m["net_win_pct"] for m in data]
    bh = [m["buy_hold_pct"] for m in data]
    x = range(len(folds))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], netwin, width=0.4, label="net-win %", color="seagreen")
    plt.bar([i + 0.2 for i in x], bh, width=0.4, label="buy&hold %", color="indianred")
    plt.axhline(50, color="gray", ls="--", lw=0.8)
    plt.xticks(list(x), folds)
    plt.title(f"{symbol} walk-forward: net-win vs buy&hold per held-out year")
    plt.legend()
    plt.tight_layout()
    out = PLOTS / f"{symbol}_walkforward_bars.png"
    plt.savefig(out, dpi=110)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSD")
    args = ap.parse_args()
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_equity(args.symbol)
    plot_bars(args.symbol)


if __name__ == "__main__":
    main()
