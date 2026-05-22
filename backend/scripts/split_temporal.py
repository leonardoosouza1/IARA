"""Split crypto M1 parquet(s) into a train slice (<= train_end) and a backtest
slice (> train_end), writing {OUT}_TR.parquet and {OUT}_BT.parquet to
data/processed/ so feature_factory can shard them by exact stem.

Lets us honour the spec: train 2021-2025, backtest 2026 as a true holdout.
Multiple --extra files (e.g. the FRESH M1 continuation) are concatenated into
whichever side their dates fall on, deduped on timestamp.

Usage (container only):
  python -m scripts.split_temporal --src data/processed/BTCUSD.parquet \
    --extra data/processed/BTCUSDFRESHM1.parquet \
    --train-end 2025-12-31 --out BTCUSD2125
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="primary M1 parquet")
    ap.add_argument("--extra", nargs="*", default=[], help="extra parquets to merge (e.g. FRESH continuation)")
    ap.add_argument("--train-end", required=True, help="last train date inclusive, e.g. 2025-12-31")
    ap.add_argument("--out", required=True, help="output stem, e.g. BTCUSD2125 → {stem}_TR / {stem}_BT")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    frames = [pd.read_parquet(args.src)]
    for e in args.extra:
        frames.append(pd.read_parquet(e))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    dcol = "date" if "date" in df.columns else df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol], utc=True)
    df = df.drop_duplicates(subset=[dcol]).sort_values(dcol).reset_index(drop=True)

    cutoff = pd.Timestamp(args.train_end, tz="UTC") + pd.Timedelta(days=1)  # inclusive end-of-day
    tr = df[df[dcol] < cutoff]
    bt = df[df[dcol] >= cutoff]

    outdir = Path(args.outdir)
    tr_path = outdir / f"{args.out}_TR.parquet"
    bt_path = outdir / f"{args.out}_BT.parquet"
    tr.to_parquet(tr_path, index=False)
    bt.to_parquet(bt_path, index=False)
    print(f"TRAIN {args.out}_TR: {len(tr):>9} rows  {tr[dcol].min()} → {tr[dcol].max()}")
    print(f"BTEST {args.out}_BT: {len(bt):>9} rows  {bt[dcol].min()} → {bt[dcol].max()}")


if __name__ == "__main__":
    main()
