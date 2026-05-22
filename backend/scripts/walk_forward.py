"""Walk-forward validation: the gold-standard out-of-sample test.

Expanding train window, each subsequent year a true holdout:
  fold 1: train 2021..2022 → test 2023
  fold 2: train 2021..2023 → test 2024
  fold 3: train 2021..2024 → test 2025
  fold 4: train 2021..2025 → test 2026

Per fold: slice train/test parquets by date → shard train (return label) →
train SwarmBrain → cost-aware backtest on the test year with realistic position
sizing. Collects metrics to docs/study/results/walkforward_{symbol}.json.

Resumable: skips a fold whose per-fold metrics JSON already exists.

Run in the container, in background (heavy, ~25 min/fold):
  docker exec saci-brain python -m scripts.walk_forward --symbol BTCUSD \
    --src data/processed/BTCUSD.parquet --extra data/processed/BTCUSDFRESHM1.parquet
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
RESULTS = Path("docs/study/results")
MODELS = Path("logs/models")

# Fewer folds (3), each a true held-out year. M5 source (M1 dropped — the brain
# uses M5 base; M1 just wasted resample/Hurst time).
FOLDS = [
    ("2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("2024-12-31", "2025-01-01", "2025-12-31", "2025"),
    ("2025-12-31", "2026-01-01", "2026-12-31", "2026"),
]


def run(cmd: list[str], log: Path):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--risk_frac", type=float, default=0.1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    log = Path(f"logs/wf_{args.symbol}.log")

    # Load full history once
    frames = [pd.read_parquet(args.src)] + [pd.read_parquet(e) for e in args.extra]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    summary = []
    for train_end, test_start, test_end, tag in FOLDS:
        sym_tr = f"WF{args.symbol}{tag}_TR"
        mfile = RESULTS / f"wf_{args.symbol}_{tag}.json"
        if mfile.exists():
            summary.append(json.loads(mfile.read_text()))
            print(f"[fold {tag}] cached, skip")
            continue

        te = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
        ts0 = pd.Timestamp(test_start, tz="UTC")
        ts1 = pd.Timestamp(test_end, tz="UTC") + pd.Timedelta(days=1)
        tr = df[df["date"] < te]
        bt = df[(df["date"] >= ts0) & (df["date"] < ts1)]
        if len(bt) < 5000:
            print(f"[fold {tag}] test slice too small ({len(bt)}), skip")
            continue
        tr_path = PROC / f"{sym_tr}.parquet"
        bt_path = PROC / f"WF{args.symbol}{tag}_BT.parquet"
        tr.to_parquet(tr_path, index=False)
        bt.to_parquet(bt_path, index=False)
        print(f"[fold {tag}] train {len(tr)} (→{train_end})  test {len(bt)} ({test_start}..{test_end})")

        # shard train (return label)
        run(["python", "-m", "src.data.feature_factory", "--symbol", sym_tr,
             "--market-type", "crypto", "--seq-len", str(args.seq_len),
             "--stride", "2", "--label-mode", "return"], log)
        # train
        run(["python", "-m", "src.brain.train_v9", "--symbol", sym_tr,
             "--epochs", str(args.epochs), "--patience", "20",
             "--batch_size", "1024", "--seq_len", str(args.seq_len)], log)
        # backtest on the held-out year
        model = MODELS / f"best_model_v9_{sym_tr}.pth"
        run(["python", "-m", "scripts.pnl_backtest", "--model", str(model),
             "--data", str(bt_path), "--threshold", str(args.threshold),
             "--consensus", "--risk_frac", str(args.risk_frac),
             "--metrics_json", str(mfile),
             "--equity_csv", str(RESULTS / f"wf_{args.symbol}_{tag}_equity.csv")], log)

        m = json.loads(mfile.read_text())
        m["fold"] = tag
        m["train_end"] = train_end
        summary.append(m)
        print(f"[fold {tag}] net-win={m['net_win_pct']}% net/trade={m['net_per_trade_pct']}% "
              f"sharpe={m['sharpe']} maxDD={m['max_drawdown_pct']}% bh={m['buy_hold_pct']}%")

    out = RESULTS / f"walkforward_{args.symbol}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n=== WALK-FORWARD SUMMARY ({args.symbol}) → {out} ===")
    print(f"{'fold':6} {'net-win':>8} {'net/trade':>10} {'sharpe':>7} {'maxDD':>7} {'buyhold':>8} {'trades':>7}")
    for m in summary:
        print(f"{m.get('fold',''):6} {m['net_win_pct']:>7}% {m['net_per_trade_pct']:>9}% "
              f"{m['sharpe']:>7} {m['max_drawdown_pct']:>6}% {m['buy_hold_pct']:>7}% {m['trades']:>7}")


if __name__ == "__main__":
    main()
