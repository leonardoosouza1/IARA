"""Honest out-of-sample evaluation for a trained SwarmBrain v9 model.

Loads a checkpoint via the Backtester (so the multi-timeframe construction is
IDENTICAL to training), then reports on a dataset's val_shards:

  - raw 3-class accuracy (Long/Short/Wait, no sniper filter)
  - directional accuracy (only rows the model called Long/Short)
  - win_rate at a sniper confidence threshold (the panel's metric)
  - confusion matrix + label/prediction distributions

Run BTCUSD (in-sample temporal val) and BTCUSDFRESHM1 (true OOS, post-train
cutoff) with the SAME model to see the real generalization gap.

Usage (inside container only):
  docker exec saci-brain python -m scripts.oos_eval \
      --model logs/models/best_model_v9_BTCUSD.pth \
      --datasets BTCUSD,BTCUSDFRESHM1 --threshold 0.5
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.train.backtester import Backtester


def eval_dataset(bt: Backtester, dataset: str, threshold: float) -> dict:
    val_dir = f"data/cache/{dataset}/val_shards"
    shards = sorted(glob.glob(os.path.join(val_dir, "*.pt")))
    if not shards:
        return {"dataset": dataset, "error": f"no val_shards in {val_dir}"}

    n_cls = 3
    conf = np.zeros((n_cls, n_cls), dtype=np.int64)  # [true, pred]
    correct = 0
    total = 0
    dir_correct = 0
    dir_total = 0          # rows the model called Long/Short (raw, no sniper)
    win = 0
    trades = 0             # sniper-filtered trades

    bt.model.eval()
    with torch.no_grad():
        for sf in shards:
            d = torch.load(sf, map_location="cpu")
            X, y = d["X"], d["y"]
            for j in range(0, len(X), 512):
                bx = X[j:j + 512].to(bt.device).float()
                by = y[j:j + 512].long()

                dim = bx.shape[-1]
                if dim > bt.input_dim:
                    bx = bx[:, :, :bt.input_dim]
                elif dim < bt.input_dim:
                    pad = torch.zeros(bx.shape[0], bx.shape[1], bt.input_dim - dim, device=bx.device)
                    bx = torch.cat([bx, pad], dim=-1)
                bx = torch.clamp(torch.nan_to_num(bx), -20.0, 20.0)

                x_dict = bt._simulate_timeframes(bx)
                logits = bt.model(x_dict, compute_telemetry=False)[0]
                probs = F.softmax(logits, dim=-1)
                maxp, pred = probs.max(dim=-1)
                pred = pred.cpu()
                maxp = maxp.cpu()

                for t, p, mp in zip(by.tolist(), pred.tolist(), maxp.tolist()):
                    conf[t, p] += 1
                    total += 1
                    if p == t:
                        correct += 1
                    if p in (0, 1):                 # raw directional call
                        dir_total += 1
                        if p == t:
                            dir_correct += 1
                    # sniper: act only if confident
                    act = p if (p in (0, 1) and mp >= threshold) else 2
                    if act != 2:
                        trades += 1
                        if act == t:
                            win += 1

    return {
        "dataset": dataset,
        "samples": total,
        "acc_3class": round(100 * correct / max(total, 1), 2),
        "acc_directional_raw": round(100 * dir_correct / max(dir_total, 1), 2),
        "directional_calls": dir_total,
        "win_rate": round(100 * win / max(trades, 1), 2),
        "trades": trades,
        "label_dist": np.bincount(
            np.concatenate([torch.load(s, map_location="cpu")["y"].long().numpy() for s in shards]),
            minlength=3).tolist(),
        "confusion_true_pred": conf.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="BTCUSD,BTCUSDFRESHM1")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bt = Backtester(model_path=args.model, device=device)
    print(f"Loaded {args.model} | device={device} | input_dim={bt.input_dim} | worker_hidden={bt.worker_hidden}")

    for ds in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        r = eval_dataset(bt, ds, args.threshold)
        print("=" * 64)
        if "error" in r:
            print(f"{ds}: {r['error']}")
            continue
        print(f"DATASET: {ds}  ({r['samples']} val samples)")
        print(f"  3-class accuracy        : {r['acc_3class']}%")
        print(f"  directional accuracy raw: {r['acc_directional_raw']}%  ({r['directional_calls']} calls)")
        print(f"  win_rate @ thr={args.threshold:.2f}  : {r['win_rate']}%  ({r['trades']} trades)")
        print(f"  label dist [L,S,W]      : {r['label_dist']}")
        print(f"  confusion [true][pred]  : {r['confusion_true_pred']}")


if __name__ == "__main__":
    main()
