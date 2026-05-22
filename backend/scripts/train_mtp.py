"""MTP retrain — does a multi-horizon auxiliary head help (or at least not hurt)
the main trading head? Cross-project transfer from byte-language-lab's MTP.

Self-contained on purpose: builds multi-horizon labels DIRECTLY from the M5
parquet via the same build_features + make_timeframe_dict path that pnl_backtest
is validated against (no fragile cache regen), so train and backtest can't drift.
The model's main head still predicts direction at the trained horizon (12 bars =
1h); the aux heads predict the SAME entry's direction at 24 (2h) and 48 (4h) bars.

A/B by design: --mtp on/off changes ONLY the aux heads + aux loss; everything else
(seed, data, main loss, optimizer) is identical, so the comparison isolates the
MTP effect. We report (a) main-head val accuracy — must NOT drop with MTP, (b) aux
head val accuracy — must beat chance (the heads learn real multi-step structure),
(c) OOS net/trade of the main head on a held-out parquet, replaying pnl_backtest's
exact non-overlapping cost-aware logic.

Run in container (GPU):
  docker exec saci-brain python -m scripts.train_mtp --mtp  --epochs 25 --tag MTP
  docker exec saci-brain python -m scripts.train_mtp --no-mtp --epochs 25 --tag BASE
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from src.data.model_input import build_features, make_timeframe_dict
from src.brain.models.workers import WorkerConfig
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig

TR = "data/processed/BTCUSD2125_TR.parquet"   # 2021-2025 train
BT = "data/processed/BTCUSD2125_BT.parquet"   # 2026 held-out
RESULTS = "/app/docs/study/results"
MAIN_H, AUX_H = 12, (24, 48)
SEQ, STRIDE, THR = 128, 2, 0.001


def build_samples(path, base_tf="5m"):
    """Windows + multi-horizon direction labels on consensus golden bars (same
    gate as feature_factory: fractal consensus AND |1h return| >= threshold)."""
    feats, close, _, consensus = build_features(path, base_tf=base_tf)
    feats = feats[:, :12].astype(np.float32)
    n = len(feats)
    hmax = max(AUX_H)
    X, Y = [], []
    for i in range(0, n - SEQ - hmax, STRIDE):
        t = i + SEQ - 1
        if not consensus[t]:
            continue
        p0 = close[t]
        if abs(close[t + MAIN_H] - p0) / p0 < THR:        # main-horizon move filter
            continue
        X.append(feats[i:i + SEQ])
        Y.append([0 if close[t + h] > p0 else 1 for h in (MAIN_H,) + AUX_H])
    return np.asarray(X, np.float32), np.asarray(Y, np.int64)


def batches(X, Y, bs, device, shuffle):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for j in range(0, len(idx), bs):
        b = idx[j:j + bs]
        xb = torch.clamp(torch.nan_to_num(torch.from_numpy(X[b])), -20, 20).to(device)
        yield make_timeframe_dict(xb, SEQ), torch.from_numpy(Y[b]).to(device)


@torch.no_grad()
def evaluate(model, X, Y, device, use_mtp):
    model.eval()
    mc = ac = tot = 0
    aux_correct = np.zeros(len(AUX_H))
    for xd, y in batches(X, Y, 512, device, False):
        logits = model(xd)[0]
        pred = logits[:, :2].argmax(-1)            # golden labels are {0,1}
        mc += (pred == y[:, 0]).sum().item(); tot += len(y)
        if use_mtp and model.coordinator.mtp_logits is not None:
            for k, al in enumerate(model.coordinator.mtp_logits):
                aux_correct[k] += (al[:, :2].argmax(-1) == y[:, k + 1]).sum().item()
    return mc / tot, (aux_correct / tot if use_mtp else None)


def replay_oos(model, path, device, h=MAIN_H, thr=0.6, cost=2 * (5 + 2) / 1e4):
    """OOS net/trade of the MAIN head — pnl_backtest's exact non-overlapping,
    cost-aware logic at the trained horizon, on a held-out parquet."""
    feats, close, _, consensus = build_features(path, base_tf="5m")
    feats = feats[:, :12].astype(np.float32)
    n = len(feats)
    starts = list(range(0, n - SEQ - h))
    ent = np.array([i + SEQ - 1 for i in starts])
    preds, confs = [], []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(starts), 1024):
            xb = np.stack([feats[s:s + SEQ] for s in starts[j:j + 1024]]).astype(np.float32)
            bx = torch.clamp(torch.nan_to_num(torch.from_numpy(xb)), -20, 20).to(device)
            p = F.softmax(model(make_timeframe_dict(bx, SEQ))[0], dim=-1)
            mp, pr = p.max(-1)
            preds.append(pr.cpu().numpy()); confs.append(mp.cpu().numpy())
    preds, confs = np.concatenate(preds), np.concatenate(confs)
    raw = (close[ent + h] - close[ent]) / close[ent]
    elig = np.isin(preds, [0, 1]) & (confs >= thr) & consensus[ent] & (np.abs(raw) >= THR)
    rets, k = [], 0
    while k < len(starts):
        if elig[k]:
            rets.append((raw[k] if preds[k] == 0 else -raw[k]) - cost); k += h
        else:
            k += 1
    rets = np.array(rets)
    if len(rets) == 0:
        return {"trades": 0, "net_per_trade_pct": 0.0, "net_win_pct": 0.0}
    return {"trades": int(len(rets)), "net_per_trade_pct": round(100 * rets.mean(), 4),
            "net_win_pct": round(100 * (rets > 0).mean(), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtp", dest="mtp", action="store_true", default=True)
    ap.add_argument("--no-mtp", dest="mtp", action="store_false")
    ap.add_argument("--lam", type=float, default=0.3, help="aux loss weight (byte-lab used 0.3)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="MTP")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = build_samples(TR)
    cut = int(len(X) * 0.8)                       # temporal split (no shuffle across the cut)
    Xtr, Ytr, Xva, Yva = X[:cut], Y[:cut], X[cut:], Y[cut:]
    base = max(Ytr[:, 0].mean(), 1 - Ytr[:, 0].mean())   # majority-class baseline
    print(f"[{args.tag}] samples {len(X)} (train {len(Xtr)}, val {len(Xva)})  "
          f"main-label balance {Ytr[:,0].mean():.3f}  majority {base:.3f}  device {device}  mtp={args.mtp}")

    wc = WorkerConfig(input_dim=12, hidden_dim=64, dropout=0.1)
    cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256, dropout=0.3,
                           mtp_horizons=AUX_H if args.mtp else ())
    model = SwarmBrain(wc, cc).to(device)
    print(f"[{args.tag}] params {model.count_parameters():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_main, best_state, best_aux = 0.0, None, None
    for ep in range(args.epochs):
        model.train()
        tl = []
        for xd, y in batches(Xtr, Ytr, args.bs, device, True):
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits = model(xd)[0]
                loss = F.cross_entropy(logits, y[:, 0])
                if args.mtp and model.coordinator.mtp_logits is not None:
                    aux = sum(F.cross_entropy(al, y[:, k + 1])
                              for k, al in enumerate(model.coordinator.mtp_logits)) / len(AUX_H)
                    loss = loss + args.lam * aux
            if torch.isnan(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl.append(loss.item())
        macc, aacc = evaluate(model, Xva, Yva, device, args.mtp)
        astr = "" if aacc is None else "  aux=" + "/".join(f"{a:.3f}" for a in aacc)
        print(f"[{args.tag}] ep {ep+1:2d}  loss {np.mean(tl):.4f}  val_main_acc {macc:.4f}{astr}")
        if macc > best_main:
            best_main, best_aux = macc, aacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    path = f"logs/models/best_model_v9_{args.tag}_BTC.pth"
    torch.save(best_state, path)
    oos = replay_oos(model, BT, device)
    print(f"[{args.tag}] BEST val_main_acc {best_main:.4f} (majority {base:.3f})"
          + ("" if best_aux is None else "  aux=" + "/".join(f"{a:.3f}" for a in best_aux)))
    print(f"[{args.tag}] OOS 2026 main head: {oos}")
    print(f"[{args.tag}] saved {path}")

    out = {"tag": args.tag, "mtp": args.mtp, "lam": args.lam, "samples": int(len(X)),
           "majority_baseline": round(float(base), 4), "best_val_main_acc": round(float(best_main), 4),
           "aux_val_acc": None if best_aux is None else [round(float(a), 4) for a in best_aux],
           "aux_horizons": list(AUX_H), "oos_2026": oos}
    with open(args.out_json or f"{RESULTS}/train_mtp_{args.tag}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[{args.tag}] saved {args.out_json or f'{RESULTS}/train_mtp_{args.tag}.json'}")


if __name__ == "__main__":
    main()
