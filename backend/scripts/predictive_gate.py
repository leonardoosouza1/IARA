"""Lens 13: PREDICTIVE GATE — the byte-lab draft/verify idea at EXECUTION time.

The MTP retrain (Lens 11) showed aux heads don't help as a TRAINING signal. But the
byte-lab's other half — speculative draft/verify — is an INFERENCE-time idea: the
model drafts future steps, and as those drafts VERIFY correct, you trust it more.
Leonardo's intuition: "conforme ele prevê melhor e acerta, isso dá respostas melhores."

We test that as a causal confidence gate, no retrain:

  (A) DRAFT-VERIFY gate — run the trained MTP model over the price series. At each
      bar it drafts the 2h/4h direction. A draft made at bar m is VERIFIABLE at m+H
      (when that candle closes). At an entry bar E we know only drafts whose target
      already passed (m+H <= E) -> a strictly-causal rolling "draft hit-rate" = is the
      model in sync with the market right now? Gate: trade only when in-sync.

  (B) HOT-HAND gate — simpler, model-agnostic: the model's realized win-rate over its
      last K CLOSED trades (strictly prior). Does recent skill predict the next trade?

Both are causal (no lookahead). Honest prior: the aux heads are weak (~0.545 acc), so
the draft signal may be noisy; but if it tracks regime (Lens 7), being "in sync" could
still mark better-edge windows.

Run in container (GPU):
  docker exec saci-brain python -m scripts.predictive_gate \
    --model logs/models/best_model_v9_MTP_BTC.pth \
    --data data/processed/BTCUSD2125_BT.parquet \
    --out_json docs/study/results/predictive_gate.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from src.data.model_input import build_features, make_timeframe_dict
from src.brain.models.workers import WorkerConfig
from src.brain.models.liquid_v9 import SwarmBrain, CoordinatorConfig

SEQ, MAIN_H, AUX_H, THR = 128, 12, (24, 48), 0.001
COST = 2 * (5 + 2) / 1e4
RESULTS = "/app/docs/study/results"


def rolling_hit(correct, active, end_offset, win):
    """Causal rolling hit-rate of drafts. correct[m], active[m] indexed by the bar
    the draft was MADE. A draft at m is known at bar E only if m <= E-end_offset
    (its target candle has closed). Returns per-bar rolling hit-rate aligned to E."""
    n = len(correct)
    c = (correct & active).astype(np.float64)
    a = active.astype(np.float64)
    cc, ca = np.concatenate([[0], np.cumsum(c)]), np.concatenate([[0], np.cumsum(a)])
    out = np.full(n, np.nan)
    for E in range(n):
        hi = E - end_offset + 1          # drafts made up to bar (E-end_offset) are known
        lo = max(0, hi - win)
        if hi > lo and (ca[hi] - ca[lo]) >= 5:   # need >=5 verified drafts
            out[E] = (cc[hi] - cc[lo]) / (ca[hi] - ca[lo])
    return out


def draft_verify_gate(model, path, device, win=200, thr=0.6):
    feats, close, _, consensus = build_features(path, base_tf="5m")
    feats = feats[:, :12].astype(np.float32)
    n = len(feats)
    starts = list(range(0, n - SEQ - max(AUX_H)))
    ent = np.array([i + SEQ - 1 for i in starts])

    main_pred, conf, d2, d4 = [], [], [], []
    model.eval()
    with torch.no_grad():
        for j in range(0, len(starts), 1024):
            xb = np.stack([feats[s:s + SEQ] for s in starts[j:j + 1024]]).astype(np.float32)
            bx = torch.clamp(torch.nan_to_num(torch.from_numpy(xb)), -20, 20).to(device)
            logits = model(make_timeframe_dict(bx, SEQ))[0]
            p = F.softmax(logits, -1); mp, pr = p.max(-1)
            main_pred.append(pr.cpu().numpy()); conf.append(mp.cpu().numpy())
            al = model.coordinator.mtp_logits
            d2.append(al[0][:, :2].argmax(-1).cpu().numpy())
            d4.append(al[1][:, :2].argmax(-1).cpu().numpy())
    main_pred, conf = np.concatenate(main_pred), np.concatenate(conf)
    d2, d4 = np.concatenate(d2), np.concatenate(d4)

    # verify drafts (indexed by the entry bar where the draft was made)
    def verify(draft, h):
        e = ent
        realized_up = close[e + h] > close[e]            # actual move sign
        draft_up = draft == 0                            # 0=LONG/up,1=SHORT/down
        correct = draft_up == realized_up
        active = np.isin(draft, [0, 1])
        return correct, active

    c2, a2 = verify(d2, AUX_H[0])
    c4, a4 = verify(d4, AUX_H[1])
    # rolling sync = mean of 2h and 4h rolling hit-rates (each known at the right lag)
    h2 = rolling_hit(c2, a2, AUX_H[0], win)
    h4 = rolling_hit(c4, a4, AUX_H[1], win)
    sync = np.nanmean(np.vstack([h2, h4]), axis=0)

    # base eligibility (same as pnl_backtest)
    raw = (close[ent + MAIN_H] - close[ent]) / close[ent]
    elig = np.isin(main_pred, [0, 1]) & (conf >= thr) & consensus[ent] & (np.abs(raw) >= THR)

    # non-overlapping trade walk; record sync at entry
    rets, syncs, k = [], [], 0
    while k < len(starts):
        if elig[k] and not np.isnan(sync[k]):
            rets.append((raw[k] if main_pred[k] == 0 else -raw[k]) - COST)
            syncs.append(sync[k]); k += MAIN_H
        else:
            k += 1
    rets, syncs = np.array(rets), np.array(syncs)
    return rets, syncs


def report_gate(name, rets, signal, out):
    base = 100 * rets.mean()
    print(f"\n=== {name} gate ({len(rets)} trades, base {base:+.4f}%/tr) ===")
    q = np.quantile(signal, [1/3, 2/3])
    terc = np.searchsorted(q, signal)
    rows = []
    for t, lab in [(0, "low (out of sync)"), (1, "mid"), (2, "high (in sync)")]:
        m = terc == t
        if m.sum() == 0:
            continue
        npt = 100 * rets[m].mean()
        print(f"  {lab:<18s} n={m.sum():4d}  net/trade {npt:+.4f}%  net-win {100*(rets[m]>0).mean():.1f}%")
        rows.append({"bucket": lab, "n": int(m.sum()), "net_per_trade_pct": round(npt, 4)})
    hi = signal >= np.median(signal)
    delta = 100 * (rets[hi].mean() - rets.mean())
    print(f"  GATE: trade only top-half sync -> {100*rets[hi].mean():+.4f}%/tr (Δ{delta:+.4f}pp vs base)")
    # correlation of the signal with realized net
    if signal.std() > 0:
        rho = np.corrcoef(signal, rets)[0, 1]
        print(f"  corr(sync, net) = {rho:+.3f}")
    out[name] = {"base_net_per_trade": round(base, 4), "buckets": rows,
                 "tophalf_net_per_trade": round(float(100*rets[hi].mean()), 4),
                 "tophalf_delta_pp": round(float(delta), 4),
                 "corr_signal_net": round(float(rho), 4) if signal.std() > 0 else None}


def hot_hand_gate(out, win=20):
    """(B) Model-agnostic: trailing realized win-rate over last `win` CLOSED trades
    (strictly prior) gates the next trade. Pools 2026 per-trade CSVs."""
    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/trades_*_2026.csv")):
        d = pl.read_csv(f, try_parse_dates=True)
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        frames.append(d.select(["entry", "net"]))
    tr = pl.concat(frames).sort("entry")
    net = tr["net"].to_numpy()
    win_flag = (net > 0).astype(np.float64)
    # trailing win-rate over previous `win` trades, shifted 1 (strictly past)
    cs = np.concatenate([[0], np.cumsum(win_flag)])
    trail = np.full(len(net), np.nan)
    for i in range(win + 1, len(net)):
        trail[i] = (cs[i] - cs[i - win]) / win
    mask = ~np.isnan(trail)
    report_gate("hot_hand", net[mask], trail[mask], out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logs/models/best_model_v9_MTP_BTC.pth")
    ap.add_argument("--data", default="data/processed/BTCUSD2125_BT.parquet")
    ap.add_argument("--win", type=int, default=200, help="rolling window (bars) for draft hit-rate")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wc = WorkerConfig(input_dim=12, hidden_dim=64, dropout=0.1)
    cc = CoordinatorConfig(num_workers=5, worker_dim=64, hidden_dim=256, dropout=0.3,
                           mtp_horizons=AUX_H)
    model = SwarmBrain(wc, cc).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=True)
    print(f"loaded {args.model} (MTP heads {AUX_H}) on {device}")

    out = {}
    rets, syncs = draft_verify_gate(model, args.data, device, win=args.win)
    report_gate("draft_verify", rets, syncs, out)
    hot_hand_gate(out)

    with open(args.out_json or f"{RESULTS}/predictive_gate.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved: {args.out_json or f'{RESULTS}/predictive_gate.json'}")


if __name__ == "__main__":
    main()
