"""Lens 11: TERM STRUCTURE OF EDGE — how the model's edge depends on holding horizon.

Cross-project transfer: byte-language-lab's transformer added Multi-Token
Prediction (MTP) — auxiliary heads that predict t+2, t+3, ... jointly. The finding
there: predicting MORE steps ahead regularizes the trunk and the aux heads learn
real multi-step structure (head t+2 hit ≥50% argmax) WITHOUT hurting the main head.

The trading analog of "predict more steps ahead" is the HOLDING HORIZON. Today the
backtest hard-codes horizon=12 bars (1h on M5) and exits mechanically. Nobody ever
measured what the SAME entry signal earns at other horizons. This script does:

  1. Run the model ONCE → causal direction + confidence at each entry bar
     (independent of horizon — the decision is made at entry).
  2. For h in HORIZONS, replay the EXACT non-overlapping, cost-aware trade logic
     of pnl_backtest and measure net/trade, net-win, Sharpe, CAGR per horizon.
  3. Cross-horizon agreement: does winning at short h predict winning at long h?
     High agreement => one head suffices. Low => horizons carry distinct info =>
     a multi-horizon (MTP-style) model head would add real value (the verdict on
     whether the heavy retrain is worth it).

Honesty: choosing the holding horizon is a design lever fixed BEFORE trading, so
picking the best-Sharpe h on a train period and validating OOS is causal (no
per-trade lookahead). The per-trade "verify across horizons" gate would need the
model to actually emit multi-horizon heads — that is the proposed next step, not
claimed here.

Run in container (GPU):
  docker exec saci-brain python -m scripts.horizon_term_structure \
    --model logs/models/best_model_v9_BTCETHSOL2125_TR.pth \
    --data data/processed/BTCUSD2125_TR.parquet \
    --consensus --horizons 3,6,12,24,48,96 \
    --out_json docs/study/results/horizon_term_structure.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.model_input import build_features
from src.train.backtester import Backtester


def model_decisions(bt, feats12, seq, starts, device):
    """One causal forward pass: direction (0=LONG,1=SHORT) + confidence at each
    entry bar. Built per-batch (no giant pre-stack) to bound memory."""
    preds, confs = [], []
    bt.model.eval()
    with torch.no_grad():
        for j in range(0, len(starts), 1024):
            chunk = starts[j:j + 1024]
            xb = np.stack([feats12[s:s + seq] for s in chunk]).astype(np.float32)
            bx = torch.clamp(torch.nan_to_num(torch.from_numpy(xb).to(device)), -20.0, 20.0)
            logits = bt.model(bt._simulate_timeframes(bx), compute_telemetry=False)[0]
            p = F.softmax(logits, dim=-1)
            mp, pr = p.max(dim=-1)
            preds.append(pr.cpu().numpy()); confs.append(mp.cpu().numpy())
    return np.concatenate(preds), np.concatenate(confs)


def replay(preds, confs, eligible_base, close, entry_idx, cost_frac, h):
    """Non-overlapping, cost-aware sequential trade replay for one horizon h.
    Mirrors pnl_backtest exactly: take an eligible signal, hold h bars, realize
    signed net P&L, then skip ahead h bars before the next entry."""
    exit_idx = entry_idx + h
    raw = (close[exit_idx] - close[entry_idx]) / close[entry_idx]
    rets, trade_dates_idx = [], []
    k, n = 0, len(entry_idx)
    while k < n:
        if eligible_base[k]:
            sret = raw[k] if preds[k] == 0 else -raw[k]
            rets.append(sret - cost_frac)
            trade_dates_idx.append(entry_idx[k])
            k += h
        else:
            k += 1
    return np.array(rets), np.array(trade_dates_idx), raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--move_thr", type=float, default=0.001)
    ap.add_argument("--consensus", action="store_true")
    ap.add_argument("--fee_bps", type=float, default=5.0)
    ap.add_argument("--slippage_bps", type=float, default=2.0)
    ap.add_argument("--horizons", type=str, default="3,6,12,24,48,96")
    ap.add_argument("--base_tf", type=str, default="5m")
    ap.add_argument("--by_vol", action="store_true",
                    help="also break the term structure by trailing-vol tercile "
                         "(causal ATR at entry) — does the optimal horizon shift "
                         "with volatility? => adaptive exit, no retrain")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bt = Backtester(model_path=args.model, device=device)
    feats12, close, dates, consensus = build_features(args.data, base_tf=args.base_tf)
    n = len(feats12)
    cost_frac = 2.0 * (args.fee_bps + args.slippage_bps) / 1e4
    seq = args.seq_len
    horizons = [int(x) for x in args.horizons.split(",")]
    hmax = max(horizons)

    # Common entry set shared by ALL horizons (fair comparison: same signals).
    starts = list(range(0, n - seq - hmax))
    entry_idx = np.array([i + seq - 1 for i in starts])
    preds, confs = model_decisions(bt, feats12, seq, starts, device)

    # Per-bar minute return at the longest horizon used only for the agreement
    # analysis; the trade gate is identical across horizons (decided at entry).
    eligible_base = np.isin(preds, [0, 1]) & (confs >= args.threshold)
    if args.consensus:
        # consensus + a minimum move at the *reference* horizon=12 (training regime)
        ref_h = 12 if 12 in horizons else horizons[len(horizons) // 2]
        ref_raw = (close[entry_idx + ref_h] - close[entry_idx]) / close[entry_idx]
        eligible_base = eligible_base & consensus[entry_idx] & (np.abs(ref_raw) >= args.move_thr)

    span_days = (pd.Timestamp(dates.iloc[entry_idx[-1]]) - pd.Timestamp(dates.iloc[entry_idx[0]])).total_seconds() / 86400
    years = max(span_days / 365.25, 1e-6)

    print("=" * 72)
    print(f"TERM STRUCTURE OF EDGE — {args.model.split('/')[-1]}")
    print(f"DATA {args.data.split('/')[-1]}  {dates.iloc[0]} -> {dates.iloc[-1]}  ({years:.2f}y)")
    print(f"cost round-trip {cost_frac*100:.3f}%  | consensus={args.consensus} | thr={args.threshold}")
    print(f"common entries (eligible signals): {int(eligible_base.sum())}")
    print("=" * 72)
    print(f"{'horizon':>8s} {'~time':>7s} {'trades':>7s} {'net-win':>8s} {'net/tr':>9s} "
          f"{'Sharpe':>7s} {'CAGR':>8s} {'total':>9s}")

    rows = []
    bar_min = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(args.base_tf, 5)
    for h in horizons:
        rets, _, _ = replay(preds, confs, eligible_base, close, entry_idx, cost_frac, h)
        ntr = len(rets)
        if ntr < 2:
            continue
        nw = 100 * (rets > 0).mean()
        npt = 100 * rets.mean()
        tpy = ntr / years
        sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(tpy))
        eq = np.cumprod(1.0 + rets)
        total = (eq[-1] - 1.0) * 100
        cagr = (eq[-1] ** (1.0 / years) - 1.0) * 100
        tmin = h * bar_min
        tlabel = f"{tmin}m" if tmin < 60 else f"{tmin/60:.0f}h"
        print(f"{h:>8d} {tlabel:>7s} {ntr:>7d} {nw:>7.1f}% {npt:>+8.4f}% "
              f"{sharpe:>7.2f} {cagr:>+7.1f}% {total:>+8.1f}%")
        rows.append({"horizon_bars": h, "approx": tlabel, "trades": ntr,
                     "net_win_pct": round(nw, 2), "net_per_trade_pct": round(npt, 4),
                     "sharpe": round(sharpe, 2), "cagr_pct": round(cagr, 2),
                     "total_pct": round(total, 2)})

    # --- cross-horizon agreement: is the signed edge the SAME sign across h? ---
    # If yes, one head suffices. If horizons disagree often, a multi-horizon
    # (MTP-style) head carries distinct, exploitable information.
    print("-" * 72)
    ent = entry_idx[eligible_base]
    sgn = {}
    for h in horizons:
        raw = (close[ent + h] - close[ent]) / close[ent]
        sret = np.where(preds[eligible_base] == 0, raw, -raw)
        sgn[h] = np.sign(sret)
    short_h, long_h = horizons[0], horizons[-1]
    agree = float((sgn[short_h] == sgn[long_h]).mean())
    win_short = float((sgn[short_h] > 0).mean())
    # P(profitable at long | profitable at short) — does short-horizon success carry forward?
    m = sgn[short_h] > 0
    p_long_given_short = float((sgn[long_h][m] > 0).mean()) if m.any() else float("nan")
    base_long = float((sgn[long_h] > 0).mean())
    print(f"cross-horizon sign agreement  h={short_h} vs h={long_h}: {100*agree:.1f}%")
    print(f"P(win@{long_h}) = {100*base_long:.1f}%   "
          f"P(win@{long_h} | win@{short_h}) = {100*p_long_given_short:.1f}%   "
          f"(lift {100*(p_long_given_short-base_long):+.1f}pp)")
    verdict = ("horizontes carregam info DISTINTA → head MTP multi-horizonte vale o retrain"
               if agree < 0.80 else
               "horizontes concordam forte → 1 head basta, MTP daria pouco")
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    # --- optional: does the optimal horizon shift with volatility? ---
    # Split eligible entries by trailing ATR (col 6, causal at entry) into terciles.
    # If high-vol favors short h and calm favors long h, a vol-conditioned exit is
    # a causal, no-retrain improvement.
    vol_rows = []
    if args.by_vol:
        atr_e = feats12[entry_idx, 6]
        lo, hi = np.quantile(atr_e[eligible_base], [1/3, 2/3])
        terc = {"calm (low ATR)": atr_e <= lo,
                "normal": (atr_e > lo) & (atr_e <= hi),
                "stress (high ATR)": atr_e > hi}
        print("-" * 72)
        print("ADAPTIVE EXIT — peak horizon by trailing-vol tercile (causal):")
        for label, vmask in terc.items():
            base = eligible_base & vmask
            best = None
            per_h = []
            for h in horizons:
                rets, _, _ = replay(preds, confs, base, close, entry_idx, cost_frac, h)
                if len(rets) < 2:
                    continue
                tpy = len(rets) / years
                sh = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(tpy))
                per_h.append((h, sh, 100 * (rets > 0).mean(), len(rets)))
                if best is None or sh > best[1]:
                    best = (h, sh, 100 * (rets > 0).mean(), len(rets))
            if best:
                print(f"  {label:>18s}: best h={best[0]:>3d} bars  "
                      f"Sharpe {best[1]:5.2f}  win {best[2]:.1f}%  (n={best[3]})")
                vol_rows.append({"regime": label, "best_horizon_bars": best[0],
                                 "sharpe": round(best[1], 2), "net_win_pct": round(best[2], 2),
                                 "trades": best[3]})
        print("=" * 72)

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"model": args.model, "data": args.data, "years": round(years, 2),
                       "cost_round_trip_pct": round(cost_frac * 100, 4),
                       "common_entries": int(eligible_base.sum()),
                       "term_structure": rows,
                       "agreement": {"short_h": short_h, "long_h": long_h,
                                     "sign_agreement_pct": round(100 * agree, 2),
                                     "p_win_long": round(100 * base_long, 2),
                                     "p_win_long_given_short": round(100 * p_long_given_short, 2),
                                     "lift_pp": round(100 * (p_long_given_short - base_long), 2)},
                       "by_vol": vol_rows,
                       "verdict": verdict}, fh, indent=2)
        print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
