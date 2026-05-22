"""Faithful, cost-aware P&L backtest for a trained SwarmBrain v9 model.

Why not src/brain/backtest_v9.py: it has no transaction-cost model, uses a
different (consensus-vote) decision path than the validated argmax, and a
separate feature path — so its P&L can't be trusted against the eval numbers.

This rebuilds the EXACT representation the model was validated on
(resample_data → compute_features → first-12 cols → _simulate_timeframes),
adds a real cost model, and trades REALISTICALLY:
  - one position at a time (NON-overlapping): take a signal, hold `horizon`
    bars, realize P&L, only then look for the next signal. (Trading every
    overlapping window pays the round-trip cost thousands of times → ruin.)
  - optional --consensus gate: only trade fractal-aligned "golden" bars
    (H1 trend == M5 trend, |future return| ≥ threshold) — the regime the model
    was actually trained on.

Costs (crypto taker, conservative): (fee_bps + slippage_bps) per side, charged
on entry and exit.

Usage (container only):
  docker exec saci-brain python -m scripts.pnl_backtest \
    --model logs/models/best_model_v9_BTCUSD.pth \
    --data data/processed/BTCUSDFRESHM1.parquet \
    --threshold 0.6 --consensus --fee_bps 5 --slippage_bps 2 --horizon 12
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F

from src.data.model_input import build_features
from src.train.backtester import Backtester


def build_m5(data_path: str, base_tf: str = "5m"):
    """Thin wrapper over the shared model_input.build_features — single source of
    truth for the model's input (same code training will use). Now computes only
    the 12 core features (no Hurst/FFT/etc) → much faster than the old path."""
    return build_features(data_path, base_tf=base_tf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=12, help="bars to hold (12=1h on M5)")
    ap.add_argument("--threshold", type=float, default=0.6, help="min confidence to trade")
    ap.add_argument("--move_thr", type=float, default=0.001, help="min |future return| for a golden bar")
    ap.add_argument("--consensus", action="store_true", help="only trade fractal-aligned golden bars")
    ap.add_argument("--fee_bps", type=float, default=5.0, help="taker fee per side (bps)")
    ap.add_argument("--slippage_bps", type=float, default=2.0, help="slippage per side (bps)")
    ap.add_argument("--trades_csv", type=str, default=None, help="dump per-trade (entry,net) to CSV")
    ap.add_argument("--equity_csv", type=str, default=None, help="dump equity curve (entry,equity) to CSV")
    ap.add_argument("--metrics_json", type=str, default=None, help="dump metrics dict to JSON")
    ap.add_argument("--risk_frac", type=float, default=1.0, help="fraction of equity deployed per trade (1.0=full, 0.1=realistic)")
    ap.add_argument("--base_tf", type=str, default="5m", help="base trading timeframe (5m,15m,30m,1h) — probe robustness across cadences")
    ap.add_argument("--slippage_atr_mult", type=float, default=0.0, help="extra slippage = mult × ATR-fraction per side (volatility-scaled; 0=off)")
    ap.add_argument("--baseline", type=str, default="model", choices=["model", "momentum"],
                    help="'momentum' = trade sign of velocity(ROC) instead of model; honesty check (is the edge just momentum?)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bt = Backtester(model_path=args.model, device=device)
    feats12, close, dates, consensus = build_m5(args.data, base_tf=args.base_tf)
    n = len(feats12)
    cost_frac = 2.0 * (args.fee_bps + args.slippage_bps) / 1e4

    seq, hz = args.seq_len, args.horizon
    starts = list(range(0, n - seq - hz))
    X = np.stack([feats12[i:i + seq] for i in starts]).astype(np.float32)
    entry_idx = np.array([i + seq - 1 for i in starts])
    exit_idx = entry_idx + hz
    entry_px, exit_px = close[entry_idx], close[exit_idx]
    raw_ret = (exit_px - entry_px) / entry_px
    realized_up = exit_px > entry_px

    if args.baseline == "momentum":
        # Honesty baseline: trade the sign of velocity (ROC, feature col 9) at the
        # decision bar. Same gates/costs/sizing → isolates whether the model adds
        # anything over plain momentum. conf=1.0 so the threshold doesn't filter.
        vel = feats12[entry_idx, 9]
        preds = np.where(vel > 0, 0, 1)
        confs = np.ones(len(preds))
    else:
        preds, confs = [], []
        bt.model.eval()
        with torch.no_grad():
            for j in range(0, len(X), 1024):
                bx = torch.clamp(torch.nan_to_num(torch.from_numpy(X[j:j + 1024]).to(device)), -20.0, 20.0)
                logits = bt.model(bt._simulate_timeframes(bx), compute_telemetry=False)[0]
                p = F.softmax(logits, dim=-1)
                mp, pr = p.max(dim=-1)
                preds.append(pr.cpu().numpy()); confs.append(mp.cpu().numpy())
        preds = np.concatenate(preds); confs = np.concatenate(confs)

    # Fidelity gate: directional accuracy on the golden subset (matches training)
    gate_mask = np.isin(preds, [0, 1])
    if args.consensus:
        gate_mask &= consensus[entry_idx]
    pred_up = preds == 0
    dir_acc = 100.0 * (pred_up[gate_mask] == realized_up[gate_mask]).mean() if gate_mask.any() else 0.0

    # Per-trade cost: fixed (fee+slip) + volatility-scaled slippage. atr_norm
    # (col 6) = (ATR/close)*100, so atr_norm/100 = ATR fraction of price. Real
    # slippage widens with volatility; --slippage_atr_mult scales it (0 = off).
    atr_frac = feats12[entry_idx, 6] / 100.0
    cost_arr = cost_frac + 2.0 * args.slippage_atr_mult * np.clip(atr_frac, 0, None)

    # NON-overlapping sequential trading
    eligible = np.isin(preds, [0, 1]) & (confs >= args.threshold)
    if args.consensus:
        eligible &= consensus[entry_idx] & (np.abs(raw_ret) >= args.move_thr)
    rets, wins, n_long, n_short, trade_dates = [], 0, 0, 0, []
    k = 0
    while k < len(starts):
        if eligible[k]:
            sret = raw_ret[k] if preds[k] == 0 else -raw_ret[k]
            net = sret - cost_arr[k]
            rets.append(net)
            trade_dates.append(dates.iloc[entry_idx[k]])
            wins += int(net > 0)
            n_long += int(preds[k] == 0); n_short += int(preds[k] == 1)
            k += hz  # hold to exit before re-entering (no overlap)
        else:
            k += 1
    rets = np.array(rets)
    n_tr = len(rets)
    if args.trades_csv and n_tr:
        pd.DataFrame({"entry": trade_dates, "net": rets}).to_csv(args.trades_csv, index=False)
        print(f"wrote {n_tr} trades → {args.trades_csv}")

    # Realistic position sizing: deploy a fixed fraction `risk_frac` of current
    # equity per trade (risk_frac=1.0 reproduces the unrealistic full-capital
    # compound; ~0.1 is a sane bet size). This is the credible equity metric.
    f = args.risk_frac
    eq = np.cumprod(1.0 + f * rets) if n_tr else np.array([1.0])
    total = (eq[-1] - 1.0) * 100 if n_tr else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100) if n_tr else 0.0
    buyhold = (close[exit_idx[-1]] / entry_px[0] - 1.0) * 100 if starts else 0.0

    # Span (years) for annualization, from first→last trade
    if n_tr >= 2:
        span_days = (pd.Timestamp(trade_dates[-1]) - pd.Timestamp(trade_dates[0])).total_seconds() / 86400
        years = max(span_days / 365.25, 1e-6)
        trades_per_year = n_tr / years
        sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(trades_per_year))
        cagr = (eq[-1] ** (1.0 / years) - 1.0) * 100
    else:
        sharpe, cagr, trades_per_year = 0.0, 0.0, 0.0

    if args.equity_csv and n_tr:
        pd.DataFrame({"entry": trade_dates, "equity": eq}).to_csv(args.equity_csv, index=False)

    metrics = {
        "model": args.model, "data": args.data, "threshold": args.threshold,
        "consensus": args.consensus, "base_tf": args.base_tf, "risk_frac": f, "cost_round_trip_pct": cost_frac * 100,
        "trades": n_tr, "long": n_long, "short": n_short,
        "directional_acc": round(dir_acc, 2),
        "net_win_pct": round(100 * wins / max(n_tr, 1), 2),
        "net_per_trade_pct": round(100 * rets.mean(), 4) if n_tr else 0.0,
        "total_return_pct": round(total, 2), "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2), "max_drawdown_pct": round(mdd, 2),
        "buy_hold_pct": round(buyhold, 2),
    }
    if args.metrics_json:
        import json
        with open(args.metrics_json, "w") as fh:
            json.dump(metrics, fh, indent=2)

    print("=" * 64)
    print(f"MODEL: {args.model}")
    print(f"DATA : {args.data}  ({dates.iloc[0]} → {dates.iloc[-1]})")
    print(f"input_dim={bt.input_dim} | windows={len(starts)} | horizon={hz} | thr={args.threshold} | consensus={args.consensus} | risk_frac={f}")
    print(f"cost round-trip = {cost_frac*100:.3f}%")
    print("-" * 64)
    print(f"FIDELITY — directional acc on {'golden' if args.consensus else 'all'} L/S calls: {dir_acc:.2f}%")
    print("-" * 64)
    print(f"trades (non-overlapping): {n_tr}  (long {n_long}, short {n_short}, {trades_per_year:.0f}/yr)")
    print(f"win rate (net>0)        : {metrics['net_win_pct']:.2f}%")
    print(f"avg net / trade         : {metrics['net_per_trade_pct']:.4f}%")
    print(f"Sharpe (annualized)     : {sharpe:.2f}")
    print(f"equity total (f={f})    : {total:+.2f}%   CAGR {cagr:+.2f}%   ← after costs, sized")
    print(f"max drawdown            : {mdd:.2f}%")
    print(f"buy & hold (same span)  : {buyhold:+.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
