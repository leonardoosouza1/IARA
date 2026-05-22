"""Lens 14: RISK-BASED SIZING — the deployable lever the whole study points to.

Synthesis of the verdicts: direction is random-walk (can't predict better), but
amplitude/regime/tail-risk ARE structured. The model's edge is MAX in the stress
regime (Lens 12, +1.0%/tr) — exactly where tail risk is MAX (Lens 8, tail-λ 0.61,
crypto crashes together). Edge and risk coincide. The backtest today sizes with a
FIXED fraction; sizing has never been measured as a P&L lever.

This compares, at EQUAL AVERAGE EXPOSURE (fair — same aggression), three sizing
policies on the 2026 OOS trades:
  1. fixed         — constant fraction (today's baseline)
  2. vol-target    — fraction ∝ 1/ATR  (constant risk per trade)
  3. vol+stresscap — vol-target, then rotate exposure OUT of the HMM stress regime
                     (survive the co-crash even though edge there is high)

Per-trade net is unchanged by sizing; what changes is the equity path. The win to
look for: higher Sharpe and smaller max drawdown at the same mean exposure (better
risk-adjusted return, not more betting). Sharpe is sample-based (robust); CAGR/maxDD
use sequential compounding by entry time (approx — ignores concurrent cross-coin
capital, noted).

Run in container:
  docker exec saci-brain python -m scripts.risk_sizing \
    --out_json docs/study/results/risk_sizing.json
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import polars as pl

from scripts.market_state import regime_by_day, RESULTS


def load_2026_trades():
    reg = regime_by_day()
    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/trades_*_2026.csv")):
        d = pl.read_csv(f, try_parse_dates=True)
        if "atr" not in d.columns:
            continue
        coin = f.split("trades_")[-1].split("_")[0]
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None),
                           coin=pl.lit(coin))
        frames.append(d.select(["entry", "net", "atr", "conf", "coin"]))
    tr = pl.concat(frames).with_columns(pl.col("entry").dt.truncate("1d").alias("day"))
    tr = tr.join(reg, on="day", how="left").sort("entry")
    return tr


def metrics(f, net, years):
    """Equity, Sharpe, maxDD, CAGR, Calmar for a sizing vector f (per trade)."""
    pnl = f * net                      # portfolio per-trade return contribution
    eq = np.cumprod(1.0 + pnl)
    tpy = len(net) / years
    sharpe = float(pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(tpy))
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min() * 100)
    cagr = float((eq[-1] ** (1.0 / years) - 1.0) * 100)
    total = float((eq[-1] - 1.0) * 100)
    calmar = cagr / abs(mdd) if mdd != 0 else float("inf")
    return {"sharpe": round(sharpe, 2), "max_drawdown_pct": round(mdd, 2),
            "cagr_pct": round(cagr, 1), "total_pct": round(total, 1),
            "calmar": round(calmar, 2), "mean_exposure": round(float(f.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", type=float, default=0.10, help="target average exposure per trade")
    ap.add_argument("--clip", type=float, default=3.0, help="max/min multiple of f0 (leverage guard)")
    ap.add_argument("--stress_factor", type=float, default=0.5, help="exposure multiplier in stress regime")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    tr = load_2026_trades()
    net = tr["net"].to_numpy()
    atr = tr["atr"].to_numpy()
    conf = tr["conf"].to_numpy()
    regime = tr["regime"].fill_null(1).to_numpy()        # 0 calm,1 normal,2 stress
    n = len(net)
    span = (tr["entry"].max() - tr["entry"].min()).total_seconds() / 86400
    years = max(span / 365.25, 1e-6)
    print(f"pooled 2026 trades: {n}  span {years:.2f}y  base net/trade {100*net.mean():+.4f}%  "
          f"stress trades {(regime==2).sum()}")

    def normalize(w):
        """Scale raw weights so MEAN exposure == f0 (equal aggression), clip leverage."""
        w = np.clip(w, 1e-9, None)
        f = args.f0 * w / w.mean()
        return np.clip(f, args.f0 / args.clip, args.f0 * args.clip)

    # 1. fixed
    f_fixed = np.full(n, args.f0)
    # 2. vol-target: inverse ATR (constant risk per trade)
    inv_vol = 1.0 / np.clip(atr, 1e-6, None)
    f_vt = normalize(inv_vol)
    # 3. vol-target + stress cap: down-weight stress BEFORE normalizing (rotates the
    #    freed exposure into calm/normal -> same mean exposure, less tail risk)
    w_sc = inv_vol.copy()
    w_sc[regime == 2] *= args.stress_factor
    f_sc = normalize(w_sc)
    # 4. conf-scale: size ∝ calibrated confidence (Lens 9, ρ=0.99) — the actual edge driver
    f_cf = normalize(np.clip(conf - 0.5, 1e-3, None))
    # 5. anti-vol: size ∝ ATR — lean INTO high-vol where the edge lives (Lens 12).
    #    Tests whether the crypto edge-vol coupling rewards the OPPOSITE of textbook.
    f_av = normalize(np.clip(atr, 1e-6, None))

    schemes = {"fixed": f_fixed, "vol_target": f_vt, "vol_target+stress_cap": f_sc,
               "conf_scale": f_cf, "anti_vol (lean in)": f_av}
    out = {"trades": n, "years": round(years, 2), "f0": args.f0,
           "base_net_per_trade_pct": round(100 * net.mean(), 4), "schemes": {}}

    print(f"\n{'scheme':<24s}{'Sharpe':>8s}{'maxDD%':>9s}{'CAGR%':>9s}{'Calmar':>8s}{'meanExp':>9s}")
    for name, f in schemes.items():
        m = metrics(f, net, years)
        out["schemes"][name] = m
        print(f"{name:<24s}{m['sharpe']:>8.2f}{m['max_drawdown_pct']:>9.2f}"
              f"{m['cagr_pct']:>9.1f}{m['calmar']:>8.2f}{m['mean_exposure']:>9.4f}")

    # headline deltas vs fixed (same mean exposure)
    base = out["schemes"]["fixed"]
    print("-" * 67)
    # rank by Calmar (return per unit drawdown) — the deployable risk-adjusted metric
    ranked = sorted(out["schemes"].items(), key=lambda kv: -kv[1]["calmar"])
    print("ranked by Calmar (CAGR/|maxDD|):")
    for name, m in ranked:
        tag = " <- baseline" if name == "fixed" else ""
        print(f"  {name:<24s} Calmar {m['calmar']:>8.2f}  Sharpe {m['sharpe']:.2f}  CAGR {m['cagr_pct']:.0f}%{tag}")
    out["winner_by_calmar"] = ranked[0][0]

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
