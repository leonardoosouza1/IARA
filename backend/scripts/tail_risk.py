"""Lens 8: tail risk — the risk axis of the market-state vector.

Four risk views (BTC/ETH/SOL/BNB):
  1. Return distribution — skew, excess kurtosis, fat tails (daily)
  2. VaR / CVaR (Expected Shortfall) — historical, 95% & 99% (daily)
  3. Drawdown — max depth, longest duration, recovery (on close price)
  4. Lower-tail dependence — do crashes co-occur MORE than correlation implies?
     λ = P(B in worst q% | A in worst q%); λ >> q means diversification fails in
     the tail (complements Lens 3: correlation spikes in stress).

Run in container:
  docker exec saci-brain python -m scripts.tail_risk
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
from scipy import stats

from scripts.market_intel import hourly_logret

ASSETS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
RESULTS = "/app/docs/study/results"


def daily_ret(stem):
    s = hourly_logret(stem).sort("date")
    d = (s.with_columns(pl.col("date").dt.truncate("1d").alias("day"))
          .group_by("day").agg(pl.col("r").sum().alias("r")).sort("day"))
    return d


def max_drawdown(logret):
    eq = np.cumsum(logret)              # log-equity
    peak = np.maximum.accumulate(eq)
    dd = eq - peak                      # log drawdown (<=0)
    trough = int(dd.argmin())
    mdd = float(np.expm1(dd[trough]))   # back to simple return
    peak_i = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    # longest underwater duration (bars below a running peak)
    under = dd < -1e-9
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return mdd, peak_i, trough, longest


def main():
    series = {k: daily_ret(v) for k, v in ASSETS.items()}
    names = list(ASSETS.keys())
    out = {}

    print("=== 1. daily return distribution ===")
    print(f"{'asset':>6s} {'mean%':>7s} {'std%':>7s} {'skew':>7s} {'exkurt':>7s} {'min%':>8s} {'max%':>8s}")
    for k in names:
        r = series[k]["r"].to_numpy()
        rp = 100 * (np.expm1(r))  # simple % per day
        sk = float(stats.skew(rp)); ku = float(stats.kurtosis(rp))  # excess
        print(f"{k:>6s} {rp.mean():>+6.3f} {rp.std():>7.2f} {sk:>+7.2f} {ku:>7.1f} "
              f"{rp.min():>+8.2f} {rp.max():>+8.2f}")
        out.setdefault("distribution", {})[k] = {"skew": round(sk, 3), "exkurt": round(ku, 2),
                                                  "std_pct": round(float(rp.std()), 3)}

    print("\n=== 2. VaR / CVaR (historical daily, loss is negative) ===")
    print(f"{'asset':>6s} {'VaR95':>8s} {'CVaR95':>8s} {'VaR99':>8s} {'CVaR99':>8s}")
    for k in names:
        rp = 100 * np.expm1(series[k]["r"].to_numpy())
        v95, v99 = np.percentile(rp, [5, 1])
        c95 = rp[rp <= v95].mean(); c99 = rp[rp <= v99].mean()
        print(f"{k:>6s} {v95:>+8.2f} {c95:>+8.2f} {v99:>+8.2f} {c99:>+8.2f}")
        out.setdefault("var_cvar", {})[k] = {"var95": round(float(v95), 2), "cvar95": round(float(c95), 2),
                                             "var99": round(float(v99), 2), "cvar99": round(float(c99), 2)}

    print("\n=== 3. drawdown (buy & hold, full history) ===")
    print(f"{'asset':>6s} {'maxDD%':>8s} {'underwater(days)':>17s}")
    for k in names:
        r = series[k]["r"].to_numpy()
        mdd, pi, tr, longest = max_drawdown(r)
        print(f"{k:>6s} {100*mdd:>+8.1f} {longest:>17d}")
        out.setdefault("drawdown", {})[k] = {"max_dd_pct": round(100 * mdd, 1), "underwater_days": int(longest)}

    print("\n=== 4. lower-tail dependence λ = P(B worst 5% | A worst 5%) ===")
    print("(independence baseline = 0.05; higher = crashes co-occur)")
    # align daily returns
    df = series[names[0]].rename({"r": names[0]})
    for k in names[1:]:
        df = df.join(series[k].rename({"r": k}), on="day", how="inner")
    A = {k: df[k].to_numpy() for k in names}
    q = 0.05
    print(f"{'':>6s} " + " ".join(f"{x:>6s}" for x in names))
    lam = {}
    for a in names:
        thr_a = np.percentile(A[a], q * 100)
        bad_a = A[a] <= thr_a
        row = []
        for b in names:
            if a == b:
                row.append(1.0); continue
            thr_b = np.percentile(A[b], q * 100)
            bad_b = A[b] <= thr_b
            l = (bad_a & bad_b).sum() / max(bad_a.sum(), 1)
            row.append(l); lam[f"{a}|{b}"] = round(float(l), 3)
        print(f"{a:>6s} " + " ".join(f"{v:>6.2f}" for v in row))
    out["lower_tail_dep"] = lam
    off = [v for k, v in lam.items()]
    print(f"\nmean off-diagonal λ = {np.mean(off):.2f}  (vs 0.05 independence) "
          f"→ {np.mean(off)/q:.0f}× more co-crashing than independent")

    with open(f"{RESULTS}/tail_risk.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {RESULTS}/tail_risk.json")


if __name__ == "__main__":
    main()
