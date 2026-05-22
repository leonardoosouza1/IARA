"""Lens 5: temporal structure of returns — why a directional model works, and
whether the high-edge regime is predictable.

Three sub-views (BTC/ETH/SOL/BNB, 1h log-returns, full history):
  A. Return autocorrelation + variance ratio — mean-reversion vs momentum by horizon
  B. Volatility persistence — is the high-vol (= high-edge) regime forecastable?
  C. Market clock — volatility/return seasonality by hour and weekday

Run in container:
  docker exec saci-brain python -m scripts.return_structure
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from scripts.market_intel import hourly_logret

ASSETS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
RESULTS = "/app/docs/study/results"


def autocorr(x, lags):
    x = x - x.mean()
    v = (x * x).mean()
    return [float((x[:-k] * x[k:]).mean() / v) for k in lags]


def variance_ratio(r, q):
    # VR(q) = Var(q-period ret)/(q*Var(1-period)); ~1 random walk, >1 trend, <1 revert
    n = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    varq = ((rq - q * mu) ** 2).sum() / (len(rq) - 1)
    return float(varq / (q * var1))


def main():
    series = {k: hourly_logret(v) for k, v in ASSETS.items()}
    out = {}

    print("=== A. return autocorrelation (1h, lags 1-6) ===")
    lags = [1, 2, 3, 4, 5, 6]
    print(f"{'asset':>6s} " + " ".join(f"lag{l:>2d}" for l in lags))
    acs = {}
    for k, s in series.items():
        ac = autocorr(s["r"].to_numpy(), lags)
        acs[k] = ac
        print(f"{k:>6s} " + " ".join(f"{a:+.3f}" for a in ac))
    out["autocorr_1h"] = {k: [round(a, 4) for a in v] for k, v in acs.items()}

    print("\n=== A. variance ratio (q hours) — >1 trend, <1 mean-revert ===")
    qs = [2, 4, 8, 12, 24, 48]
    print(f"{'asset':>6s} " + " ".join(f"q={q:>2d}" for q in qs))
    vrs = {}
    for k, s in series.items():
        r = s["r"].to_numpy()
        vr = [variance_ratio(r, q) for q in qs]
        vrs[k] = vr
        print(f"{k:>6s} " + " ".join(f"{v:.2f}" for v in vr))
    out["variance_ratio"] = {k: [round(a, 3) for a in v] for k, v in vrs.items()}

    print("\n=== B. volatility persistence: autocorr of |ret| (lags 1,6,24,72,168h) ===")
    vlags = [1, 6, 24, 72, 168]
    print(f"{'asset':>6s} " + " ".join(f"{l:>4d}h" for l in vlags))
    vps = {}
    for k, s in series.items():
        a = np.abs(s["r"].to_numpy())
        vp = autocorr(a, vlags)
        vps[k] = vp
        print(f"{k:>6s} " + " ".join(f"{v:+.3f}" for v in vp))
    out["vol_persistence_absret"] = {k: [round(a, 4) for a in v] for k, v in vps.items()}

    # forecastability of the high-vol regime: P(next-day high-vol | today high-vol)
    print("\n=== B. high-vol regime persistence (daily |ret|, top tercile) ===")
    for k, s in series.items():
        d = (s.with_columns(pl.col("date").dt.truncate("1d").alias("day"))
              .group_by("day").agg(pl.col("r").abs().mean().alias("v")).sort("day"))
        v = d["v"].to_numpy()
        hi = v > np.quantile(v, 2 / 3)
        p_base = hi.mean()
        p_cond = (hi[:-1] & hi[1:]).sum() / max(hi[:-1].sum(), 1)
        print(f"  {k:>6s}  P(hi)={p_base:.2f}  P(hi tomorrow | hi today)={p_cond:.2f}  "
              f"lift={p_cond - p_base:+.2f}")
        out.setdefault("highvol_persistence", {})[k] = {"p_base": round(float(p_base), 3),
                                                          "p_cond": round(float(p_cond), 3)}

    print("\n=== C. market clock: BTC volatility by hour (UTC) ===")
    s = series["BTC"]
    byh = (s.with_columns(pl.col("date").dt.hour().alias("h"))
            .group_by("h").agg((100 * pl.col("r").abs().mean()).alias("vol"),
                               (100 * pl.col("r").mean()).alias("ret")).sort("h"))
    rows = byh.to_dicts()
    vmax = max(r["vol"] for r in rows)
    for r in rows:
        bar = "#" * int(40 * r["vol"] / vmax)
        print(f"  {int(r['h']):02d}h  vol={r['vol']:.3f}%  {bar}")
    peak = sorted(rows, key=lambda r: -r["vol"])[:3]
    print(f"  peak vol hours (UTC): {[int(r['h']) for r in peak]}  "
          f"(US session — matches EDGE_FINDER §2)")
    out["btc_vol_by_hour"] = {int(r["h"]): round(r["vol"], 4) for r in rows}

    # plots: vol clock + |ret| autocorr decay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS = "/app/docs/study/plots"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2))
    hrs = [int(r["h"]) for r in rows]
    a1.bar(hrs, [r["vol"] for r in rows],
           color=["#d62728" if h in (13, 14, 15, 16) else "#1f77b4" for h in hrs])
    a1.set_title("BTC volatility by hour (UTC) — red=US session")
    a1.set_xlabel("hour UTC"); a1.set_ylabel("mean |1h ret| %")
    vl = [1, 6, 12, 24, 48, 72, 120, 168]
    for k, s in series.items():
        a2.plot(vl, autocorr(np.abs(s["r"].to_numpy()), vl), marker="o", ms=3, label=k)
    a2.axhline(0, color="k", lw=0.5)
    a2.set_title("Volatility persistence: autocorr of |ret|")
    a2.set_xlabel("lag (hours)"); a2.set_ylabel("autocorr"); a2.set_xscale("log"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{PLOTS}/return_structure.png", dpi=110); plt.close(fig)
    print(f"saved plot: {PLOTS}/return_structure.png")

    with open(f"{RESULTS}/return_structure.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {RESULTS}/return_structure.json")


if __name__ == "__main__":
    main()
