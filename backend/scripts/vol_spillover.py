"""Lens 6: volatility spillover — does BTC volatility LEAD altcoin volatility?

CROSS_ASSET showed crypto RETURNS have no lead-lag (efficient: co-move at k=0,
~0 at k>=1). But VOLATILITY famously spills over with lead-lag. If BTC vol leads
alt vol, that's a genuinely actionable alert — exactly Leonardo's original intuition
("BTC moves -> alert on alts"), but in the volatility domain where it may hold.

Two tests on hourly |log-ret| (vol proxy), full history, aligned:
  1. cross-lagged correlation  corr(A_vol[t], B_vol[t+k]), k=0..6, both directions
  2. Granger causality (manual OLS + F-test, no statsmodels): does past A_vol
     improve prediction of B_vol beyond B's own past?

Run in container:
  docker exec saci-brain python -m scripts.vol_spillover
"""
from __future__ import annotations

import json
import itertools

import numpy as np
import polars as pl
from scipy import stats

from scripts.market_intel import hourly_logret

ASSETS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
RESULTS = "/app/docs/study/results"


def granger_f(y, x, p=6):
    """Does x Granger-cause y? Restricted: y~y_lags. Unrestricted: y~y_lags+x_lags.
    Returns (F, pvalue). Larger F / smaller p => x helps predict y."""
    n = len(y)
    rows = n - p
    Yt = y[p:]
    Yl = np.column_stack([y[p - k:n - k] for k in range(1, p + 1)])
    Xl = np.column_stack([x[p - k:n - k] for k in range(1, p + 1)])
    ones = np.ones((rows, 1))
    Zr = np.hstack([ones, Yl])              # restricted
    Zu = np.hstack([ones, Yl, Xl])          # unrestricted
    br, *_ = np.linalg.lstsq(Zr, Yt, rcond=None)
    bu, *_ = np.linalg.lstsq(Zu, Yt, rcond=None)
    rss_r = float(((Yt - Zr @ br) ** 2).sum())
    rss_u = float(((Yt - Zu @ bu) ** 2).sum())
    df1, df2 = p, rows - Zu.shape[1]
    F = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return F, float(stats.f.sf(F, df1, df2))


def main():
    vol = {k: hourly_logret(v).with_columns(pl.col("r").abs().alias("v")).select(["date", "v"])
           for k, v in ASSETS.items()}
    # align all on common timestamps
    df = vol["BTC"].rename({"v": "BTC"})
    for k in ["ETH", "SOL", "BNB"]:
        df = df.join(vol[k].rename({"v": k}), on="date", how="inner")
    names = list(ASSETS.keys())
    arr = {n: df[n].to_numpy() for n in names}
    print(f"aligned {df.height} hourly vol obs, assets={names}\n")

    # --- 1. cross-lagged vol correlation (directional) ---
    print("=== cross-lagged VOL correlation: corr(A_vol[t], B_vol[t+k]) ===")
    print("(k=0 contemporaneous; k>=1 means A LEADS B)")
    K = 6
    out = {"crosslag": {}}
    for a, b in itertools.permutations(names, 2):
        x, y = arr[a], arr[b]
        cs = []
        for k in range(0, K + 1):
            if k == 0:
                cs.append(np.corrcoef(x, y)[0, 1])
            else:
                cs.append(np.corrcoef(x[:-k], y[k:])[0, 1])
        out["crosslag"][f"{a}->{b}"] = [round(float(c), 3) for c in cs]
    # print BTC -> alts and the reverse for asymmetry
    print(f"{'pair':>10s}  " + "  ".join(f"k={k}" for k in range(K + 1)))
    for a, b in [("BTC", "ETH"), ("ETH", "BTC"), ("BTC", "SOL"), ("SOL", "BTC"),
                 ("BTC", "BNB"), ("BNB", "BTC")]:
        cs = out["crosslag"][f"{a}->{b}"]
        print(f"{a+'->'+b:>10s}  " + "  ".join(f"{c:+.2f}" for c in cs))

    # --- 2. Granger causality (both directions) ---
    print("\n=== Granger causality on VOL (p=6 lags, F-test) ===")
    print("does ROW's past vol help predict COL's vol beyond COL's own past?")
    out["granger"] = {}
    print(f"{'cause->effect':>14s}  {'F':>8s}  {'p-value':>10s}  signif")
    for a, b in itertools.permutations(names, 2):
        F, pv = granger_f(arr[b], arr[a], p=6)  # a -> b
        sig = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 0.05 else ""))
        out["granger"][f"{a}->{b}"] = {"F": round(F, 2), "p": pv}
        print(f"{a+'->'+b:>14s}  {F:8.1f}  {pv:10.2e}  {sig}")

    # --- 3. directional dominance: net BTC leadership ---
    print("\n=== spillover summary ===")
    for alt in ["ETH", "SOL", "BNB"]:
        f_btc_alt = out["granger"][f"BTC->{alt}"]["F"]
        f_alt_btc = out["granger"][f"{alt}->BTC"]["F"]
        lead = "BTC leads" if f_btc_alt > f_alt_btc else f"{alt} leads"
        print(f"  BTC<->{alt}: F(BTC->{alt})={f_btc_alt:.0f} vs F({alt}->BTC)={f_alt_btc:.0f}  => {lead}")

    # --- 4. practical payoff: does cross-asset vol improve a coin's vol forecast? ---
    print("\n=== cross-asset vol nowcast: predict target vol[t] from lags (OOS R²) ===")
    p = 6
    out["nowcast_oos_r2"] = {}
    for tgt in names:
        y = arr[tgt]
        n = len(y)
        Yl = np.column_stack([y[p - k:n - k] for k in range(1, p + 1)])
        others = [c for c in names if c != tgt]
        Xl = np.column_stack([arr[c][p - k:n - k] for c in others for k in range(1, p + 1)])
        Yt = y[p:]
        rows = len(Yt); split = int(rows * 0.7)
        def oos_r2(Z):
            Ztr, Zte = Z[:split], Z[split:]
            b, *_ = np.linalg.lstsq(np.hstack([np.ones((len(Ztr), 1)), Ztr]), Yt[:split], rcond=None)
            pred = np.hstack([np.ones((len(Zte), 1)), Zte]) @ b
            ss_res = ((Yt[split:] - pred) ** 2).sum()
            ss_tot = ((Yt[split:] - Yt[:split].mean()) ** 2).sum()
            return 1 - ss_res / ss_tot
        r2_own = oos_r2(Yl)
        r2_all = oos_r2(np.hstack([Yl, Xl]))
        out["nowcast_oos_r2"][tgt] = {"own": round(float(r2_own), 4), "own+cross": round(float(r2_all), 4)}
        print(f"  {tgt:>5s}: own-lags R²={r2_own:.3f}   +cross-asset R²={r2_all:.3f}   "
              f"gain={r2_all-r2_own:+.3f}")

    with open(f"{RESULTS}/vol_spillover.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {RESULTS}/vol_spillover.json")


if __name__ == "__main__":
    main()
