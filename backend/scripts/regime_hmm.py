"""Lens 7: regime-switching (Gaussian HMM) — discover latent market states.

Unifies the ad-hoc regime work (vol tercile, corr regime) into DISCOVERED latent
states. Fits a Gaussian HMM on BTC daily [return, realized-vol] (BTC = the crypto
factor, PC1). Reports state means (calm/bull/stress), transition matrix +
persistence (expected duration), and the model's edge per regime on 2026 trades.

Run in container:
  docker exec saci-brain python -m scripts.regime_hmm
"""
from __future__ import annotations

import json
import glob

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM

from scripts.market_intel import hourly_logret

RESULTS = "/app/docs/study/results"
N_STATES = 3


def btc_daily():
    s = hourly_logret("BTCUSDM5").sort("date")
    d = (s.with_columns(pl.col("date").dt.truncate("1d").alias("day"))
          .group_by("day")
          .agg(pl.col("r").sum().alias("ret"), pl.col("r").abs().mean().alias("vol"))
          .sort("day").drop_nulls())
    return d


def main():
    d = btc_daily()
    X = d.select(["ret", "vol"]).to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)

    best = None
    for seed in range(8):  # multiple inits, keep best log-likelihood
        m = GaussianHMM(n_components=N_STATES, covariance_type="full",
                        n_iter=200, random_state=seed)
        try:
            m.fit(Xs)
            ll = m.score(Xs)
            if best is None or ll > best[0]:
                best = (ll, m)
        except Exception:
            continue
    m = best[1]
    states = m.predict(Xs)

    # label states by mean realized vol (col 1, raw scale)
    means_raw = np.array([X[states == k].mean(0) for k in range(N_STATES)])
    order = np.argsort(means_raw[:, 1])  # ascending vol
    label = {order[0]: "calm", order[1]: "normal", order[2]: "stress"}

    print(f"=== Gaussian HMM, {N_STATES} states, {len(states)} days (logL={best[0]:.0f}) ===")
    print(f"{'state':>8s} {'days':>5s} {'%':>5s}  {'mean ret/d':>11s} {'mean vol/d':>11s}")
    for k in range(N_STATES):
        n = int((states == k).sum())
        print(f"{label[k]:>8s} {n:>5d} {100*n/len(states):>4.0f}%  "
              f"{100*means_raw[k,0]:>+10.3f}% {100*means_raw[k,1]:>10.3f}%")

    print("\n=== transition matrix (rows=from, cols=to) + persistence ===")
    T = m.transmat_
    hdr = "  ".join(f"{label[j]:>7s}" for j in range(N_STATES))
    print(f"{'from\\to':>8s}  {hdr}")
    for i in range(N_STATES):
        row = "  ".join(f"{T[i,j]:7.2f}" for j in range(N_STATES))
        dur = 1 / (1 - T[i, i]) if T[i, i] < 1 else float("inf")
        print(f"{label[i]:>8s}  {row}   (stay {T[i,i]:.2f}, ~{dur:.1f}d)")

    # stationary distribution
    vals, vecs = np.linalg.eig(T.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); pi = pi / pi.sum()
    print("stationary:", "  ".join(f"{label[k]}={pi[k]:.2f}" for k in range(N_STATES)))

    # --- model edge per regime (2026 trades) ---
    reg = d.select(["day"]).with_columns(pl.Series("state", [label[s] for s in states]))
    files = sorted(glob.glob(f"{RESULTS}/trades_*_2026.csv"))
    frames = []
    for f in files:
        t = pl.read_csv(f, try_parse_dates=True).with_columns(
            pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None))
        frames.append(t.select(["entry", "net"]))
    tr = pl.concat(frames).with_columns(pl.col("entry").dt.truncate("1d").alias("day"))
    tr = tr.join(reg, on="day", how="inner")
    print(f"\n=== model edge by regime (2026, {tr.height} trades) ===")
    g = (tr.group_by("state").agg(
        pl.len().alias("n"),
        (100 * (pl.col("net") > 0).mean()).alias("netwin"),
        (100 * pl.col("net").mean()).alias("npt"))).to_dicts()
    rank = {"calm": 0, "normal": 1, "stress": 2}
    out_edge = {}
    for row in sorted(g, key=lambda r: rank[r["state"]]):
        print(f"  {row['state']:>7s}  n={row['n']:4d}  net-win={row['netwin']:5.1f}%  "
              f"net/trade={row['npt']:+.3f}%")
        out_edge[row["state"]] = {"n": row["n"], "netwin": round(row["netwin"], 1),
                                  "npt": round(row["npt"], 3)}

    out = {
        "states": {label[k]: {"days": int((states == k).sum()),
                              "mean_ret_d": round(float(means_raw[k, 0]), 5),
                              "mean_vol_d": round(float(means_raw[k, 1]), 5)} for k in range(N_STATES)},
        "transition": {label[i]: {label[j]: round(float(T[i, j]), 3) for j in range(N_STATES)}
                       for i in range(N_STATES)},
        "stationary": {label[k]: round(float(pi[k]), 3) for k in range(N_STATES)},
        "edge_2026_by_regime": out_edge,
    }
    with open(f"{RESULTS}/regime_hmm.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {RESULTS}/regime_hmm.json")


if __name__ == "__main__":
    main()
