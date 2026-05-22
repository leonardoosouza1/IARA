"""Market intelligence: different lenses on the same data.

Beyond the directional model, this characterizes market STRUCTURE — how assets
relate, what drives them, when those relations break. Three lenses:

  1. Correlation network  — who moves together (clustered blocs)
  2. PCA factor model     — how many factors drive everything; the "market factor"
  3. Rolling correlation  — when correlations spike (risk-on/off / stress regimes)

All on 1h log-returns. Pairwise correlation uses each pair's own overlap (max
data); PCA/rolling use the globally-aligned core (assets with full overlap).

Run in container:
  docker exec saci-brain python -m scripts.market_intel
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

DATA = "/app/data/processed"
PLOTS = "/app/docs/study/plots"
RESULTS = "/app/docs/study/results"

# name -> parquet stem. Crypto4 + EURUSD have full 2021-2026 overlap (core).
# GBP/AUD only to 2023 → included; pairwise uses own overlap, core PCA drops them.
ASSETS = {
    "BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT",
    "BNB": "BNBUSDT", "EURUSD": "EURUSD", "GBPUSD": "GBPUSD",
}
CORE = ["BTC", "ETH", "SOL", "BNB", "EURUSD"]  # full-overlap subset for PCA/rolling


def hourly_logret(stem: str) -> pl.DataFrame:
    df = pl.read_parquet(f"{DATA}/{stem}.parquet").select(["date", "close"])
    df = df.with_columns(pl.col("date").dt.cast_time_unit("us").dt.replace_time_zone(None))
    h = (df.sort("date")
           .group_by_dynamic("date", every="1h")
           .agg(pl.col("close").last())
           .drop_nulls())
    h = h.with_columns((pl.col("close").log() - pl.col("close").log().shift(1)).alias("r")).drop_nulls()
    return h.select(["date", "r"])


def main():
    series = {k: hourly_logret(v) for k, v in ASSETS.items()}
    names = list(series.keys())

    # --- 1. pairwise correlation matrix (each pair on its own overlap) ---
    n = len(names)
    C = np.eye(n)
    overlap = {}
    for i in range(n):
        for j in range(i + 1, n):
            m = series[names[i]].join(series[names[j]], on="date", how="inner", suffix="_b")
            if m.height < 100:
                C[i, j] = C[j, i] = np.nan
                overlap[(names[i], names[j])] = m.height
                continue
            r = np.corrcoef(m["r"].to_numpy(), m["r_b"].to_numpy())[0, 1]
            C[i, j] = C[j, i] = r
            overlap[(names[i], names[j])] = m.height

    print("=== 1h return correlation (pairwise, own overlap) ===")
    print("        " + " ".join(f"{x:>7s}" for x in names))
    for i, x in enumerate(names):
        print(f"{x:>7s} " + " ".join(f"{C[i,j]:+7.2f}" if not np.isnan(C[i,j]) else "    nan" for j in range(n)))

    # hierarchical clustering on the non-nan core (blocs)
    cidx = [names.index(c) for c in CORE]
    Cc = C[np.ix_(cidx, cidx)]
    dist = 1 - np.abs(Cc)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(Z)
    clusters = fcluster(Z, t=0.6, criterion="distance")
    print("\n=== blocs (corr-distance clustering, core assets) ===")
    blocs = {}
    for c, lab in zip(CORE, clusters):
        blocs.setdefault(int(lab), []).append(c)
    for lab, members in sorted(blocs.items()):
        print(f"  bloc {lab}: {members}")

    # clustered heatmap PNG
    fig, ax = plt.subplots(figsize=(6, 5))
    ord_names = [CORE[i] for i in order]
    M = Cc[np.ix_(order, order)]
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(ord_names))); ax.set_xticklabels(ord_names, rotation=45)
    ax.set_yticks(range(len(ord_names))); ax.set_yticklabels(ord_names)
    for i in range(len(ord_names)):
        for j in range(len(ord_names)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="corr (1h ret)")
    ax.set_title("Correlation network (clustered)")
    fig.tight_layout(); fig.savefig(f"{PLOTS}/corr_network.png", dpi=110); plt.close(fig)

    # --- 2. PCA factor model on globally-aligned core ---
    df = series[CORE[0]].rename({"r": CORE[0]})
    for c in CORE[1:]:
        df = df.join(series[c].rename({"r": c}), on="date", how="inner")
    X = df.select(CORE).to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cov = np.cov(X.T)
    eig, vec = np.linalg.eigh(cov)
    idx = np.argsort(eig)[::-1]
    eig, vec = eig[idx], vec[:, idx]
    var_ratio = eig / eig.sum()
    print(f"\n=== PCA factor model (core {CORE}, {X.shape[0]} aligned 1h bars) ===")
    print("eigen var ratio:", " ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(var_ratio)))
    print(f"PC1 (market factor) explains {var_ratio[0]*100:.1f}% of variance")
    print("PC1 loadings:", " ".join(f"{c}={vec[k,0]:+.2f}" for k, c in enumerate(CORE)))
    print("PC2 loadings:", " ".join(f"{c}={vec[k,1]:+.2f}" for k, c in enumerate(CORE)))

    # --- 3. rolling correlation regime (mean pairwise corr over time) ---
    WIN = 720  # ~30 days of 1h bars
    crypto = [c for c in CORE if c not in ("EURUSD", "GBPUSD")]
    Xc = df.select(crypto).to_numpy()
    dates = df["date"].to_numpy()
    roll = []
    for t in range(WIN, Xc.shape[0]):
        w = Xc[t - WIN:t]
        cc = np.corrcoef(w.T)
        roll.append(cc[np.triu_indices(len(crypto), 1)].mean())
    roll = np.array(roll)
    rdates = dates[WIN:]
    print(f"\n=== rolling 30d mean crypto corr ({crypto}) ===")
    print(f"  mean {roll.mean():.2f}  min {roll.min():.2f}  max {roll.max():.2f}")
    print(f"  high-corr (>{roll.mean()+roll.std():.2f}) = stress/risk-off clustering")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(rdates, roll, lw=0.8)
    ax.axhline(roll.mean(), color="k", ls="--", lw=0.6, label=f"mean {roll.mean():.2f}")
    ax.axhline(roll.mean() + roll.std(), color="r", ls=":", lw=0.6, label="stress (+1σ)")
    ax.set_title("Rolling 30d mean crypto correlation (risk-on/off regime)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{PLOTS}/rolling_corr.png", dpi=110); plt.close(fig)

    out = {
        "corr_matrix": {names[i]: {names[j]: (None if np.isnan(C[i, j]) else round(float(C[i, j]), 3))
                                   for j in range(n)} for i in range(n)},
        "blocs": {str(k): v for k, v in blocs.items()},
        "pca_var_ratio": [round(float(v), 4) for v in var_ratio],
        "pc1_market_factor_pct": round(float(var_ratio[0] * 100), 1),
        "pc1_loadings": {c: round(float(vec[k, 0]), 3) for k, c in enumerate(CORE)},
        "rolling_corr": {"mean": round(float(roll.mean()), 3), "min": round(float(roll.min()), 3),
                          "max": round(float(roll.max()), 3)},
        "n_aligned_bars": int(X.shape[0]),
    }
    with open(f"{RESULTS}/market_intel.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {PLOTS}/corr_network.png, {PLOTS}/rolling_corr.png, {RESULTS}/market_intel.json")


if __name__ == "__main__":
    main()
