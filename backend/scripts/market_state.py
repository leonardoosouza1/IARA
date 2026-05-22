"""Lens 10 (capstone): the FUSION — a market-state vector + a unified conditioner.

Pulls every lens into one causal state vector per trade and asks the synthesis
question: does fusing market state select better trades than the model's
confidence ALONE — and does it generalize out-of-sample?

State vector per trade (all known at/before entry):
  - vol_ratio : trailing-24h vol / expanding median  (Lens 5/vol_gate, causal)
  - corr      : rolling-30d mean crypto correlation   (Lens 3)
  - regime    : HMM stress/normal/calm dummies         (Lens 7)
  - us        : US-session hour flag                    (Lens 5C)
  - conf      : model confidence                        (the model itself)

Fusion = logistic regression of P(net>0) on the state vector. Trained on
2024-2025 (BTC), tested on 2026 (4 coins) — a true temporal + cross-coin holdout.
Compares conf-only vs full-state to show the fusion adds beyond confidence.

Run in container:
  docker exec saci-brain python -m scripts.market_state
"""
from __future__ import annotations

import json
import glob

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from scripts.market_intel import hourly_logret

ASSETS = {"BTC": "BTCUSDM5", "ETH": "ETHUSDM5", "SOL": "SOLUSDT", "BNB": "BNBUSDT"}
RESULTS = "/app/docs/study/results"

_VOL_CACHE = {}


def vol_feature(stem: str) -> pl.DataFrame:
    """Causal vol ratio = trailing-24h mean|ret| / trailing-720h mean|ret|, both
    ending strictly BEFORE the bar (shift 1). O(n) via cumsum (no expanding median)."""
    if stem in _VOL_CACHE:
        return _VOL_CACHE[stem]
    s = hourly_logret(stem).sort("date")
    a = np.abs(s["r"].to_numpy()); n = len(a)
    cs = np.concatenate([[0], np.cumsum(a)])
    def trail(W):
        out = np.full(n, np.nan)
        for t in range(W, n):
            out[t] = (cs[t] - cs[t - W]) / W
        return out
    short, long = trail(24), trail(720)
    ratio = np.full(n, np.nan)
    mask = ~np.isnan(short) & ~np.isnan(long) & (long > 0)
    ratio[mask] = short[mask] / long[mask]
    ratio = np.concatenate([[np.nan], ratio[:-1]])  # known before the bar
    df = pl.DataFrame({"date": s["date"].to_numpy(), "vol_ratio": ratio}).drop_nulls()
    _VOL_CACHE[stem] = df
    return df


def regime_by_day():
    s = hourly_logret("BTCUSDM5").sort("date")
    d = (s.with_columns(pl.col("date").dt.truncate("1d").alias("day"))
          .group_by("day").agg(pl.col("r").sum().alias("ret"), pl.col("r").abs().mean().alias("vol"))
          .sort("day").drop_nulls())
    X = d.select(["ret", "vol"]).to_numpy()
    Xs = (X - X.mean(0)) / X.std(0)
    best = None
    for seed in range(8):
        m = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=seed)
        try:
            m.fit(Xs); ll = m.score(Xs)
            if best is None or ll > best[0]: best = (ll, m)
        except Exception: pass
    st = best[1].predict(Xs)
    vol_rank = np.argsort(np.argsort([X[st == k, 1].mean() for k in range(3)]))  # 0 calm..2 stress
    lab = np.array([vol_rank[s_] for s_ in st])  # 0 calm,1 normal,2 stress
    return d.select("day").with_columns(pl.Series("regime", lab))


def corr_by_day(win=30):
    s = {c: hourly_logret(v).sort("date")
            .with_columns(pl.col("date").dt.truncate("1d").alias("day"))
            .group_by("day").agg(pl.col("r").sum().alias(c)).sort("day")
         for c, v in ASSETS.items()}
    df = s["BTC"]
    for c in ["ETH", "SOL", "BNB"]:
        df = df.join(s[c], on="day", how="inner")
    A = df.select(["BTC", "ETH", "SOL", "BNB"]).to_numpy()
    iu = np.triu_indices(4, 1)
    out = np.full(len(A), np.nan)
    for t in range(win, len(A)):
        out[t] = np.corrcoef(A[t - win:t].T)[iu].mean()
    return df.select("day").with_columns(pl.Series("corr", out)).drop_nulls()


def main():
    reg = regime_by_day()
    cor = corr_by_day()

    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/trades_*.csv")):
        d = pl.read_csv(f, try_parse_dates=True)
        if "conf" not in d.columns:
            continue
        coin = f.split("trades_")[-1].split("_")[0]
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None),
                           coin=pl.lit(coin))
        # causal vol ratio for this coin (cached)
        vf = vol_feature(ASSETS[coin])
        d = d.sort("entry").join_asof(vf, left_on="entry", right_on="date", strategy="backward")
        frames.append(d.select(["entry", "net", "conf", "hour", "coin", "vol_ratio"]))
    tr = pl.concat(frames).with_columns(pl.col("entry").dt.truncate("1d").alias("day"))
    tr = tr.join(reg, on="day", how="left").join(cor, on="day", how="left").drop_nulls(["vol_ratio", "corr", "regime"])

    tr = tr.with_columns([
        ((pl.col("hour") >= 13) & (pl.col("hour") < 20)).cast(pl.Int8).alias("us"),
        (pl.col("regime") == 2).cast(pl.Int8).alias("stress"),
        (pl.col("regime") == 1).cast(pl.Int8).alias("normal"),
        (pl.col("net") > 0).cast(pl.Int8).alias("win"),
        pl.col("entry").dt.year().alias("yr")])

    feat = ["vol_ratio", "corr", "conf", "us", "stress", "normal"]
    train = tr.filter(pl.col("yr") < 2026)
    test = tr.filter(pl.col("yr") == 2026)
    print(f"train (2024-2025, BTC): {train.height} trades   test (2026, 4 coins): {test.height} trades")

    Xtr, ytr = train.select(feat).to_numpy(), train["win"].to_numpy()
    Xte, yte = test.select(feat).to_numpy(), test["win"].to_numpy()
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd

    # conf-only baseline vs full-state fusion
    ci = feat.index("conf")
    lr_conf = LogisticRegression(max_iter=1000).fit(Xtr_n[:, [ci]], ytr)
    lr_full = LogisticRegression(max_iter=1000).fit(Xtr_n, ytr)
    p_conf = lr_conf.predict_proba(Xte_n[:, [ci]])[:, 1]
    p_full = lr_full.predict_proba(Xte_n)[:, 1]
    auc_conf = roc_auc_score(yte, p_conf)
    auc_full = roc_auc_score(yte, p_full)
    print(f"\n=== OOS (2026) discrimination of net>0 ===")
    print(f"  conf-only      AUC = {auc_conf:.3f}")
    print(f"  full-state     AUC = {auc_full:.3f}   (Δ{auc_full-auc_conf:+.3f})")
    print("  coefs (full, standardized): " + ", ".join(f"{f}={c:+.2f}" for f, c in zip(feat, lr_full.coef_[0])))

    # net/trade by fused-quality quartile (the deliverable)
    net = test["net"].to_numpy()
    print(f"\n=== net/trade by fused state-score quartile (2026 holdout) ===")
    for name, p in [("conf-only", p_conf), ("full-state", p_full)]:
        q = np.searchsorted(np.quantile(p, [.25, .5, .75]), p)
        print(f"  [{name}]")
        for k in range(4):
            m = q == k
            print(f"    Q{k+1}  n={m.sum():4d}  net-win={100*(net[m]>0).mean():5.1f}%  net/trade={100*net[m].mean():+.3f}%")

    # actionable gate: trade only top-half fused score
    top = p_full >= np.median(p_full)
    print(f"\n=== fused gate: trade only top-half state-score (2026) ===")
    print(f"  baseline  n={len(net):4d}  net/trade {100*net.mean():+.3f}%")
    print(f"  top-half  n={top.sum():4d}  net/trade {100*net[top].mean():+.3f}%  (Δ{100*(net[top].mean()-net.mean()):+.3f}pp)")

    out = {"auc_conf_only": round(float(auc_conf), 3), "auc_full_state": round(float(auc_full), 3),
           "coefs": {f: round(float(c), 3) for f, c in zip(feat, lr_full.coef_[0])},
           "tophalf_net_per_trade": round(float(100 * net[top].mean()), 3),
           "baseline_net_per_trade": round(float(100 * net.mean()), 3)}
    with open(f"{RESULTS}/market_state.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {RESULTS}/market_state.json")


if __name__ == "__main__":
    main()
