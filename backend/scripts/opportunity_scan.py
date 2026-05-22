"""Lens 12: OPPORTUNITY SCANNER — systematic, validated mining of hidden edge.

The other lenses each test ONE hypothesis. This scans the WHOLE conditioning space
at once: for every slice of the market (and every 2-way interaction), what is the
model's net/trade — and is the deviation real or noise?

Rigour built in so this is discovery, not data-snooping:
  - sample size floor (Nmin)             -> ignore tiny slices
  - t-stat of net/trade vs 0             -> is the edge statistically there?
  - OOS confirmation (train -> 2026)     -> the slice's edge must PERSIST out of
                                            sample with the SAME sign
  - multiple-testing guard               -> report #tests + Bonferroni alpha so a
                                            "significant" pocket is judged honestly

Conditioning dimensions (each tagged causal vs contemporaneous):
  causal (actionable, known before entry): session/hour, day-of-week, month,
    confidence decile, direction (long/short), coin, trailing vol_ratio (Lens 5),
    HMM regime (Lens 7), rolling crypto corr (Lens 3)
  contemporaneous (descriptive only — own-bar ATR has hindsight): atr tercile

Output: ranked OPPORTUNITIES (positive pockets confirmed OOS) and TRAPS (negative
pockets) — the hidden structure of where the edge lives and dies.

Run in container:
  docker exec saci-brain python -m scripts.opportunity_scan \
    --out_json docs/study/results/opportunity_scan.json
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations

import numpy as np
import polars as pl

from scripts.market_state import ASSETS, RESULTS, regime_by_day, corr_by_day, vol_feature


def load_trades() -> pl.DataFrame:
    """Pool every per-trade CSV, attach causal state (vol/regime/corr) per coin/day."""
    reg, cor = regime_by_day(), corr_by_day()
    frames = []
    for f in sorted(glob.glob(f"{RESULTS}/trades_*.csv")):
        d = pl.read_csv(f, try_parse_dates=True)
        if "conf" not in d.columns:
            continue
        coin = f.split("trades_")[-1].split("_")[0]
        if coin not in ASSETS:
            continue
        d = d.with_columns(pl.col("entry").cast(pl.Datetime("us")).dt.replace_time_zone(None),
                           coin=pl.lit(coin))
        vf = vol_feature(ASSETS[coin])
        d = d.sort("entry").join_asof(vf, left_on="entry", right_on="date", strategy="backward")
        frames.append(d.select(["entry", "net", "conf", "hour", "atr", "pred", "coin", "vol_ratio"]))
    tr = pl.concat(frames).with_columns(pl.col("entry").dt.truncate("1d").alias("day"))
    tr = tr.join(reg, on="day", how="left").join(cor, on="day", how="left")
    return tr.with_columns([
        pl.col("entry").dt.year().alias("yr"),
        pl.col("entry").dt.weekday().alias("dow"),       # 1=Mon..7=Sun
        pl.col("entry").dt.month().alias("month"),
        ((pl.col("hour") >= 13) & (pl.col("hour") < 20)).cast(pl.Int8).alias("us"),
        pl.when(pl.col("pred") == 0).then(pl.lit("LONG")).otherwise(pl.lit("SHORT")).alias("dir"),
    ])


def bin_cols(tr: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Discretise continuous dims into labelled bins. Returns (df, {dim: causal?})."""
    def terc(col, name):
        q1, q2 = tr[col].quantile(1/3), tr[col].quantile(2/3)
        return (pl.when(pl.col(col) <= q1).then(pl.lit(f"{name}:lo"))
                  .when(pl.col(col) <= q2).then(pl.lit(f"{name}:mid"))
                  .otherwise(pl.lit(f"{name}:hi")).alias(name))
    tr = tr.with_columns([
        terc("vol_ratio", "vol"), terc("corr", "corr"), terc("atr", "atr"),
        ("conf~" + ((pl.col("conf") * 20).floor() / 20).round(2).cast(pl.Utf8)).alias("confbin"),  # 0.05-wide
        pl.when(pl.col("regime") == 2).then(pl.lit("stress"))
          .when(pl.col("regime") == 1).then(pl.lit("normal"))
          .otherwise(pl.lit("calm")).alias("reg"),
        pl.when(pl.col("us") == 1).then(pl.lit("US")).otherwise(pl.lit("non-US")).alias("session"),
        ("dow=" + pl.col("dow").cast(pl.Utf8)).alias("dowL"),
        ("m=" + pl.col("month").cast(pl.Utf8)).alias("monthL"),
    ])
    causal = {"session": True, "dowL": True, "monthL": True, "confbin": True,
              "dir": True, "coin": True, "vol": True, "reg": True, "corr": True,
              "atr": False}  # atr = contemporaneous own-bar -> descriptive only
    return tr, causal


def tstat(x: np.ndarray) -> float:
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def scan(tr: pl.DataFrame, dims, causal, nmin: int, nmin_oos: int):
    """1-way + 2-way slice scan with OOS (2026) confirmation."""
    train = tr.filter(pl.col("yr") < 2026)
    test = tr.filter(pl.col("yr") == 2026)
    base_tr = 100 * train["net"].mean()
    base_te = 100 * test["net"].mean()
    results, n_tests = [], 0

    def eval_slice(label, keys, vals, is_causal):
        nonlocal n_tests
        def f(df):
            m = df
            for k, v in zip(keys, vals):
                m = m.filter(pl.col(k) == v)
            return m
        s_tr, s_te = f(train), f(test)
        if s_tr.height < nmin:
            return
        n_tests += 1
        net_tr = s_tr["net"].to_numpy()
        npt_tr, t_tr = 100 * net_tr.mean(), tstat(net_tr)
        # OOS confirm
        if s_te.height >= nmin_oos:
            npt_te = 100 * s_te["net"].mean()
            confirmed = np.sign(npt_tr - base_tr) == np.sign(npt_te - base_te)
        else:
            npt_te, confirmed = float("nan"), False
        results.append({
            "slice": label, "causal": bool(is_causal),
            "n_train": int(s_tr.height), "net_tr": round(npt_tr, 4),
            "edge_tr": round(npt_tr - base_tr, 4), "t": round(t_tr, 2),
            "n_test": int(s_te.height), "net_te": round(npt_te, 4) if s_te.height >= nmin_oos else None,
            "edge_te": round(npt_te - base_te, 4) if s_te.height >= nmin_oos else None,
            "oos_confirmed": bool(confirmed),
        })

    # 1-way
    for d in dims:
        for v in train[d].unique().to_list():
            if v is None:
                continue
            eval_slice(f"{d}={v}", [d], [v], causal[d])
    # 2-way interactions (causal dims only — actionable)
    cdims = [d for d in dims if causal[d]]
    for a, b in combinations(cdims, 2):
        va = train[a].unique().to_list(); vb = train[b].unique().to_list()
        if len(va) * len(vb) > 60:   # skip explosive crosses (month×coin etc)
            continue
        for x in va:
            for y in vb:
                if x is None or y is None:
                    continue
                eval_slice(f"{a}={x} & {b}={y}", [a, b], [x, y], True)
    return results, n_tests, base_tr, base_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=40, help="min train trades for a slice")
    ap.add_argument("--nmin_oos", type=int, default=15, help="min 2026 trades to confirm OOS")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    tr, causal = bin_cols(load_trades())
    dims = list(causal.keys())
    results, n_tests, base_tr, base_te = scan(tr, dims, causal, args.nmin, args.nmin_oos)

    alpha = 0.05 / max(n_tests, 1)
    t_bonf = 1.96 + (3.5 if n_tests > 200 else 2.6 if n_tests > 50 else 1.0)  # rough Bonferroni-z
    print("=" * 78)
    print(f"OPPORTUNITY SCAN — {tr.height} trades pooled  "
          f"(train<2026 base {base_tr:+.3f}%/tr, 2026 base {base_te:+.3f}%/tr)")
    print(f"slices tested: {n_tests}   Bonferroni alpha={alpha:.2e}  (|t| gate ~{t_bonf:.1f})")
    print("=" * 78)

    conf = [r for r in results if r["oos_confirmed"]]
    # OPPORTUNITIES: positive train edge, strong t, confirmed OOS
    opp = sorted([r for r in conf if r["edge_tr"] > 0 and r["t"] >= t_bonf and (r["edge_te"] or 0) > 0],
                 key=lambda r: r["edge_te"], reverse=True)
    print(f"\n### OPPORTUNITIES (edge>base, |t|>={t_bonf:.1f}, confirmed OOS same-sign) — top 12")
    print(f"{'slice':<34s}{'n_tr':>5s}{'net_tr':>9s}{'t':>6s}{'n_te':>5s}{'net_te':>9s}{'c':>3s}")
    for r in opp[:12]:
        print(f"{r['slice']:<34s}{r['n_train']:>5d}{r['net_tr']:>+8.3f}%{r['t']:>6.1f}"
              f"{r['n_test']:>5d}{r['net_te']:>+8.3f}%{'C' if r['causal'] else 'd':>3s}")

    # TRAPS: negative or far-below-base edge, confirmed OOS
    trap = sorted([r for r in conf if r["edge_tr"] < 0 and r["t"] <= -t_bonf and (r["edge_te"] or 0) < 0],
                  key=lambda r: r["edge_te"])
    print(f"\n### TRAPS (edge<base, |t|>={t_bonf:.1f}, confirmed OOS same-sign) — top 12")
    print(f"{'slice':<34s}{'n_tr':>5s}{'net_tr':>9s}{'t':>6s}{'n_te':>5s}{'net_te':>9s}{'c':>3s}")
    for r in trap[:12]:
        print(f"{r['slice']:<34s}{r['n_train']:>5d}{r['net_tr']:>+8.3f}%{r['t']:>6.1f}"
              f"{r['n_test']:>5d}{r['net_te']:>+8.3f}%{'C' if r['causal'] else 'd':>3s}")

    # COLDEST confirmed pockets (lowest OOS net/trade) — where the edge thins,
    # even if still positive. The "do less here" map (no money-losers cleared the
    # bar => the strategy has no confirmed systematic loss-zone).
    cold = sorted([r for r in conf if r["net_te"] is not None], key=lambda r: r["net_te"])[:8]
    print(f"\n### COLDEST confirmed pockets (lowest OOS net/trade) — where edge thins")
    print(f"{'slice':<34s}{'n_tr':>5s}{'net_tr':>9s}{'t':>6s}{'n_te':>5s}{'net_te':>9s}{'c':>3s}")
    for r in cold:
        print(f"{r['slice']:<34s}{r['n_train']:>5d}{r['net_tr']:>+8.3f}%{r['t']:>6.1f}"
              f"{r['n_test']:>5d}{r['net_te']:>+8.3f}%{'C' if r['causal'] else 'd':>3s}")

    # honesty: how many that looked great in-sample FAILED to confirm OOS?
    looked = [r for r in results if r["edge_tr"] > 0 and r["t"] >= t_bonf and r["net_te"] is not None]
    failed = [r for r in looked if not (r["edge_te"] and r["edge_te"] > 0)]
    print(f"\n### snooping check: {len(looked)} pockets significant in-sample, "
          f"{len(failed)} FAILED to confirm OOS ({100*len(failed)/max(len(looked),1):.0f}% were noise)")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"n_trades": tr.height, "base_train": round(base_tr, 4),
                       "base_test": round(base_te, 4), "n_tests": n_tests,
                       "t_gate": round(t_bonf, 2),
                       "opportunities": opp[:20], "traps": trap[:20], "coldest": cold,
                       "snoop_failed_pct": round(100*len(failed)/max(len(looked),1), 1)}, fh, indent=2)
        print(f"\nsaved: {args.out_json}")
    print("=" * 78)


if __name__ == "__main__":
    main()
