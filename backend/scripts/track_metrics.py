"""Metric tracker / regression diff — see how every change moves every number.

The study now emits many JSON result files (one per lens + backtests). This tool
turns them into a single, git-SHA-tagged SNAPSHOT and DIFFS it against the previous
snapshot, so after any change (model swap, threshold, cost, relabel, new lens) you
see — in one table — exactly which metrics moved, by how much, and whether it's a
regression.

Flow:
  1. Recursively flatten every numeric leaf in docs/study/results/*.json into
     dotted keys (e.g. "opportunity_scan.base_test", "market_state.auc_full_state").
  2. Write snapshot -> docs/study/results/snapshots/<utc>_<sha>.json
  3. Diff vs the most recent prior snapshot: added / removed / changed keys, ranked
     by |%change|, with a REGRESSION flag on metrics where "down is bad".

Run in container (after regenerating any lens JSON you changed):
  docker exec saci-brain python -m scripts.track_metrics
  docker exec saci-brain python -m scripts.track_metrics --against <snapshot.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

RESULTS = "/app/docs/study/results"
SNAPDIR = f"{RESULTS}/snapshots"

# substrings of metric keys where a DECREASE is a regression (higher = better)
HIGHER_BETTER = ("net_per_trade", "net_te", "net_tr", "sharpe", "auc", "win",
                 "total", "cagr", "edge", "tophalf", "p_win", "lift")
# substrings where an INCREASE is a regression (lower = better)
LOWER_BETTER = ("drawdown", "mdd", "cvar", "var_", "snoop", "cost")


def flatten(obj, prefix="", out=None):
    """Numeric leaves -> {dotted_key: float}. Short lists indexed; long lists skipped
    (only their length recorded) to avoid snapshotting bulk arrays."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        if len(obj) <= 6:
            for i, v in enumerate(obj):
                flatten(v, f"{prefix}[{i}]", out)
        else:
            out[f"{prefix}.len"] = float(len(obj))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def gather():
    metrics = {}
    for f in sorted(glob.glob(f"{RESULTS}/*.json")):
        name = os.path.basename(f)[:-5]
        try:
            with open(f) as fh:
                flatten(json.load(fh), name, metrics)
        except Exception as e:
            print(f"  skip {name}: {e}")
    return metrics


def git_sha():
    # container mounts ./backend as /app, so .git (repo root) isn't visible inside;
    # let the host pass it via GIT_SHA. Fall back to a local git call, then "nogit".
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd="/app", stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def is_regression(key, delta):
    kl = key.lower()
    if any(s in kl for s in LOWER_BETTER):
        return delta > 0
    if any(s in kl for s in HIGHER_BETTER):
        return delta < 0
    return None  # neutral metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", type=str, default=None, help="prior snapshot to diff against (default: latest)")
    ap.add_argument("--no_write", action="store_true", help="diff only, don't save a new snapshot")
    ap.add_argument("--eps", type=float, default=1e-9, help="ignore |delta| below this")
    args = ap.parse_args()

    os.makedirs(SNAPDIR, exist_ok=True)
    cur = gather()
    sha, ts = git_sha(), time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"snapshot @ {sha} {ts}: {len(cur)} metrics from {len(glob.glob(f'{RESULTS}/*.json'))} json files")

    priors = sorted(glob.glob(f"{SNAPDIR}/*.json"))
    prior_path = args.against or (priors[-1] if priors else None)

    if not args.no_write:
        path = f"{SNAPDIR}/{ts}_{sha}.json"
        with open(path, "w") as fh:
            json.dump({"sha": sha, "ts": ts, "metrics": cur}, fh, indent=2)
        print(f"saved: {path}")

    if not prior_path:
        print("no prior snapshot — this is the BASELINE. Re-run after a change to see the diff.")
        return

    with open(prior_path) as fh:
        prev = json.load(fh)
    pm = prev["metrics"]
    print("=" * 78)
    print(f"DIFF vs {os.path.basename(prior_path)}  (prev @ {prev.get('sha','?')})")
    print("=" * 78)

    added = sorted(set(cur) - set(pm))
    removed = sorted(set(pm) - set(cur))
    changed = []
    for k in set(cur) & set(pm):
        d = cur[k] - pm[k]
        if abs(d) > args.eps:
            base = abs(pm[k]) if abs(pm[k]) > 1e-9 else 1.0
            changed.append((k, pm[k], cur[k], d, 100 * d / base))
    changed.sort(key=lambda r: -abs(r[4]))

    if not changed and not added and not removed:
        print("IDENTICAL — no metric moved.")
        return

    print(f"{'metric':<46s}{'prev':>11s}{'curr':>11s}{'Δ%':>9s}  flag")
    for k, p, c, d, pct in changed[:40]:
        reg = is_regression(k, d)
        flag = "REGRESSION" if reg else ("improved" if reg is False else "")
        print(f"{k[:46]:<46s}{p:>11.4f}{c:>11.4f}{pct:>+8.1f}%  {flag}")
    if len(changed) > 40:
        print(f"  ... +{len(changed)-40} more changed")
    if added:
        print(f"\nNEW metrics ({len(added)}): " + ", ".join(added[:12]) + (" ..." if len(added) > 12 else ""))
    if removed:
        print(f"REMOVED metrics ({len(removed)}): " + ", ".join(removed[:12]) + (" ..." if len(removed) > 12 else ""))

    regs = [k for (k, p, c, d, pct) in changed if is_regression(k, d)]
    print("=" * 78)
    print(f"summary: {len(changed)} changed, {len(added)} new, {len(removed)} removed, "
          f"{len(regs)} REGRESSIONS")


if __name__ == "__main__":
    main()
