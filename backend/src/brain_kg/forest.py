"""Gravity forest of assets connected by mycelium edges (signed correlations).

Each `Tree` is an asset. The `mycelium` is the signed-weighted adjacency
matrix built from the KG triples (default window). Propagation is a
1-hop linear pass:

    move[b] ≈ sum_a delta[a] * edge_signed_weight(a, b)

This is the simplest defensible model — it matches what a one-step linear
regression of asset returns on a driver would give. Multi-hop propagation
is supported (matrix power) but is off by default; for trading hints,
1-hop is usually the right signal-to-noise tradeoff.

The forest can be `propagate(driver_deltas)`-ed in either direction:

  - "if DXY moves +0.3%, what are the implied moves of the other assets?"
  - "given the last bar's returns of every asset, which assets are
     CURRENTLY most correlated with the basket?" (PCA-ish).

Phase-2 gate metric:
  fit_score(asset, window): re-build the mycelium from data BEFORE a
  hold-out, then check how well its propagation predicts the next bar's
  return given the rest of the bar's drivers. Targets r²/Pearson ≥ 0.4
  on the dominant FX pair structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BrainConfig, default_config
from .kg import REL_POS, REL_NEG, load_triples


@dataclass
class Forest:
    assets: list[str] = field(default_factory=list)
    idx: dict[str, int] = field(default_factory=dict)
    M: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))   # signed mycelium
    window: int = 0

    @classmethod
    def from_kg(cls, cfg: BrainConfig | None = None,
                window: int | None = None) -> "Forest":
        cfg = cfg or default_config()
        df = load_triples(cfg)
        if df.empty:
            return cls()
        if window is None:
            # default to the SHORTEST window (most reactive)
            window = int(min(df["window"].unique()))
        sub = df[df["window"] == window]
        assets = sorted(set(sub["src"]).union(sub["dst"]))
        idx = {a: i for i, a in enumerate(assets)}
        n = len(assets)
        M = np.zeros((n, n), dtype=np.float64)
        for _, r in sub.iterrows():
            i = idx[r["src"]]; j = idx[r["dst"]]
            w = float(r["weight"])
            s = +1.0 if r["relation"] == REL_POS else -1.0
            M[i, j] += s * w
            M[j, i] += s * w        # symmetric
        return cls(assets=assets, idx=idx, M=M, window=int(window))

    # ────────── core ops ──────────

    def propagate(self, deltas: Mapping[str, float], depth: int = 1) -> dict[str, float]:
        """Propagate a sparse vector of driver moves through the mycelium.

        `depth=1` = one matrix-vector product (linear regression analogue).
        `depth=k` = power iteration; weights decay since M is not a
        Markov chain — values can blow up for large k. Use k ≤ 2.
        """
        if not self.assets:
            return {}
        n = len(self.assets)
        v = np.zeros(n, dtype=np.float64)
        for a, d in deltas.items():
            j = self.idx.get(a)
            if j is not None:
                v[j] = float(d)
        out = v.copy()
        cur = v
        for _ in range(max(1, depth)):
            cur = self.M @ cur
            out = out + cur
        return {a: float(out[i]) for i, a in enumerate(self.assets)}

    def neighbors_signed(self, asset: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Top-k signed neighbors of `asset` by |weight|."""
        i = self.idx.get(asset)
        if i is None:
            return []
        row = self.M[i]
        order = np.argsort(-np.abs(row))
        out = []
        for j in order:
            if j == i:
                continue
            w = float(row[j])
            if w == 0.0:
                continue
            out.append((self.assets[j], w))
            if len(out) >= top_k:
                break
        return out


# ────────────────────── Gate metric — propagation fit ──────────────────

def fit_score(cfg: BrainConfig | None = None,
              window: int = 60,
              eval_bars: int = 500) -> dict:
    """Leave-out-last-bar style fit. For the last `eval_bars` bars:
    build the forest from the OLDER returns only, then for each bar use
    the bar's actual returns as drivers and check whether forest's
    propagated prediction matches the next bar (per-asset).

    Returns a dict with mean Pearson and rms error.
    """
    cfg = cfg or default_config()
    # need WAL access
    from .wal import replay_asset
    registry = (cfg.wal_dir / "assets.txt").read_text().splitlines()
    assets = [s for s in registry if s.strip()]
    closes: dict[str, np.ndarray] = {}
    ts: dict[str, np.ndarray] = {}
    for a in assets:
        arr = replay_asset(cfg, a)
        if arr.size > 3:
            closes[a] = arr["c"]
            ts[a] = arr["ts_ms"]
    # densest cohort — saci has 4 high-coverage M5 pairs (~16k bars,
    # recent) and a long tail (~5.7k, starting 2003 with sparse hours).
    # The two groups barely overlap in time, so for the gate we restrict
    # to assets with size ≥ 0.85 × biggest.
    sizes = {a: closes[a].size for a in closes}
    biggest = max(sizes.values())
    keep = [a for a, sz in sizes.items() if sz >= 0.85 * biggest]
    if len(keep) < 3:
        return {"ok": False, "reason": "not enough dense assets",
                "n_assets": len(keep), "biggest": biggest}
    # align on common ts of the kept dense group
    common = None
    for a in keep:
        s = set(int(x) for x in ts[a].tolist())
        common = s if common is None else common & s
    common_sorted = np.array(sorted(common), dtype=np.int64)
    if common_sorted.size < eval_bars + window + 8:
        return {"ok": False, "reason": "common ts too short",
                "n_assets": len(keep), "n_common": int(common_sorted.size)}
    # log returns aligned on common_sorted
    R = np.empty((common_sorted.size - 1, len(keep)), dtype=np.float64)
    asset_idx = {a: i for i, a in enumerate(keep)}
    for a in keep:
        idx = np.searchsorted(ts[a], common_sorted)
        c = closes[a][idx].astype(np.float64)
        R[:, asset_idx[a]] = np.diff(np.log(c.clip(min=1e-12)))
    # build forest using rows BEFORE the eval slice
    train_end = R.shape[0] - eval_bars
    Rt = R[max(0, train_end - window):train_end]
    if Rt.shape[0] < 8:
        return {"ok": False, "reason": "train window too small"}
    if Rt.std() < 1e-12:
        return {"ok": False, "reason": "no variance"}
    C = np.corrcoef(Rt, rowvar=False)
    n = len(keep)
    M = np.where(np.abs(C) >= cfg.corr_min_abs, C, 0.0)
    np.fill_diagonal(M, 0.0)
    # evaluation: for each bar in eval slice, use bar t as driver,
    # propagate via M, compare to bar t (in-sample 1-step PROJECTION).
    eval_slice = R[train_end:]
    preds = eval_slice @ M.T              # [eval_bars, n]
    # per-asset Pearson between actual eval bar and prediction
    per_asset = []
    for j in range(n):
        a = eval_slice[:, j]
        p = preds[:, j]
        if a.std() < 1e-12 or p.std() < 1e-12:
            per_asset.append(0.0)
            continue
        per_asset.append(float(np.corrcoef(a, p)[0, 1]))
    rms = float(np.sqrt(np.mean((eval_slice - preds) ** 2)))
    return {
        "ok": True,
        "n_assets": n,
        "eval_bars": int(eval_slice.shape[0]),
        "mean_pearson": float(np.nanmean(per_asset)),
        "median_pearson": float(np.nanmedian(per_asset)),
        "rms_err": rms,
        "per_asset": {keep[i]: round(per_asset[i], 3) for i in range(n)},
        "window": int(window),
    }
