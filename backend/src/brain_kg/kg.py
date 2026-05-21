"""Cross-asset knowledge graph.

Triples are typed (entity, relation, value, weight, ts). The two relations
introduced here are:

  correlates_pos   weight in (corr_min_abs, 1]   correlation > 0
  correlates_neg   weight in (corr_min_abs, 1]   |correlation|, was negative

Triples are computed from log-returns over `corr_windows_bars` (e.g.
60/288/864 bars at M5 ≈ 5h/24h/3d). For each (asset_a, asset_b, window)
pair we emit at most one triple per direction. Edges below `corr_min_abs`
are dropped to keep the graph sparse.

Storage: a single parquet at `<kg_dir>/triples.parquet` with columns:
  src, relation, dst, weight, window, computed_at_ms

Append-only semantically. On rebuild we DROP the previous rows for the
window we recomputed and write the new ones — keeps the file canonical.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import BrainConfig, default_config
from .wal import replay_asset


# ─────────────────────────────── Triples ──────────────────────────────

KG_COLUMNS = ("src", "relation", "dst", "weight", "window", "computed_at_ms")

REL_POS = "correlates_pos"
REL_NEG = "correlates_neg"


def _log_returns(closes: np.ndarray) -> np.ndarray:
    """Log returns of a 1D float close series. Length len-1."""
    if closes.size < 2:
        return np.empty(0, dtype=np.float64)
    c = closes.astype(np.float64)
    return np.diff(np.log(c.clip(min=1e-12)))


def _pairwise_corr(ra: np.ndarray, tsa: np.ndarray,
                   rb: np.ndarray, tsb: np.ndarray,
                   window: int) -> tuple[float, int]:
    """Inner-join returns of two assets on ts, take the last `window`
    aligned bars, return (pearson, n_used). NaN-safe (returns 0 on tiny).

    `tsa`/`tsb` are the ts of the LOG-RETURNS (i.e. wal_ts[1:]).
    """
    if ra.size == 0 or rb.size == 0:
        return 0.0, 0
    common, ia, ib = np.intersect1d(tsa, tsb, assume_unique=True, return_indices=True)
    if common.size < 8:
        return 0.0, int(common.size)
    a = ra[ia]
    b = rb[ib]
    if common.size > window:
        a = a[-window:]
        b = b[-window:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0, int(len(a))
    c = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(c):
        return 0.0, int(len(a))
    return c, int(len(a))


# ─────────────────────────────── Build ────────────────────────────────

def compute_triples(cfg: BrainConfig | None = None,
                    assets: Sequence[str] | None = None) -> pd.DataFrame:
    """Recompute the cross-asset correlation graph from the WAL.

    For each window in `cfg.corr_windows_bars`, take the LAST `window`
    aligned returns across assets and compute Pearson. Emit one triple per
    unordered pair with |corr| ≥ cfg.corr_min_abs.
    """
    cfg = cfg or default_config()
    # discover assets
    if assets is None:
        registry = cfg.wal_dir / "assets.txt"
        if not registry.exists():
            return pd.DataFrame(columns=KG_COLUMNS)
        assets = [s for s in registry.read_text().splitlines() if s.strip()]

    # load returns once per asset (longest needed = max window + headroom)
    max_window = max(cfg.corr_windows_bars)
    needed = max_window + 16
    returns: dict[str, np.ndarray] = {}
    ts_by_asset: dict[str, np.ndarray] = {}
    for a in assets:
        arr = replay_asset(cfg, a)
        if arr.size < 2:
            continue
        tail = arr[-(needed + 1):]
        returns[a] = _log_returns(tail["c"])
        ts_by_asset[a] = tail["ts_ms"]
    if not returns:
        return pd.DataFrame(columns=KG_COLUMNS)

    # ts of log returns (one shorter than the bar ts) per asset
    ts_ret: dict[str, np.ndarray] = {a: ts_by_asset[a][1:] for a in returns}
    asset_list = sorted(returns.keys())

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    rows = []
    for win in cfg.corr_windows_bars:
        for i in range(len(asset_list)):
            a = asset_list[i]
            for j in range(i + 1, len(asset_list)):
                b = asset_list[j]
                c, n_used = _pairwise_corr(returns[a], ts_ret[a],
                                           returns[b], ts_ret[b], win)
                if n_used < max(8, win // 4):
                    continue
                if abs(c) < cfg.corr_min_abs:
                    continue
                rel = REL_POS if c > 0 else REL_NEG
                rows.append({
                    "src": a,
                    "relation": rel,
                    "dst": b,
                    "weight": abs(c),
                    "window": int(win),
                    "computed_at_ms": now_ms,
                })

    df = pd.DataFrame(rows, columns=KG_COLUMNS)
    return df


def write_triples(cfg: BrainConfig | None, df: pd.DataFrame,
                  replace_windows: bool = True) -> Path:
    """Persist triples parquet. By default DROP rows for windows being
    rewritten (keeps the file canonical for those windows)."""
    cfg = cfg or default_config()
    cfg.kg_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.kg_dir / "triples.parquet"
    if out.exists() and replace_windows and not df.empty:
        old = pd.read_parquet(out)
        keep = ~old["window"].isin(df["window"].unique())
        df = pd.concat([old[keep], df], ignore_index=True)
    df.to_parquet(out, index=False)
    return out


def load_triples(cfg: BrainConfig | None = None) -> pd.DataFrame:
    cfg = cfg or default_config()
    p = cfg.kg_dir / "triples.parquet"
    if not p.exists():
        return pd.DataFrame(columns=KG_COLUMNS)
    return pd.read_parquet(p)


# ─────────────────────────────── Query ────────────────────────────────

@dataclass
class Edge:
    src: str
    dst: str
    relation: str
    weight: float
    window: int


def neighbors(triples: pd.DataFrame, asset: str,
              window: int | None = None,
              min_weight: float = 0.0) -> list[Edge]:
    """All triples touching `asset`, optionally restricted to a window.

    The KG is undirected in the correlation case, so we union rows where
    src or dst equals the asset and normalize so that `src` is the queried
    asset in the returned Edge list.
    """
    if triples.empty:
        return []
    df = triples
    if window is not None:
        df = df[df["window"] == window]
    df = df[df["weight"] >= min_weight]
    out: list[Edge] = []
    for _, r in df[df["src"] == asset].iterrows():
        out.append(Edge(r["src"], r["dst"], r["relation"], float(r["weight"]), int(r["window"])))
    for _, r in df[df["dst"] == asset].iterrows():
        out.append(Edge(asset, r["src"], r["relation"], float(r["weight"]), int(r["window"])))
    out.sort(key=lambda e: -e.weight)
    return out


def signed_neighbors(triples: pd.DataFrame, asset: str,
                     window: int | None = None,
                     min_weight: float = 0.0) -> list[tuple[str, float, int]]:
    """Same as neighbors but weight is signed (+ for correlates_pos, - for
    correlates_neg). Convenient for the forest propagation step."""
    edges = neighbors(triples, asset, window=window, min_weight=min_weight)
    out = []
    for e in edges:
        w = e.weight if e.relation == REL_POS else -e.weight
        out.append((e.dst, w, e.window))
    return out
