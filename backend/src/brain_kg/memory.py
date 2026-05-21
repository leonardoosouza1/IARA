"""HDC multi-asset memory bank.

For each historical bar that we want to remember, we store the composite
HV of the (window-of-returns × multi-asset) state AT that bar, together
with the FUTURE returns of every tracked asset over 1h / 4h / 1d.

At inference time the brain encodes the *current* window the same way,
hammings against the memory bank, and the top-k matches' future-return
distributions become the analogy hint passed to the SwarmCoordinator.

Storage layout:
  <mem_dir>/bank.parquet
    columns:
      ts_ms       : uint64
      hv          : binary (bits/8 bytes)
      future_1h   : list<float32>   one per tracked asset
      future_4h   : list<float32>
      future_1d   : list<float32>
  <mem_dir>/assets.json — order of assets (positions in the lists)

Bank is rebuilt by `build_bank()` over the WAL; cheap (linear scan) and
deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import BrainConfig, default_config
from .encoder import encode_multi_asset, hamming_sim
from .wal import replay_asset

# Horizons in M5 bars. M5 = 5 min. 12 = 1h, 48 = 4h, 288 = 1d.
HORIZONS = (("future_1h", 12), ("future_4h", 48), ("future_1d", 288))


def _aligned_returns(cfg: BrainConfig, assets: Sequence[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (common_ts[T+1], returns[T, n_assets], assets_kept).

    Restricted to the dense cohort with ≥85% rows of the biggest WAL so
    the alignment is non-empty. Same heuristic as forest.fit_score.
    """
    series = {a: replay_asset(cfg, a) for a in assets}
    series = {a: s for a, s in series.items() if s.size > 0}
    if not series:
        return np.empty(0, dtype=np.int64), np.empty((0, 0)), []
    biggest = max(s.size for s in series.values())
    series = {a: s for a, s in series.items() if s.size >= 0.85 * biggest}
    if len(series) < 2:
        return np.empty(0, dtype=np.int64), np.empty((0, 0)), []
    common = None
    for a, s in series.items():
        ts_set = set(int(x) for x in s["ts_ms"].tolist())
        common = ts_set if common is None else common & ts_set
    common_sorted = np.array(sorted(common), dtype=np.int64)
    if common_sorted.size < 4:
        return np.empty(0, dtype=np.int64), np.empty((0, 0)), []
    kept = sorted(series.keys())
    mat = np.empty((common_sorted.size, len(kept)), dtype=np.float64)
    for j, a in enumerate(kept):
        idx = np.searchsorted(series[a]["ts_ms"], common_sorted)
        mat[:, j] = series[a]["c"][idx].astype(np.float64)
    # log returns
    R = np.diff(np.log(mat.clip(min=1e-12)), axis=0)
    return common_sorted, R, kept


@dataclass
class MemoryBank:
    assets: list[str]
    ts_ms: np.ndarray                    # [N] bar timestamp at which the HV is anchored
    hvs: np.ndarray                      # [N, bytes_per_hv] uint8
    future: dict[str, np.ndarray]        # name → [N, n_assets] float32

    def save(self, mem_dir: Path) -> Path:
        mem_dir.mkdir(parents=True, exist_ok=True)
        # store as parquet — list-of-bytes for hv, list<float32> for futures
        df = pd.DataFrame({
            "ts_ms": self.ts_ms.astype(np.int64),
            "hv":    [bytes(self.hvs[i].tobytes()) for i in range(self.hvs.shape[0])],
        })
        for k, v in self.future.items():
            df[k] = [v[i].astype(np.float32).tolist() for i in range(v.shape[0])]
        path = mem_dir / "bank.parquet"
        df.to_parquet(path, index=False)
        (mem_dir / "assets.json").write_text(json.dumps(self.assets))
        return path

    @classmethod
    def load(cls, mem_dir: Path, bits: int) -> "MemoryBank":
        df = pd.read_parquet(mem_dir / "bank.parquet")
        assets = json.loads((mem_dir / "assets.json").read_text())
        n = len(df)
        bpb = bits // 8
        hvs = np.empty((n, bpb), dtype=np.uint8)
        for i, b in enumerate(df["hv"].tolist()):
            hvs[i] = np.frombuffer(b, dtype=np.uint8)
        future = {}
        for k, _ in HORIZONS:
            if k in df.columns:
                future[k] = np.array(df[k].tolist(), dtype=np.float32)
        return cls(assets=list(assets), ts_ms=df["ts_ms"].to_numpy(dtype=np.int64),
                   hvs=hvs, future=future)


def build_bank(cfg: BrainConfig | None = None,
               step: int = 4,
               max_rows: int = 8000) -> MemoryBank:
    """Build the memory bank from the aligned dense cohort.

    `step` skips bars to keep the bank compact (one anchor every step bars).
    """
    cfg = cfg or default_config()
    assets_all = [s for s in (cfg.wal_dir / "assets.txt").read_text().splitlines() if s.strip()]
    ts, R, kept = _aligned_returns(cfg, assets_all)
    if R.size == 0:
        return MemoryBank(kept, np.empty(0, dtype=np.int64),
                          np.empty((0, cfg.hdc_bits // 8), dtype=np.uint8), {})

    win = cfg.hdc_window_bars
    bits = cfg.hdc_bits
    max_h = max(h for _, h in HORIZONS)
    anchors = list(range(win, R.shape[0] - max_h, step))[:max_rows]
    n = len(anchors)
    hvs = np.empty((n, bits // 8), dtype=np.uint8)
    future = {name: np.empty((n, len(kept)), dtype=np.float32)
              for name, _ in HORIZONS}

    for i, a in enumerate(anchors):
        per = {aid: R[a - win:a, aid] for aid in range(len(kept))}
        hvs[i] = encode_multi_asset(per, bits=bits)
        for name, h in HORIZONS:
            future[name][i] = R[a:a + h].sum(axis=0).astype(np.float32)

    anchor_ts = ts[np.asarray(anchors, dtype=np.int64) + 1]    # ts of the bar AFTER the window
    return MemoryBank(kept, anchor_ts, hvs, future)


def retrieve(bank: MemoryBank, query_hv: np.ndarray,
             top_k: int = 12, min_sim: float = 0.55) -> list[dict]:
    """Top-k hamming-similar entries with their future returns."""
    if bank.hvs.shape[0] == 0:
        return []
    diff = np.bitwise_xor(bank.hvs, query_hv[None, :])
    # population count per row via unpackbits
    unp = np.unpackbits(diff, axis=1)
    n_bits = unp.shape[1]
    n_diff = unp.sum(axis=1)
    sim = 1.0 - n_diff / n_bits
    order = np.argsort(-sim)[:top_k]
    out = []
    for idx in order:
        if sim[idx] < min_sim:
            break
        item = {"ts_ms": int(bank.ts_ms[idx]), "sim": float(sim[idx])}
        for name, _ in HORIZONS:
            if name in bank.future:
                item[name] = bank.future[name][idx].tolist()
        out.append(item)
    return out


def horizon_distribution(matches: list[dict], horizon: str = "future_1h"
                         ) -> dict[str, dict[str, float]]:
    """For each asset (by position in MemoryBank.assets), produce a small
    stat (mean, std, p10, p90) of the horizon returns across matches.

    Returns a dict keyed by asset-index-string for now — the caller knows
    the asset order. Empty if matches is empty.
    """
    if not matches or horizon not in matches[0]:
        return {}
    arr = np.array([m[horizon] for m in matches], dtype=np.float32)
    n_assets = arr.shape[1]
    out = {}
    for j in range(n_assets):
        v = arr[:, j]
        out[str(j)] = {
            "mean": float(v.mean()),
            "std": float(v.std()),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }
    return out


# ─────────────────────── Gate metric — analogy coverage ────────────────

def analogy_coverage(cfg: BrainConfig | None = None,
                     bank: MemoryBank | None = None,
                     n_probes: int = 200,
                     horizon: str = "future_1h",
                     cover_band: float = 0.75) -> dict:
    """Held-out probe: pick `n_probes` bars NOT in the bank's anchor set,
    encode the bar's window, retrieve top-k. For each probe, the realised
    horizon return per asset must lie inside the [p_lo, p_hi] band of the
    matches' distribution (cover_band centered) to count as covered.

    Returns mean coverage across assets and probes. Target ≥ 0.60.
    """
    cfg = cfg or default_config()
    if bank is None:
        bank = build_bank(cfg)
    if bank.hvs.shape[0] == 0:
        return {"ok": False, "reason": "empty bank"}
    assets_all = [s for s in (cfg.wal_dir / "assets.txt").read_text().splitlines() if s.strip()]
    ts, R, kept = _aligned_returns(cfg, assets_all)
    if kept != bank.assets:
        return {"ok": False, "reason": "asset cohort drifted"}
    win = cfg.hdc_window_bars
    bits = cfg.hdc_bits
    max_h = max(h for _, h in HORIZONS)
    horizon_bars = dict(HORIZONS)[horizon]
    # probe candidates = bars not used as anchors (rough — every step+1 bar)
    rng = np.random.default_rng(123)
    candidates = list(range(win + 1, R.shape[0] - max_h - 1, 1))
    bank_anchor_ts = set(bank.ts_ms.tolist())
    candidates = [a for a in candidates if int(ts[a + 1]) not in bank_anchor_ts]
    rng.shuffle(candidates)
    probes = candidates[:n_probes]
    if not probes:
        return {"ok": False, "reason": "no out-of-bank probes"}

    lo_q = (1 - cover_band) / 2
    hi_q = 1 - lo_q
    cover_per_asset = np.zeros(len(kept))
    counts_per_asset = np.zeros(len(kept))

    for a in probes:
        per = {aid: R[a - win:a, aid] for aid in range(len(kept))}
        q = encode_multi_asset(per, bits=bits)
        matches = retrieve(bank, q, top_k=cfg.hdc_top_k, min_sim=cfg.hdc_min_hamming_sim)
        if not matches or horizon not in matches[0]:
            continue
        arr = np.array([m[horizon] for m in matches], dtype=np.float32)
        actual = R[a:a + horizon_bars].sum(axis=0)
        for j in range(len(kept)):
            lo = float(np.quantile(arr[:, j], lo_q))
            hi = float(np.quantile(arr[:, j], hi_q))
            counts_per_asset[j] += 1
            if lo <= actual[j] <= hi:
                cover_per_asset[j] += 1
    cov = cover_per_asset / np.maximum(counts_per_asset, 1)
    return {
        "ok": True,
        "n_probes_used": int(counts_per_asset[0]),
        "horizon": horizon,
        "cover_band": cover_band,
        "per_asset": {kept[i]: round(float(cov[i]), 3) for i in range(len(kept))},
        "mean_coverage": float(cov.mean()),
    }
