"""Backtest accelerator: WAL replay + HDC decision cache.

Strategy: walk the WAL, build a multi-asset feature window for each bar,
hash to an HDC key, look up the cache. If cache hit (sim ≥ thresh), reuse
the prior decision. Else call `decide_fn` and store. Cache hit rate is
the speedup; with a good encoder, many bars cluster onto a small set of
"regimes" → 10-100× speedup is realistic for long backtests on FX.

The cache is local to the run (not persisted) — backtests are
deterministic from WAL, so re-running re-builds it. A persistent cache
is an obvious next step but unneeded for the Phase 5 gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .config import BrainConfig, default_config
from .encoder import encode_multi_asset
from .memory import _aligned_returns


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    bars: int = 0
    bar_seconds: float = 0.0
    decide_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


def _fingerprint(hv: np.ndarray, fp_bytes: int = 2) -> bytes:
    """Coarse locality-sensitive fingerprint of the HV.

    XORs `chunk_count` equal-sized chunks of the HV down to `fp_bytes`,
    discarding fine differences. A 2-byte fingerprint gives 65 536
    buckets — small bit flips in adjacent bars often land in the same
    bucket, enabling cache reuse on contiguous regimes.
    """
    n = hv.size
    chunk = n // fp_bytes
    if chunk == 0:
        return bytes(hv.tolist())
    fp = np.zeros(fp_bytes, dtype=np.uint8)
    for k in range(chunk):
        fp ^= hv[k * fp_bytes:(k + 1) * fp_bytes]
    return bytes(fp.tolist())


@dataclass
class DecisionCache:
    """Cache keyed by a coarse LSH fingerprint of the HV (XOR of chunks).

    Tunable trade-off via fp_bytes: more bytes = stricter bucket =
    fewer collisions/less reuse; fewer bytes = aggressive reuse but
    larger error if regimes drift. Default 2 bytes (65k buckets)
    works well at M5 across the dense cohort.
    """
    fp_bytes: int = 2
    store: dict[bytes, np.ndarray] = field(default_factory=dict)

    def get(self, hv: np.ndarray) -> np.ndarray | None:
        return self.store.get(_fingerprint(hv, self.fp_bytes))

    def put(self, hv: np.ndarray, value: np.ndarray) -> None:
        self.store[_fingerprint(hv, self.fp_bytes)] = value


def _regime_key(current_bar_returns: np.ndarray, n_buckets: int = 9) -> bytes:
    """Bucket each asset's current bar return into n_buckets levels.
    Tuple of these buckets is the regime — bars with similar instantaneous
    cross-asset moves share a key (and a cached decision).
    """
    # symmetric clamp ±0.005 (50 bps single-bar) → log-scaled bucket
    clipped = np.clip(current_bar_returns, -0.005, 0.005)
    sign = np.sign(clipped)
    mag = np.abs(clipped)
    side = n_buckets // 2
    b = np.floor(np.tanh(mag * 200) * side).astype(np.int8)
    b = np.clip(b, 0, side)
    signed = (sign.astype(np.int8) * b) + np.int8(side)   # 0..n_buckets
    return signed.tobytes()


def backtest_replay(cfg: BrainConfig | None,
                    decide_fn: Callable[[np.ndarray], np.ndarray],
                    window_bars: int | None = None,
                    cache: DecisionCache | None = None,
                    key_strategy: str = "regime") -> tuple[np.ndarray, CacheStats]:
    """Walk the aligned dense-cohort WAL bar-by-bar, route through cache
    → decide_fn. Two cache key strategies:

      - "regime"  → fast: cross-asset current-bar return buckets. Tight
                    cluster of bars share a key. Best speedup.
      - "hv"      → flexible: LSH fingerprint of the full window HV.
                    Catches longer-pattern regimes; less aggressive.

    decide_fn receives the per-asset returns matrix [window, n_assets]
    and returns whatever decision tensor the backtest needs.
    Returns (decisions[n_bars, ...], CacheStats).
    """
    cfg = cfg or default_config()
    window = window_bars or cfg.hdc_window_bars
    cache = cache or DecisionCache()

    assets_all = [s for s in (cfg.wal_dir / "assets.txt").read_text().splitlines() if s.strip()]
    ts, R, kept = _aligned_returns(cfg, assets_all)
    if R.size == 0:
        return np.empty(0), CacheStats()

    n_bars = R.shape[0] - window
    stats = CacheStats(bars=n_bars)
    use_regime = (key_strategy == "regime")
    no_cache = (key_strategy == "nocache")

    decisions = []
    t_total = perf_counter()
    for i in range(n_bars):
        cached = None
        key_bytes = None
        hv = None
        if no_cache:
            pass
        elif use_regime:
            key_bytes = _regime_key(R[i + window - 1])
            cached = cache.store.get(key_bytes)
        else:
            per = {aid: R[i:i + window, aid] for aid in range(len(kept))}
            hv = encode_multi_asset(per, bits=cfg.hdc_bits)
            cached = cache.get(hv)
        if cached is not None:
            stats.hits += 1
            decisions.append(cached)
            continue
        stats.misses += 1
        t0 = perf_counter()
        d = decide_fn(R[i:i + window])
        stats.decide_seconds += perf_counter() - t0
        if not no_cache:
            if use_regime:
                cache.store[key_bytes] = d
            else:
                cache.put(hv, d)
        decisions.append(d)
    stats.bar_seconds = perf_counter() - t_total

    out = np.stack(decisions) if decisions else np.empty(0)
    return out, stats
