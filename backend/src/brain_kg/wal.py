"""Append-only WAL of OHLCV ticks. Schema is a fixed-size binary record so
scans/replays are bit-exact and zero-copy with `numpy.fromfile`.

Record layout (little-endian, 56 bytes):
  ts_ms       : uint64  (Unix ms)
  asset_id    : uint16  (assigned by AssetRegistry)
  pad         : uint16  (zero)
  o           : float32
  h           : float32
  l           : float32
  c           : float32
  v           : float32
  vwap        : float32  (== close when volume==0)
  spread_bps  : float32  (==(h-l)/c * 1e4)
  pad2        : uint32  (zero, future flags)

File naming: `<wal_dir>/<asset>_<YYYY>.bin` — one shard per asset per year.
Append-only, append in increasing ts_ms order (enforced by the writer).

Read helpers stream a sorted multi-asset view by merging shards on ts_ms.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .config import BrainConfig, default_config

# 56-byte record. `<` = little-endian, no padding by Python (Q=u64, H=u16,
# f=f32, I=u32). 8+2+2+4*7+4 = 8+2+2+28+4 = 44? Wait — recount.
#   Q(8) + H(2) + H(2) + 7f(28) + I(4) = 44. So record is 44 bytes.
# Keep struct authoritative.
RECORD_FMT = "<QHH7fI"
RECORD_SIZE = struct.calcsize(RECORD_FMT)
assert RECORD_SIZE == 44, RECORD_SIZE

# Equivalent numpy structured dtype for vector reads.
RECORD_DTYPE = np.dtype([
    ("ts_ms", "<u8"),
    ("asset_id", "<u2"),
    ("pad", "<u2"),
    ("o", "<f4"), ("h", "<f4"), ("l", "<f4"),
    ("c", "<f4"), ("v", "<f4"),
    ("vwap", "<f4"), ("spread_bps", "<f4"),
    ("pad2", "<u4"),
])
assert RECORD_DTYPE.itemsize == RECORD_SIZE


# ────────────────────────────── Registry ──────────────────────────────

class AssetRegistry:
    """Persistent string→u16 mapping for asset symbols.

    Stored as a single newline-delimited text file in the WAL dir. The line
    number (0-based) is the asset_id. Append-only — ids never reassigned.
    """

    def __init__(self, path: Path):
        self.path = path
        self._sym_to_id: dict[str, int] = {}
        self._id_to_sym: list[str] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                self._sym_to_id[line] = len(self._id_to_sym)
                self._id_to_sym.append(line)

    def id_of(self, symbol: str) -> int:
        if symbol not in self._sym_to_id:
            new_id = len(self._id_to_sym)
            self._sym_to_id[symbol] = new_id
            self._id_to_sym.append(symbol)
            # atomic append
            with open(self.path, "a") as f:
                f.write(symbol + "\n")
        return self._sym_to_id[symbol]

    def symbol_of(self, asset_id: int) -> str:
        return self._id_to_sym[asset_id]

    def known(self) -> Sequence[str]:
        return tuple(self._id_to_sym)


# ────────────────────────────── Writer ────────────────────────────────

@dataclass
class WalShardPath:
    asset: str
    year: int
    path: Path


def shard_path(cfg: BrainConfig, asset: str, ts_ms: int) -> WalShardPath:
    year = int(np.datetime64(ts_ms, "ms").astype("datetime64[Y]").astype(int)) + 1970
    return WalShardPath(asset, year, cfg.wal_dir / f"{asset}_{year}.bin")


class WalWriter:
    """Single-asset append-only writer. Enforces monotone ts_ms within a
    shard. Open file kept on writer for batched appends.
    """

    def __init__(self, cfg: BrainConfig, asset: str):
        self.cfg = cfg
        self.asset = asset
        self.registry = AssetRegistry(cfg.wal_dir / "assets.txt")
        self.asset_id = self.registry.id_of(asset)
        self._open: dict[int, object] = {}        # year → file handle
        self._last_ts: dict[int, int] = {}        # year → last ts written

    def _file_for(self, ts_ms: int):
        year = int(np.datetime64(ts_ms, "ms").astype("datetime64[Y]").astype(int)) + 1970
        if year not in self._open:
            p = self.cfg.wal_dir / f"{self.asset}_{year}.bin"
            # if shard exists, recover last ts
            last_ts = 0
            if p.exists() and p.stat().st_size >= RECORD_SIZE:
                with open(p, "rb") as f:
                    f.seek(-RECORD_SIZE, 2)
                    rec = struct.unpack(RECORD_FMT, f.read(RECORD_SIZE))
                    last_ts = rec[0]
            self._open[year] = open(p, "ab", buffering=1024 * 64)
            self._last_ts[year] = last_ts
        return year

    def append(self, ts_ms: int, o: float, h: float, l: float,
               c: float, v: float, vwap: float | None = None,
               spread_bps: float | None = None) -> None:
        year = self._file_for(ts_ms)
        last = self._last_ts[year]
        if ts_ms <= last:
            return  # idempotent: skip duplicates/out-of-order
        if vwap is None:
            vwap = c
        if spread_bps is None:
            spread_bps = float((h - l) / c * 1e4) if c > 0 else 0.0
        buf = struct.pack(RECORD_FMT, int(ts_ms), self.asset_id, 0,
                          float(o), float(h), float(l), float(c), float(v),
                          float(vwap), float(spread_bps), 0)
        self._open[year].write(buf)
        self._last_ts[year] = ts_ms

    def close(self) -> None:
        for f in self._open.values():
            try:
                f.close()
            except Exception:
                pass
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ────────────────────────────── Reader ────────────────────────────────

def read_shard(path: Path) -> np.ndarray:
    """Return the full shard as a structured ndarray."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty(0, dtype=RECORD_DTYPE)
    return np.fromfile(path, dtype=RECORD_DTYPE)


def shards_for_asset(cfg: BrainConfig, asset: str) -> list[Path]:
    return sorted(cfg.wal_dir.glob(f"{asset}_*.bin"))


def replay_asset(cfg: BrainConfig, asset: str) -> np.ndarray:
    """Concatenate all shards for an asset in chronological order."""
    parts = [read_shard(p) for p in shards_for_asset(cfg, asset)]
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return np.empty(0, dtype=RECORD_DTYPE)
    return np.concatenate(parts)


def replay_multi(cfg: BrainConfig, assets: Sequence[str]) -> np.ndarray:
    """Merged chronological replay across multiple assets."""
    arrs = [replay_asset(cfg, a) for a in assets]
    arrs = [a for a in arrs if a.size > 0]
    if not arrs:
        return np.empty(0, dtype=RECORD_DTYPE)
    merged = np.concatenate(arrs)
    # stable sort by ts_ms (mergesort is stable)
    return np.sort(merged, order="ts_ms", kind="mergesort")


def iter_window(arr: np.ndarray, window: int, step: int = 1) -> Iterator[np.ndarray]:
    """Yield successive windows of length `window` over arr (per-asset)."""
    n = arr.shape[0]
    if n < window:
        return
    for i in range(0, n - window + 1, step):
        yield arr[i:i + window]


# ────────────────────────── Parquet → WAL migration ───────────────────

def migrate_parquet(cfg: BrainConfig | None = None,
                    assets: Iterable[str] | None = None) -> dict[str, int]:
    """Scan `cfg.raw_dir` for `<ASSET>.parquet`, ingest each in time order."""
    cfg = cfg or default_config()
    raw_dir = cfg.raw_dir
    pqs = sorted(raw_dir.glob("*.parquet"))
    if assets is not None:
        wanted = {a.upper() for a in assets}
        pqs = [p for p in pqs if p.stem.upper() in wanted]
    counts: dict[str, int] = {}
    for pq in pqs:
        df = pd.read_parquet(pq)
        if "date" not in df.columns:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        asset = pq.stem.upper()
        # Force ns precision then convert to ms — pandas 3.x astype('int64')
        # of tz-aware datetime can return the unit's own count (ms here),
        # so dividing by 1e6 truncates badly when the column is already ms.
        dt = pd.to_datetime(df["date"], utc=True)
        ts_ms = dt.values.astype("datetime64[ms]").astype("int64")
        # drop duplicate timestamps — many parquets have many bars per ts.
        keep = np.empty(len(ts_ms), dtype=bool); keep[0] = True
        keep[1:] = ts_ms[1:] > ts_ms[:-1]
        o = df["open"].to_numpy(dtype="float32")
        h = df["high"].to_numpy(dtype="float32")
        l = df["low"].to_numpy(dtype="float32")
        c = df["close"].to_numpy(dtype="float32")
        v = df["volume"].to_numpy(dtype="float32") if "volume" in df.columns else np.zeros(len(df), dtype="float32")
        n_written = 0
        with WalWriter(cfg, asset) as w:
            for i in range(len(df)):
                if not keep[i] or ts_ms[i] <= 0:
                    continue
                w.append(int(ts_ms[i]), float(o[i]), float(h[i]), float(l[i]),
                         float(c[i]), float(v[i]))
                n_written += 1
        counts[asset] = n_written
    return counts


# ────────────────────────── Bit-exact replay gate ─────────────────────

def verify_parquet_replay(cfg: BrainConfig | None, asset: str, tol: float = 1e-6) -> dict:
    """Read the parquet, read the WAL, compare close-by-close.

    Returns dict with row counts and max abs diff on each column.
    """
    cfg = cfg or default_config()
    pq = cfg.raw_dir / f"{asset}.parquet"
    df = pd.read_parquet(pq).sort_values("date").reset_index(drop=True)
    # dedup parquet on timestamp the same way the writer does, so counts match
    dt = pd.to_datetime(df["date"], utc=True)
    ts_ms = dt.values.astype("datetime64[ms]").astype("int64")
    keep = np.empty(len(ts_ms), dtype=bool); keep[0] = True
    keep[1:] = ts_ms[1:] > ts_ms[:-1]
    df = df.loc[keep].reset_index(drop=True)
    wal = replay_asset(cfg, asset)
    n = min(len(df), len(wal))
    diffs = {}
    for col, key in (("open", "o"), ("high", "h"), ("low", "l"),
                     ("close", "c"), ("volume", "v")):
        a = df[col].to_numpy(dtype="float32")[:n]
        b = wal[key][:n]
        diffs[col] = float(np.max(np.abs(a - b))) if n else 0.0
    return {"asset": asset, "rows_parquet": len(df), "rows_wal": len(wal),
            "max_abs": diffs, "ok": all(v <= tol for v in diffs.values()) and len(df) == len(wal)}
