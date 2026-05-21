"""HDC encoding of multi-asset windows — vectorised.

Atom strategy:
  For each (position, bucket) pair, materialise a deterministic random
  packed bit-vector ONCE at module import. Encoding a window is then a
  single fancy-indexed XOR-reduce — O(window) Python work, the XOR
  itself runs in numpy C.

Features per bar:
  ret_bucket : log-return discretised into signed buckets (24 each side)
  vol_bucket : |ret| relative to a rolling 64-bar EMA of |ret|
                (0 = quiet, N_VOL-1 = high vol)
This carries enough structure for the analogy retrieval; pure return
bucketing alone was too noisy in practice.

Distance: 1 - hamming/n_bits, where higher = more similar.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# Discrete return buckets per side.
N_RET = 24
# Volatility buckets (0..N_VOL-1)
N_VOL = 8


def _splitmix64_array(seeds: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64. Input: uint64 array; output: uint64 array."""
    s = seeds.astype(np.uint64, copy=True)
    s = s + np.uint64(0x9E3779B97F4A7C15)
    z = s
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z


def _make_atom_packed(seed_hi: int, seed_lo: int, n_bytes: int) -> np.ndarray:
    """Deterministic random uint8 vector of length n_bytes from (hi, lo)."""
    # generate n_bytes / 8 uint64s by hashing increasing counters
    n_u64 = (n_bytes + 7) // 8
    mask = (1 << 64) - 1
    base = ((int(seed_hi) << 32) ^ int(seed_lo) ^ 0xC0FFEE) & mask
    seeds_py = [(base ^ i) & mask for i in range(n_u64)]
    seeds = np.fromiter((np.uint64(s) for s in seeds_py),
                        dtype=np.uint64, count=n_u64)
    vals = _splitmix64_array(seeds)
    return vals.view(np.uint8)[: n_bytes]


# Lazy atom tables — keyed by (bits, channel) to allow channels of different
# semantics to live in the same hypervector space.
_ATOM_CACHE: dict[tuple[int, str], np.ndarray] = {}


def _atom_table(bits: int, channel: str, n_pos: int, n_levels: int) -> np.ndarray:
    """[n_pos, n_levels, bits/8] uint8 atom table for a channel."""
    key = (bits, channel, n_pos, n_levels)  # type: ignore
    cached = _ATOM_CACHE.get(key)
    if cached is not None:
        return cached
    bpb = bits // 8
    chan_hash = sum(ord(c) for c in channel) * 0x9E3779B9
    tbl = np.empty((n_pos, n_levels, bpb), dtype=np.uint8)
    for p in range(n_pos):
        for lvl in range(n_levels):
            tbl[p, lvl] = _make_atom_packed(chan_hash ^ p, lvl, bpb)
    _ATOM_CACHE[key] = tbl
    return tbl


def _return_bucket(rets: np.ndarray) -> np.ndarray:
    """Vector of signed log-returns → vector of bucket indices in
    [0, 2*N_RET]. 0 = most-negative, N_RET = zero, 2*N_RET = most-positive."""
    r = np.clip(rets, -0.02, 0.02)
    sign = np.sign(r)
    mag = np.abs(r)
    mag_b = np.floor(np.tanh(mag * 80.0) * N_RET).astype(np.int64)
    mag_b = np.clip(mag_b, 0, N_RET)
    signed = (sign.astype(np.int64) * mag_b) + N_RET     # → [0, 2N_RET]
    return signed.astype(np.int64)


def _vol_bucket(rets: np.ndarray) -> np.ndarray:
    """Per-bar volatility bucket via EMA of |rets|, 0..N_VOL-1."""
    abs_r = np.abs(rets).astype(np.float64)
    ema = np.empty_like(abs_r)
    alpha = 2.0 / (32 + 1)
    cur = abs_r[0] + 1e-12
    for i in range(abs_r.size):
        cur = alpha * abs_r[i] + (1 - alpha) * cur
        ema[i] = cur
    # ratio against own ema, bucketed
    ratio = abs_r / np.maximum(ema, 1e-12)
    # buckets via tanh squashing: 0=quiet (ratio<0.5), N_VOL-1=hot (>2)
    b = np.floor(np.tanh((ratio - 1.0)) * (N_VOL // 2) + (N_VOL // 2)).astype(np.int64)
    return np.clip(b, 0, N_VOL - 1)


def encode_window(returns: Sequence[float], bits: int = 1024) -> np.ndarray:
    """Encode one asset's window of log-returns → packed HV by MAJORITY
    BUNDLING (standard HDC accumulator).

    XOR-reducing many random atoms gives back a near-random vector and
    destroys signal — XOR is for BINDING (role × content), not for
    aggregating multiple atoms per channel. We unpack each atom to ±1
    bits, sum, threshold → repack.
    """
    rets = np.asarray(returns, dtype=np.float64)
    n_pos = rets.size
    if n_pos == 0:
        return np.zeros(bits // 8, dtype=np.uint8)
    rb = _return_bucket(rets)
    vb = _vol_bucket(rets)
    tbl_r = _atom_table(bits, "ret", n_pos, 2 * N_RET + 1)
    tbl_v = _atom_table(bits, "vol", n_pos, N_VOL)
    pos = np.arange(n_pos)
    a = tbl_r[pos, rb]          # [n_pos, bpb]
    b = tbl_v[pos, vb]
    stack = np.concatenate([a, b], axis=0)              # [2*n_pos, bpb]
    bits_unp = np.unpackbits(stack, axis=1).astype(np.int16)  # [2*n_pos, bits]
    signed = bits_unp * 2 - 1                                  # 0→-1, 1→+1
    summed = signed.sum(axis=0)                                # [bits]
    out = (summed > 0).astype(np.uint8)                        # majority
    return np.packbits(out)


def role_permute(hv: np.ndarray, asset_id: int) -> np.ndarray:
    n = hv.size
    shift = (asset_id * 7 + 3) % n
    return np.roll(hv, shift)


def encode_multi_asset(per_asset: Mapping[int, Sequence[float]], bits: int = 1024) -> np.ndarray:
    hv = np.zeros(bits // 8, dtype=np.uint8)
    for asset_id, returns in per_asset.items():
        h = encode_window(returns, bits=bits)
        h = role_permute(h, asset_id)
        hv ^= h
    return hv


def hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.bitwise_xor(a, b)
    n_bits = a.size * 8
    n_diff = int(np.unpackbits(diff).sum())
    return 1.0 - (n_diff / n_bits)


# Compatibility shim — older code referenced _bucket / N_BUCKETS.
N_BUCKETS = N_RET


def _bucket(ret: float) -> int:
    return int(_return_bucket(np.array([ret])).item()) - N_RET
