"""Orchestrator: encode current state → query KG + forest + HDC memory →
emit a typed `hints` dict for the SwarmCoordinator.

Output is small (per-asset stats + a global tensor) so it can be added
without changing the SwarmCoordinator's existing IO surface much. When
`enable_brain=False` (or any dep is missing) the orchestrator returns
None and the SwarmCoordinator runs unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import BrainConfig, default_config
from .encoder import encode_multi_asset, hamming_sim
from .forest import Forest
from .kg import REL_POS, neighbors, load_triples
from .memory import MemoryBank, retrieve as mem_retrieve, HORIZONS


@dataclass
class BrainState:
    cfg: BrainConfig
    triples: pd.DataFrame
    forest_short: Forest        # fastest window (most reactive)
    forest_long: Forest         # slowest window (regime)
    bank: Optional[MemoryBank] = None
    asset_to_id: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, cfg: Optional[BrainConfig] = None) -> "BrainState":
        cfg = cfg or default_config()
        triples = load_triples(cfg)
        windows = sorted(triples["window"].unique().tolist()) if not triples.empty else []
        short = Forest.from_kg(cfg, window=windows[0]) if windows else Forest()
        long_ = Forest.from_kg(cfg, window=windows[-1]) if windows else Forest()
        try:
            bank = MemoryBank.load(cfg.mem_dir, bits=cfg.hdc_bits)
        except FileNotFoundError:
            bank = None
        asset_to_id: dict[str, int] = {}
        for i, a in enumerate(short.assets):
            asset_to_id[a] = i
        return cls(cfg=cfg, triples=triples, forest_short=short, forest_long=long_,
                   bank=bank, asset_to_id=asset_to_id)


# ────────────────────────── Hint shape ────────────────────────────────

# A hint dict carries:
#   per_asset_pressure: { asset: signed scalar }   (forest propagation result)
#   analog_returns:    { horizon: { asset: {'mean','std','p10','p90'} } }
#   kg_top_edges:      list of (src, dst, signed_weight)  -- top 5 by |w|
#   active_assets:     list of assets known to the forest (stable order)
#   vector:            np.ndarray (n_assets * 5) f32 — compact tensor view
#                      [pressure, mean1h, std1h, mean4h, std4h] per asset
#                      so the SwarmCoordinator can `.from_numpy()` it directly.

def compute_hints(state: BrainState,
                  driver: Mapping[str, float],
                  windows_per_asset: Mapping[str, Sequence[float]] | None = None,
                  ) -> Optional[dict]:
    """Compute a hint payload.

    Args:
      driver  : { asset: signed log-return at current bar } — driving
                 the forest propagation.
      windows_per_asset : { asset: list[float] log-returns } sized
                 cfg.hdc_window_bars — if provided we also query the
                 HDC memory bank for the analog distribution.
    """
    if not state.cfg.enable_brain:
        return None
    if not state.forest_short.assets:
        return None

    cfg = state.cfg
    pressure_short = state.forest_short.propagate(dict(driver), depth=1)
    pressure_long  = state.forest_long.propagate(dict(driver), depth=1) \
                     if state.forest_long.assets else {}

    # Top KG edges touching any driving asset
    kg_edges: list[tuple[str, str, float]] = []
    if not state.triples.empty:
        win = state.forest_short.window
        for a in driver:
            for e in neighbors(state.triples, a, window=win)[:3]:
                signed = e.weight if e.relation == REL_POS else -e.weight
                kg_edges.append((e.src, e.dst, signed))
        kg_edges.sort(key=lambda t: -abs(t[2]))
        kg_edges = kg_edges[:5]

    # HDC analog retrieval
    analog: dict[str, dict[str, dict[str, float]]] = {}
    if state.bank is not None and windows_per_asset is not None and state.bank.hvs.size > 0:
        per_id: dict[int, Sequence[float]] = {}
        for a, w in windows_per_asset.items():
            aid = state.bank.assets.index(a) if a in state.bank.assets else None
            if aid is not None:
                per_id[aid] = w
        if per_id:
            q = encode_multi_asset(per_id, bits=cfg.hdc_bits)
            matches = mem_retrieve(state.bank, q,
                                   top_k=cfg.hdc_top_k,
                                   min_sim=cfg.hdc_min_hamming_sim)
            for name, _ in HORIZONS:
                if not matches or name not in matches[0]:
                    continue
                arr = np.array([m[name] for m in matches], dtype=np.float32)
                stat = {}
                for j, a in enumerate(state.bank.assets):
                    v = arr[:, j]
                    stat[a] = {
                        "mean": float(v.mean()),
                        "std":  float(v.std()),
                        "p10":  float(np.percentile(v, 10)),
                        "p90":  float(np.percentile(v, 90)),
                    }
                analog[name] = stat

    # Compact tensor view (stable order = forest_short.assets)
    assets = state.forest_short.assets
    vec = np.zeros((len(assets), 5), dtype=np.float32)
    a1h = analog.get("future_1h", {})
    a4h = analog.get("future_4h", {})
    for i, a in enumerate(assets):
        vec[i, 0] = pressure_short.get(a, 0.0)
        vec[i, 1] = a1h.get(a, {}).get("mean", 0.0)
        vec[i, 2] = a1h.get(a, {}).get("std",  0.0)
        vec[i, 3] = a4h.get(a, {}).get("mean", 0.0)
        vec[i, 4] = a4h.get(a, {}).get("std",  0.0)

    return {
        "per_asset_pressure": pressure_short,
        "per_asset_pressure_long": pressure_long,
        "analog_returns": analog,
        "kg_top_edges": kg_edges,
        "active_assets": assets,
        "vector": vec,
    }


def vector_for_asset(hints: dict, asset: str) -> np.ndarray:
    """Convenience: extract the 5-dim row for a single asset (zeros if not
    known). Useful for single-asset training paths."""
    if not hints or asset not in hints.get("active_assets", []):
        return np.zeros(5, dtype=np.float32)
    i = hints["active_assets"].index(asset)
    return hints["vector"][i]
