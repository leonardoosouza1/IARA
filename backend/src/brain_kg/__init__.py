"""saci-brain knowledge augmentation layer for saci-trader.

Spec: docs/architecture/SACI-BRAIN.md

Module is OPTIONAL and ADDITIVE. The SwarmBrain pipeline continues to work
unchanged when `enable_brain=False`. When enabled, brain_kg provides:

  - Cross-asset KG of correlations and causal events.
  - Gravity forest of assets connected by mycelium edges (correlations).
  - HDC multi-asset memory bank for analogy-based retrieval.
  - Append-only WAL of ticks for deterministic replay.

Storage default: backend/data_brain/{wal,kg,memory} (writeable by the
saci user — data/ is root-owned by docker artifacts and not used here).
"""

from .config import BrainConfig, default_config  # noqa: F401

__all__ = ["BrainConfig", "default_config"]
