"""saci-brain configuration — single source of truth for paths, sizes,
window lengths, correlation parameters. Override via env vars for ops."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _data_root() -> Path:
    """Where brain_kg writes its artifacts.

    Inside the saci-brain container the mounted host `./data` lives at
    `/app/data` and is writeable (container user has rw on the bind
    mount). For local dev outside the container we fall back to
    `backend/data_brain/` next to the source tree.
    """
    if Path("/app/data").exists():
        return Path("/app/data") / "brain_kg"
    # backend/src/brain_kg/config.py → backend/
    return Path(__file__).resolve().parents[2] / "data_brain"


def _raw_dir() -> Path:
    # priority: explicit env, then data_brain/raw (host-writable mirror),
    # then /app/data/raw (when docker-compose mounts it), then host repo.
    if "SACI_BRAIN_RAW_DIR" in os.environ:
        return Path(os.environ["SACI_BRAIN_RAW_DIR"])
    cand = _data_root() / "raw"
    if cand.exists():
        return cand
    if Path("/app/data/raw").exists():
        return Path("/app/data/raw")
    return Path(__file__).resolve().parents[3] / "data" / "raw"


@dataclass
class BrainConfig:
    # ── paths ─────────────────────────────────────────────────────────
    raw_dir: Path = field(default_factory=_raw_dir)
    wal_dir: Path = field(default_factory=lambda: _data_root() / "wal")
    kg_dir: Path  = field(default_factory=lambda: _data_root() / "kg")
    mem_dir: Path = field(default_factory=lambda: _data_root() / "memory")

    # ── KG ───────────────────────────────────────────────────────────
    corr_windows_bars: tuple[int, ...] = (60, 288, 864)  # ~5h, ~24h, ~3d at M5
    corr_min_abs: float = 0.20                            # drop weak edges
    corr_decay: float = 0.90                              # weight when re-emitted

    # ── forest ───────────────────────────────────────────────────────
    forest_propagate_depth: int = 1                       # mycelium hops

    # ── HDC ──────────────────────────────────────────────────────────
    hdc_bits: int = 1024                                  # 1024-bit hypervector
    hdc_window_bars: int = 32                             # bars per asset HV
    hdc_top_k: int = 12                                   # retrieval k
    hdc_min_hamming_sim: float = 0.55

    # ── runtime flags ────────────────────────────────────────────────
    enable_brain: bool = True

    def __post_init__(self) -> None:
        # let env vars override paths in deploy
        for name in ("wal_dir", "kg_dir", "mem_dir"):
            envk = "SACI_BRAIN_" + name.upper()
            if envk in os.environ:
                setattr(self, name, Path(os.environ[envk]))
        for p in (self.wal_dir, self.kg_dir, self.mem_dir):
            p.mkdir(parents=True, exist_ok=True)


def default_config() -> BrainConfig:
    return BrainConfig()
