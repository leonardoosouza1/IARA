"""Contract tests: the unified model_input must reproduce the legacy behavior
exactly, so refactoring the 3 pipelines onto it cannot silently drift.

Run in container:
  docker exec saci-brain python -m pytest tests/test_model_input_contract.py -q
  (or: python -m tests.test_model_input_contract  for a plain run)
"""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from src.data.indicators import compute_features
from src.data.feature_factory import resample_data
from src.data.model_input import compute_core12, build_features, make_timeframe_dict

DATA = "data/processed/BTCUSDM5.parquet"


def _sample(n=20000):
    df = pl.read_parquet(DATA).head(n)
    m5 = resample_data(df, "5m")
    return (m5["close"].to_numpy(), m5["high"].to_numpy(),
            m5["low"].to_numpy(), m5["open"].to_numpy(), m5["date"].to_pandas())


def test_core12_bit_identical_to_legacy():
    close, high, low, opens, ts = _sample()
    legacy = compute_features(close, high, low, opens, volume=None,
                              timestamps=ts, include_time=True)[:, :12]
    new = compute_core12(close, high, low, opens)
    assert new.shape[1] == 12
    # nan_to_num applied in both; compare with tight tolerance
    diff = np.abs(new - legacy)
    assert np.nanmax(diff) < 1e-4, f"core12 drift: max abs diff {np.nanmax(diff)}"
    print(f"[core12] max abs diff vs legacy[:, :12] = {np.nanmax(diff):.2e}  OK")


def test_make_timeframe_dict_matches_legacy_downsample():
    seq = 128
    x = torch.randn(seq, 12)
    d = make_timeframe_dict(x, seq)
    # legacy train_v9 __getitem__ logic
    assert torch.equal(d["M5"], x[-seq:])
    assert torch.equal(d["M15"], x[::3][-(seq // 3):])
    assert torch.equal(d["M30"], x[::6][-(seq // 6):])
    assert torch.equal(d["H1"], x[::12][-(seq // 12):])
    assert torch.equal(d["H4"], x[::16][-(seq // 16):])
    # batched form must agree with per-sample form
    xb = x.unsqueeze(0)
    db = make_timeframe_dict(xb, seq)
    for k in ["M5", "M15", "M30", "H1", "H4"]:
        assert torch.equal(db[k][0], d[k]), f"batched/unbatched mismatch {k}"
    print("[make_timeframe_dict] legacy downsample + batched form  OK")


def test_build_features_shape_and_consensus():
    feats, close, dates, consensus = build_features(DATA, base_tf="5m")
    assert feats.shape[1] == 12
    assert len(feats) == len(close) == len(dates) == len(consensus)
    assert not np.isnan(feats).any()
    frac = consensus.mean()
    assert 0.05 < frac < 0.95, f"consensus fraction implausible: {frac}"
    print(f"[build_features] {feats.shape} consensus={frac:.1%}  OK")


if __name__ == "__main__":
    test_core12_bit_identical_to_legacy()
    test_make_timeframe_dict_matches_legacy_downsample()
    test_build_features_shape_and_consensus()
    print("\nALL CONTRACT TESTS PASSED")
