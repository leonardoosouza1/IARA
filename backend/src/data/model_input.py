"""Single source of truth for the SwarmBrain v9 model input.

Three code paths used to build the model's input independently — the training
dataset, the backtester, and the standalone P&L backtest — and any drift between
them produced silent, expensive bugs (the "85% that lost -97%"). This module is
the ONE definition they all share:

  - compute_core12      : the exact 12 features the model consumes (bit-identical
                          to indicators.compute_features[:, :12]) — no Hurst,
                          no volume, no time. Cheap.
  - build_features      : OHLCV df → (feats12, close, dates, consensus) for a
                          given base timeframe. Used by backtest + (future) shard.
  - make_timeframe_dict : x_core[seq, 12] → {M5,M15,M30,H1,H4} multi-timeframe
                          dict via downsampling. Used by train __getitem__ AND
                          backtester. Identical logic, one place.

Contract is locked by tests/test_model_input_contract.py.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
import talib
import torch

CORE12_NAMES = [
    "rsi_norm", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_width_norm", "bb_pct_b_norm", "atr_norm",
    "stoch_k_norm", "stoch_d_norm", "velocity_norm", "adx_norm", "sentiment_norm",
]


def compute_core12(close, high, low, opens) -> np.ndarray:
    """The 12 features the model actually consumes, in order. Bit-identical to
    indicators.compute_features(...)[:, :12] (same formulas) but skips the ~20
    discarded features (Hurst/FFT/CVD/OI/SMA/wicks/time) → much cheaper."""
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    opens = opens.astype(np.float64)

    rsi_norm = (talib.RSI(close, timeperiod=14) - 50) / 25

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_norm = (macd / (close + 1e-10)) * 100
    macd_signal_norm = (macdsignal / (close + 1e-10)) * 100
    macd_hist_norm = (macdhist / (close + 1e-10)) * 100

    upper, middle, lower_bb = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = (upper - lower_bb) / (middle + 1e-10)
        bb_pct_b = (close - lower_bb) / (upper - lower_bb + 1e-10)
    bb_width_norm = np.nan_to_num(bb_width) * 10
    bb_pct_b_norm = (np.nan_to_num(bb_pct_b) - 0.5) * 2

    atr_norm = (talib.ATR(high, low, close, timeperiod=14) / (close + 1e-10)) * 100

    stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3,
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
    stoch_k_norm = (stoch_k - 50) / 25
    stoch_d_norm = (stoch_d - 50) / 25

    velocity_norm = talib.ROC(close, timeperiod=5) * 10

    adx_norm = (talib.ADX(high, low, close, timeperiod=14) - 25) / 25

    ema_50 = talib.EMA(close, timeperiod=50)
    ema_50 = np.nan_to_num(ema_50, nan=close[0] if len(close) > 0 else 0)
    sentiment_norm = (close - ema_50) / (ema_50 + 1e-10) * 100 * 10

    feats = np.column_stack([
        rsi_norm, macd_norm, macd_signal_norm, macd_hist_norm,
        bb_width_norm, bb_pct_b_norm, atr_norm,
        stoch_k_norm, stoch_d_norm, velocity_norm, adx_norm, sentiment_norm,
    ])
    return np.nan_to_num(feats).astype(np.float32)


def _trend(close: np.ndarray, period: int = 14) -> np.ndarray:
    """EMA-slope sign (matches indicators.compute_trend) for the consensus gate."""
    if len(close) < period:
        return np.zeros(len(close))
    close = np.nan_to_num(close.astype(np.float64), nan=close[0] if len(close) else 0)
    ema = talib.EMA(close, timeperiod=period)
    slope = np.zeros_like(ema)
    slope[period:] = ema[period:] - ema[period - 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_pct = np.nan_to_num(slope / close)
    trend = np.zeros_like(slope)
    trend[slope_pct > 0.00005] = 1
    trend[slope_pct < -0.00005] = -1
    return trend


def build_features(data, base_tf: str = "5m", clip: float = 1000.0):
    """OHLCV (polars df or parquet path) → (feats12, close, dates, consensus).

    base_tf = trading bar size; context trend is ~12× the base. consensus =
    fractal alignment (base trend == context trend, != 0), the tradeable gate.
    """
    from src.data.feature_factory import resample_data  # local import (avoid cycle)

    ctx_tf = {"5m": "1h", "15m": "3h", "30m": "6h", "1h": "12h"}.get(base_tf, "1h")
    df = data if isinstance(data, pl.DataFrame) else pl.read_parquet(data)
    df = df.rename({c: c.lower() for c in df.columns})
    base = resample_data(df, base_tf)
    ctx = resample_data(df, ctx_tf)

    close = base["close"].to_numpy().astype(np.float64)
    feats = compute_core12(close, base["high"].to_numpy(), base["low"].to_numpy(),
                           base["open"].to_numpy())
    feats = np.clip(feats, -clip, clip)

    t_base = _trend(close)
    t_ctx = _trend(ctx["close"].to_numpy().astype(np.float64))
    dates = base["date"].to_pandas()
    s_ctx = pd.Series(t_ctx, index=ctx["date"].to_pandas()).reindex(dates, method="ffill").fillna(0).values
    consensus = (s_ctx == t_base) & (s_ctx != 0)
    return feats, close, dates, consensus


def make_timeframe_dict(x_core: torch.Tensor, seq_len: int) -> Dict[str, torch.Tensor]:
    """[*, seq, 12] (or [seq, 12]) → {M5,M15,M30,H1,H4} by downsampling.

    The ONE definition shared by train_v9.MultiTimeframeDataset and
    backtester._simulate_timeframes. Time axis is dim -2.
    """
    batched = x_core.dim() == 3
    t_axis = 1 if batched else 0
    n = x_core.shape[t_axis]

    def take(step: int, keep: int):
        if batched:
            return x_core[:, ::step, :][:, -keep:, :]
        return x_core[::step, :][-keep:, :]

    out = {"M5": x_core[:, -seq_len:, :] if batched else x_core[-seq_len:, :]}
    out["M15"] = take(3, max(1, seq_len // 3)) if n >= 3 else x_core
    out["M30"] = take(6, max(1, seq_len // 6)) if n >= 6 else x_core
    out["H1"] = take(12, max(1, seq_len // 12)) if n >= 12 else x_core
    out["H4"] = take(16, max(1, seq_len // 16)) if n >= 16 else x_core
    return out
