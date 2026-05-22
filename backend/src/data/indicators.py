"""
I.A.R.A. — Centralized Feature Engineering (indicators.py)
==========================================================
Single source of truth for all technical indicators used in training,
backtesting, and live inference.

Features (22 total):
  Core (12): RSI, MACD (3), BBands (2), ATR, Stochastic (2), Velocity, ADX, Sentiment
  Mindset (6): Dist SMA200, Dist SMA20, Dist Resistance, Dist Support, Upper Wick, Lower Wick
  Time (4): hour_sin, hour_cos, day_sin, day_cos

References:
  - feature_factory.py (training)
  - backtest_v9.py (backtesting)
  - signal_generator.py (live inference)

Usage:
    from src.data.indicators import compute_features, compute_time_features, compute_trend
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.data.hurst import compute_hurst_rolling

logger = logging.getLogger("indicators")

try:
    import talib
except ImportError:
    talib = None
    logger.warning("TA-Lib not found. Using Mock (Zero) Indicators. Install ta-lib for real features.")

# Mock if missing
if talib is None:
    class MockTalib:
        def RSI(self, data, timeperiod=14):
            return np.zeros_like(data)
        def MACD(self, data, fastperiod=12, slowperiod=26, signalperiod=9):
            return np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)
        def BBANDS(self, data, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
            return data, data, data # Upper, Middle, Lower
        def ATR(self, high, low, close, timeperiod=14):
            return np.zeros_like(close)
        def STOCH(self, high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
            return np.zeros_like(close), np.zeros_like(close)
        def ROC(self, data, timeperiod=10):
            return np.zeros_like(data)
        def ADX(self, high, low, close, timeperiod=14):
            return np.zeros_like(close)
        def EMA(self, data, timeperiod=30):
            return data 
        def SMA(self, data, timeperiod=30):
            return data
        def MAX(self, data, timeperiod=30):
            return data
        def MIN(self, data, timeperiod=30):
            return data

    talib = MockTalib()

# ──────────────────────────────────────────────
# Feature Names (Canonical Order)
# ──────────────────────────────────────────────
FEATURE_NAMES_CORE = [
    "rsi_norm",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_width_norm", "bb_pct_b_norm",
    "atr_norm",
    "stoch_k_norm", "stoch_d_norm",
    "velocity_norm",
    "adx_norm",
    "sentiment_norm",
    "volume_norm", "vma_norm",
    "cvd_norm", "oi_norm",
    "hurst_norm", "fractal_dim_norm",
]

FEATURE_NAMES_MINDSET = [
    "dist_sma_200", "dist_sma_20",
    "dist_res_50", "dist_sup_50",
    "upper_wick_ratio", "lower_wick_ratio",
]

FEATURE_NAMES_TIME = [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
]

FEATURE_NAMES_ALL = FEATURE_NAMES_CORE + FEATURE_NAMES_MINDSET + FEATURE_NAMES_TIME
NUM_FEATURES = len(FEATURE_NAMES_ALL)  # 26 (was 24, +2 Hurst/Fractal)


# ──────────────────────────────────────────────
# Core Feature Computation
# ──────────────────────────────────────────────
def compute_features(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    opens: np.ndarray,
    volume: Optional[np.ndarray] = None,
    timestamps: Optional[pd.Series] = None,
    include_time: bool = True,
) -> np.ndarray:
    """
    Compute the canonical 22-feature vector from OHLC data.

    Args:
        close: Close prices (float64)
        high: High prices (float64)
        low: Low prices (float64)
        opens: Open prices (float64)
        timestamps: Optional datetime Series for time features
        include_time: Whether to include time features (4 cols)

    Returns:
        np.ndarray of shape [N, 22] (or [N, 18] if include_time=False)
    """
    close = close.astype(np.float64)
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    opens = opens.astype(np.float64)

    # ── 1. CORE 12 ──
    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    rsi_norm = (rsi - 50) / 25

    # MACD
    macd, macdsignal, macdhist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    # Normalize MACD as a PERCENTAGE of price so it is scale-invariant.
    # `macd * 100` was in raw price units: fine for forex (close≈1) but
    # exploded for BTC (close≈60k → values ±190k → NaN loss). Dividing by
    # close keeps forex values ~unchanged while making crypto sane.
    macd_norm = (macd / (close + 1e-10)) * 100
    macd_signal_norm = (macdsignal / (close + 1e-10)) * 100
    macd_hist_norm = (macdhist / (close + 1e-10)) * 100

    # Bollinger Bands
    upper, middle, lower_bb = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = (upper - lower_bb) / (middle + 1e-10)
        bb_pct_b = (close - lower_bb) / (upper - lower_bb + 1e-10)
    bb_width_norm = np.nan_to_num(bb_width) * 10
    bb_pct_b_norm = (np.nan_to_num(bb_pct_b) - 0.5) * 2

    # ATR
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_norm = (atr / (close + 1e-10)) * 100

    # Stochastic
    stoch_k, stoch_d = talib.STOCH(
        high, low, close,
        fastk_period=14, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0,
    )
    stoch_k_norm = (stoch_k - 50) / 25
    stoch_d_norm = (stoch_d - 50) / 25

    # Velocity (ROC)
    velocity = talib.ROC(close, timeperiod=5)
    velocity_norm = velocity * 10

    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx_norm = (adx - 25) / 25

    # Global Sentiment (EMA 50)
    ema_50 = talib.EMA(close, timeperiod=50)
    ema_50 = np.nan_to_num(ema_50, nan=close[0] if len(close) > 0 else 0)
    sentiment = (close - ema_50) / (ema_50 + 1e-10) * 100
    sentiment_norm = sentiment * 10

    # Volume Context
    if volume is not None and len(volume) > 0:
        volume = volume.astype(np.float64)
        vma = talib.SMA(volume, timeperiod=20)
        vma = np.nan_to_num(vma, nan=volume[0] if len(volume) > 0 else 0)
        volume_norm = ((volume - vma) / (vma + 1e-10)) * 10
        
        vma_long = talib.SMA(volume, timeperiod=50)
        vma_long = np.nan_to_num(vma_long, nan=volume[0] if len(volume) > 0 else 0)
        vma_norm = ((vma - vma_long) / (vma_long + 1e-10)) * 10
    else:
        volume_norm = np.zeros_like(close)
        vma_norm = np.zeros_like(close)

    # Hurst Exponent & Fractal Dimension (Chaos Mathematics).
    # step=10 (was 1): rolling Hurst was the sharding bottleneck (~13min on
    # 450k bars). It is slow-moving over a 200-bar window, AND it lives in cols
    # >12 which the model NEVER consumes (model uses x[:, :12] core). So a
    # coarser step + ffill is a ~10× speedup with zero effect on training/P&L.
    hurst_raw = compute_hurst_rolling(close, window=200, step=10)
    hurst_norm = (hurst_raw - 0.5) * 4.0  # Center at 0, scale to [-2, 2]
    fractal_dim_raw = 2.0 - hurst_raw
    fractal_dim_norm = (fractal_dim_raw - 1.5) * 4.0  # Center at 0

    # ── 2. MINDSET 6 ──
    # Trend Alignment (SMA 200 & 20)
    sma_200 = talib.SMA(close, timeperiod=200)
    sma_200 = np.nan_to_num(sma_200, nan=close[0] if len(close) > 0 else 0)
    sma_20 = talib.SMA(close, timeperiod=20)
    sma_20 = np.nan_to_num(sma_20, nan=close[0] if len(close) > 0 else 0)

    dist_sma_200 = (close - sma_200) / (sma_200 + 1e-10) * 100
    dist_sma_20 = (close - sma_20) / (sma_20 + 1e-10) * 100

    # Support & Resistance (Rolling 50)
    roll_max_50 = talib.MAX(high, timeperiod=50)
    roll_min_50 = talib.MIN(low, timeperiod=50)
    roll_max_50 = np.nan_to_num(roll_max_50, nan=close[0] if len(close) > 0 else 0)
    roll_min_50 = np.nan_to_num(roll_min_50, nan=close[0] if len(close) > 0 else 0)

    dist_res_50 = (roll_max_50 - close) / (close + 1e-10) * 100
    dist_sup_50 = (close - roll_min_50) / (close + 1e-10) * 100

    # Candle Physics (Wick Ratios)
    candle_range = high - low
    upper_wick = high - np.maximum(close, opens)
    upper_wick_ratio = upper_wick / (candle_range + 1e-10)
    lower_wick = np.minimum(close, opens) - low
    lower_wick_ratio = lower_wick / (candle_range + 1e-10)

    # ── Stack Core + Mindset (22 features) ──
    features = np.column_stack([
        rsi_norm,
        macd_norm, macd_signal_norm, macd_hist_norm,
        bb_width_norm, bb_pct_b_norm,
        atr_norm,
        stoch_k_norm, stoch_d_norm,
        velocity_norm,
        adx_norm,
        sentiment_norm,
        volume_norm, vma_norm,
        hurst_norm, fractal_dim_norm,
        # Mindset
        dist_sma_200, dist_sma_20,
        dist_res_50, dist_sup_50,
        upper_wick_ratio, lower_wick_ratio,
    ])

    # ── 3. TIME 4 ──
    if include_time and timestamps is not None:
        time_feats = compute_time_features_from_series(timestamps)
        features = np.column_stack([features, time_feats])
    elif include_time:
        # Pad with zeros if no timestamps provided
        features = np.column_stack([
            features, np.zeros((len(close), 4), dtype=np.float32)
        ])

    return np.nan_to_num(features).astype(np.float32)


# ──────────────────────────────────────────────
# Time Features
# ──────────────────────────────────────────────
def compute_time_features_from_series(dt_series: pd.Series) -> np.ndarray:
    """
    Compute cyclical time features from a pandas datetime Series.

    Returns:
        np.ndarray of shape [N, 4] with [hour_sin, hour_cos, day_sin, day_cos]
    """
    dt_index = pd.to_datetime(dt_series)
    hours = dt_index.dt.hour.values
    days = dt_index.dt.dayofweek.values

    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    day_sin = np.sin(2 * np.pi * days / 7).astype(np.float32)
    day_cos = np.cos(2 * np.pi * days / 7).astype(np.float32)

    return np.stack([hour_sin, hour_cos, day_sin, day_cos], axis=1)


def compute_time_features_polars(df) -> np.ndarray:
    """
    Compute cyclical time features from a Polars DataFrame with 'date' column.
    Fallback to zeros if 'date' column is missing or unparseable.

    Returns:
        np.ndarray of shape [N, 4] with [hour_sin, hour_cos, day_sin, day_cos]
    """
    import polars as pl

    if "date" not in df.columns:
        logger.warning("No 'date' column found, using zeros for time features")
        return np.zeros((len(df), 4), dtype=np.float32)

    try:
        dt_series = df["date"]
        if dt_series.dtype == pl.Utf8:
            dt_series = dt_series.str.to_datetime()
        return compute_time_features_from_series(dt_series.to_pandas())
    except Exception as e:
        logger.warning(f"Could not parse dates: {e}, using zeros")
        return np.zeros((len(df), 4), dtype=np.float32)


# ──────────────────────────────────────────────
# Trend Detection
# ──────────────────────────────────────────────
def compute_trend(close_arr: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Simple trend detection using EMA slope.

    Returns:
        np.ndarray with values: 1 (Up), -1 (Down), 0 (Flat)
    """
    if len(close_arr) < period:
        return np.zeros(len(close_arr))

    close_arr = np.nan_to_num(
        close_arr.astype(np.float64),
        nan=close_arr[0] if len(close_arr) > 0 else 0,
    )

    ema = talib.EMA(close_arr, timeperiod=period)

    slope = np.zeros_like(ema)
    slope[period:] = ema[period:] - ema[period - 1 : -1]

    with np.errstate(divide="ignore", invalid="ignore"):
        slope_pct = slope / close_arr
        slope_pct = np.nan_to_num(slope_pct)

    threshold = 0.00005  # 0.005% slope threshold
    trend = np.zeros_like(slope)
    trend[slope_pct > threshold] = 1
    trend[slope_pct < -threshold] = -1

    return trend


# ──────────────────────────────────────────────
# Convenience: Compute from DataFrame
# ──────────────────────────────────────────────
def compute_features_from_df(
    df: pd.DataFrame,
    time_col: str = "time",
    include_time: bool = True,
) -> np.ndarray:
    """
    Compute features from a pandas DataFrame with OHLCV columns.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns
        time_col: Name of the timestamp column (for time features)
        include_time: Whether to include cyclical time features

    Returns:
        np.ndarray of shape [N, 22] (or [N, 18])
    """
    timestamps = None
    if include_time and time_col in df.columns:
        timestamps = pd.to_datetime(df[time_col])

    return compute_features(
        close=df["close"].values,
        high=df["high"].values,
        low=df["low"].values,
        opens=df["open"].values,
        volume=df["volume"].values if "volume" in df.columns else None,
        timestamps=timestamps,
        include_time=include_time,
    )
