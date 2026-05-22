"""
Multi-Asset Data Processor for I.A.R.A. v6.0
=============================================
Extended indicators with momentum filtering.

Features:
- Multi-asset support (all forex majors)
- Core indicators: RSI, MACD, BBands, ATR
- NEW: Stochastic Oscillator (%K, %D) for fake breakout filtering
- NEW: Price Velocity (rate of change) for momentum
- Cyclical time encoding (hour, day of week)
- Smart label filtering (volatility threshold)
- Unified StandardScaler across all assets
"""

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
RAW_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
SCALER_PATH = Path(__file__).parent.parent.parent / "data" / "scaler.pkl"


@dataclass
class NormalizerState:
    """Serializable state for the normalizer."""
    feature_names: List[str] = field(default_factory=list)
    scaler_params: Dict = field(default_factory=dict)


class Normalizer:
    """Multi-asset normalizer with unified scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.fitted = False
    
    def fit(self, features: np.ndarray, feature_names: List[str]):
        """Fit scaler on pooled multi-asset data."""
        self.scaler.fit(features)
        self.feature_names = feature_names
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet")
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, feature_names)
        return self.transform(features)
    
    def save(self, path: Path = SCALER_PATH):
        """Save normalizer state."""
        state = NormalizerState(
            feature_names=self.feature_names,
            scaler_params={
                'mean_': self.scaler.mean_.tolist(),
                'var_': self.scaler.var_.tolist(),
                'scale_': self.scaler.scale_.tolist(),
                'n_features_in_': self.scaler.n_features_in_
            }
        )
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"Saved normalizer to {path}")
    
    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> 'Normalizer':
        """Load normalizer from file."""
        normalizer = cls()
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        normalizer.feature_names = state.feature_names
        normalizer.scaler.mean_ = np.array(state.scaler_params['mean_'])
        normalizer.scaler.var_ = np.array(state.scaler_params['var_'])
        normalizer.scaler.scale_ = np.array(state.scaler_params['scale_'])
        normalizer.scaler.n_features_in_ = state.scaler_params['n_features_in_']
        normalizer.fitted = True
        
        logger.info(f"Loaded normalizer. Features: {len(normalizer.feature_names)}")
        return normalizer


def compute_rsi(close: pl.Series, period: int = 14) -> pl.Series:
    """Compute RSI indicator."""
    delta = close.diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)
    
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def compute_macd(close: pl.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pl.Series, pl.Series, pl.Series]:
    """Compute MACD indicator."""
    ema_fast = close.ewm_mean(span=fast, min_samples=fast)
    ema_slow = close.ewm_mean(span=slow, min_samples=slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm_mean(span=signal, min_samples=signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger(close: pl.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pl.Series, pl.Series, pl.Series, pl.Series]:
    """Compute Bollinger Bands."""
    sma = close.rolling_mean(window_size=period)
    std = close.rolling_std(window_size=period)
    
    upper = sma + num_std * std
    lower = sma - num_std * std
    
    # Percent B: position within bands
    pct_b = (close - lower) / (upper - lower + 1e-10)
    
    # Bandwidth: volatility measure
    bandwidth = (upper - lower) / (sma + 1e-10)
    
    return upper, lower, pct_b, bandwidth


def compute_atr(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pl.max_horizontal(tr1, tr2, tr3)
    atr = tr.rolling_mean(window_size=period)
    
    return atr


def compute_stochastic(high: pl.Series, low: pl.Series, close: pl.Series, 
                       k_period: int = 14, d_period: int = 3) -> Tuple[pl.Series, pl.Series]:
    """
    Compute Stochastic Oscillator (14, 3, 3).
    Critical for filtering fake breakouts.
    """
    lowest_low = low.rolling_min(window_size=k_period)
    highest_high = high.rolling_max(window_size=k_period)
    
    # %K: Raw stochastic
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # %D: Smoothed %K
    stoch_d = stoch_k.rolling_mean(window_size=d_period)
    
    return stoch_k, stoch_d


def compute_price_velocity(close: pl.Series, period: int = 5) -> pl.Series:
    """
    Compute Price Velocity (1st derivative of price).
    Rate of change over the last N periods.
    """
    # Rate of change: (close - close[n]) / close[n]
    roc = (close - close.shift(period)) / (close.shift(period) + 1e-10) * 100
    return roc


def compute_adx(high: pl.Series, low: pl.Series, close: pl.Series, period: int = 14) -> pl.Series:
    """
    Compute Average Directional Index (ADX).
    Measures trend strength (0-100).
    """
    # 1. True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pl.max_horizontal(tr1, tr2, tr3)
    
    # 2. Directional Movement
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    pos_dm = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    neg_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    
    # 3. Smooth TR, +DM, -DM (Wilder's Smoothing = EMA with alpha=1/period)
    tr_smooth = tr.ewm_mean(alpha=1/period, min_samples=period)
    pos_dm_smooth = pos_dm.ewm_mean(alpha=1/period, min_samples=period)
    neg_dm_smooth = neg_dm.ewm_mean(alpha=1/period, min_samples=period)
    
    # 4. DI
    pos_di = 100 * (pos_dm_smooth / (tr_smooth + 1e-10))
    neg_di = 100 * (neg_dm_smooth / (tr_smooth + 1e-10))
    
    # 5. DX
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-10)
    
    # 6. ADX
    adx = dx.ewm_mean(alpha=1/period, min_samples=period)
    
    return adx


def compute_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute cyclical time features using native Polars."""
    # Extract hour and day of week
    if "time" in df.columns:
        hour = df["time"].dt.hour()
        day = df["time"].dt.weekday()
    else:
        # Fallback: create dummy features
        n = len(df)
        hour = pl.Series("hour", [12] * n)
        day = pl.Series("day", [2] * n)
    
    # Cyclical encoding
    hour_rad = hour * (2 * math.pi / 24)
    day_rad = day * (2 * math.pi / 7)
    
    return df.with_columns([
        hour.alias("hour"),
        hour_rad.sin().alias("hour_sin"),
        hour_rad.cos().alias("hour_cos"),
        day_rad.sin().alias("day_sin"),
        day_rad.cos().alias("day_cos")
    ])


def process_single_asset(file_path: Path, volatility_threshold: float = 0.0002) -> Optional[pl.DataFrame]:
    """Process a single asset file with all indicators."""
    try:
        df = pl.read_parquet(file_path)
        ticker = file_path.stem
        
        # Required columns
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            logger.warning(f"{ticker}: Missing OHLC columns")
            return None
        
        # Compute indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]
        
        # Core Indicators
        rsi = compute_rsi(close, 14)
        macd, macd_signal, macd_hist = compute_macd(close)
        bb_upper, bb_lower, bb_pct_b, bb_width = compute_bollinger(close)
        atr = compute_atr(high, low, close, 14)
        
        # NEW v6.0: Stochastic Oscillator
        stoch_k, stoch_d = compute_stochastic(high, low, close, 14, 3)
        
        # NEW v6.0: Price Velocity
        velocity = compute_price_velocity(close, 5)
        
        # NEW v6.1: ADX
        adx = compute_adx(high, low, close, 14)
        
        # NEW v6.7: Global Sentiment (EMA 50 Trend Proxy)
        ema_50 = close.ewm_mean(span=50, min_samples=50)
        global_sentiment = (close - ema_50) / (ema_50 + 1e-10) * 100
        
        # Add all indicators
        df = df.with_columns([
            rsi.alias("rsi"),
            macd.alias("macd"),
            macd_signal.alias("macd_signal"),
            macd_hist.alias("macd_hist"),
            bb_pct_b.alias("bb_pct_b"),
            bb_width.alias("bb_width"),
            atr.alias("atr"),
            stoch_k.alias("stoch_k"),
            stoch_d.alias("stoch_d"),
            velocity.alias("velocity"),
            adx.alias("adx"),
            global_sentiment.alias("global_sentiment")
        ])
        
        # Time features
        df = compute_time_features(df)
        
        # Target: log return
        close_log_ret = close.log().diff()
        df = df.with_columns([
            close_log_ret.alias("close_log_ret")
        ])
        
        # Smart label: valid if volatility > threshold
        valid_sample = close_log_ret.abs() > volatility_threshold
        df = df.with_columns([
            valid_sample.alias("valid_sample")
        ])
        
        # Add ticker column
        df = df.with_columns([
            pl.lit(ticker).alias("ticker")
        ])
        
        # Drop warmup rows (increased for EMA 50)
        df = df.slice(50, len(df) - 50)
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def process_all_assets(
    tickers: Optional[List[str]] = None,
    volatility_threshold: float = 0.0001  # Lowered to 1 pip for 5m data
) -> None:
    """Process all assets with unified scaling."""
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get files to process
    if tickers:
        files = [RAW_DATA_DIR / f"{t}.parquet" for t in tickers if (RAW_DATA_DIR / f"{t}.parquet").exists()]
    else:
        files = list(RAW_DATA_DIR.glob("*.parquet"))
    
    logger.info(f"Processing {len(files)} assets...")
    
    # Feature columns to normalize (v6.7: 16 features)
    feature_cols = [
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_pct_b", "bb_width", "atr",
        "stoch_k", "stoch_d",
        "velocity",
        "adx",
        "global_sentiment", # NEW v6.7
        "hour_sin", "hour_cos", "day_sin", "day_cos"
    ]
    
    # First pass: collect all data for unified scaling
    all_features = []
    processed_dfs = {}
    
    for file_path in files:
        df = process_single_asset(file_path, volatility_threshold)
        if df is not None:
            ticker = file_path.stem
            processed_dfs[ticker] = df
            
            # Extract features for scaling
            available_cols = [c for c in feature_cols if c in df.columns]
            features = df.select(available_cols).to_numpy()
            
            # Remove NaN rows for scaling
            mask = ~np.isnan(features).any(axis=1)
            all_features.append(features[mask])
    
    if not all_features:
        logger.error("No data processed!")
        return
    
    # Fit normalizer on pooled data
    pooled_features = np.vstack(all_features)
    logger.info(f"Pooled features shape: {pooled_features.shape}")
    
    normalizer = Normalizer()
    normalizer.fit(pooled_features, feature_cols)
    normalizer.save()
    
    # Second pass: normalize and save
    valid_count = 0
    total_samples = 0
    
    for ticker, df in processed_dfs.items():
        available_cols = [c for c in feature_cols if c in df.columns]
        features = df.select(available_cols).to_numpy()
        
        # Normalize
        features_norm = normalizer.transform(features)
        
        # Add normalized columns
        for i, col in enumerate(available_cols):
            df = df.with_columns([
                pl.Series(f"{col}_norm", features_norm[:, i])
            ])
        
        # Save
        output_path = PROCESSED_DATA_DIR / f"{ticker}.parquet"
        df.write_parquet(output_path)
        
        # Stats
        valid_mask = df["valid_sample"].to_numpy()
        valid_count += valid_mask.sum()
        total_samples += len(df)
        
        logger.info(f"  {ticker}: {len(df)} samples, {valid_mask.sum()} valid ({valid_mask.mean()*100:.1f}%)")
    
    logger.info(f"Processing complete. {len(processed_dfs)} tickers, {total_samples} total samples, {valid_count} valid ({valid_count/total_samples*100:.1f}%)")
    logger.info(f"Features ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    # Process main 4 pairs as requested + others if available
    target_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    
    # Check what we have
    available = [f.stem for f in RAW_DATA_DIR.glob("*.parquet")]
    to_process = [t for t in target_pairs if t in available]
    
    if len(to_process) < len(target_pairs):
        logger.warning(f"Only found {len(to_process)} of 4 target pairs: {to_process}")
    
    process_all_assets(tickers=to_process)
