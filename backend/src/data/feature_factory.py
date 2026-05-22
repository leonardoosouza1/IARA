#!/usr/bin/env python3
"""
I.A.R.A. v6.10 - CPU Feature Factory (Curriculum Learning Edition)
===================================================================
Converts pre-processed parquet files into ready-to-consume PyTorch Tensors.

v6.10 Improvements:
- Recalculates time features from 'date' column (fixes zero bug)
- Curriculum Learning: Threshold-based labels (ignores noise/lateral movements)
- "Golden Dataset": Only strong moves are labeled as UP/DOWN

Usage:
    docker compose exec saci-brain python src/data/feature_factory.py
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

import numpy as np
import polars as pl
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FeatureFactory")

# Paths
DATA_PROCESSED_DIR = Path("data/processed")
CACHE_DIR = Path("data/cache")

# v6.10 Curriculum Learning: Minimum return threshold
# Movements smaller than this are considered "noise" and excluded
# These are adjusted per market type
MIN_RETURN_THRESHOLD = {
    "forex": 0.0003,  # 0.03% for forex (low volatility)
    "crypto": 0.002,  # 0.2% for crypto (higher volatility)
}


import talib
import pandas as pd

# Centralized feature computation
from src.data.indicators import (
    compute_features,
    compute_time_features_polars as compute_time_features,
    compute_trend,
)


def resample_data(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """
    Resample M1 data to higher timeframes (5m, 1h).
    """
    # Ensure date is sorted
    df = df.sort("date")
    
    # Polars efficient resampling
    # Note: 'every' argument handles 5m, 1h etc.
    return (
        df.group_by_dynamic("date", every=timeframe)
        .agg([
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("volume").sum(),
        ])
    )


def process_hierarchical(
    file_path: Path,
    seq_len: int,
    stride: int,
    market_type: str = "crypto",
    label_mode: str = "trend",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process M1 data with Hierarchical Consensus (M1 + M5 + H1).

    label_mode:
      - "trend"  : label = H1 EMA-slope trend direction (legacy; NOT tradeable —
                   it lags price, see project_saci_trader_label_flaw).
      - "return" : label = SIGN of the realized future return over the horizon
                   (0=LONG if price rises, 1=SHORT if it falls). Same consensus +
                   |return|≥threshold filter, so the model answers the tradeable
                   question "a significant move is coming — which way?".
    """
    logger.info(f"Processing Hierarchical: {file_path.name}")
    
    try:
        # Load Data
        if file_path.suffix == '.csv':
            df_m1 = pl.read_csv(file_path, ignore_errors=True)
        else:
            df_m1 = pl.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None, None
        
    # Standardize column names
    df_m1 = df_m1.rename({c: c.lower() for c in df_m1.columns})
    
    # Date parsing
    if 'date' not in df_m1.columns:
        # Check alternatives
        for alt in ['datetime', 'time', 'timestamp', 'date time']:
            if alt in df_m1.columns:
                df_m1 = df_m1.rename({alt: 'date'})
                break
    
    if 'date' not in df_m1.columns:
        logger.error(f"Skipping {file_path.name}: No date column")
        return None, None
        
    # Ensure date type
    if df_m1['date'].dtype == pl.Utf8:
        try:
             # Try generic parsing first, then specific formats if needed
            df_m1 = df_m1.with_columns(pl.col('date').str.to_datetime())
        except:
             # Fallback using pandas for robust parsing
            pdf = df_m1.to_pandas()
            pdf['date'] = pd.to_datetime(pdf['date'])
            df_m1 = pl.from_pandas(pdf)

    # 1. Resample Layer
    # 1. Resample Layer
    # Base is now M5 (5 minutes)
    logger.info(f"  Resampling M1 ({len(df_m1)}) -> M5 (Base), M15, M30, H1 (Context)...")
    df_m5 = resample_data(df_m1, "5m")
    
    # Contexts
    df_m15 = resample_data(df_m1, "15m")
    df_m30 = resample_data(df_m1, "30m")
    df_h1 = resample_data(df_m1, "1h")
    
    if len(df_h1) < 50:
        logger.error(f"Skipping {file_path.name}: Transformation resulted in too few H1 samples")
        return None, None

    # 2. Compute Trends for each layer
    close_m5 = df_m5['close'].to_numpy()
    close_m15 = df_m15['close'].to_numpy()
    close_m30 = df_m30['close'].to_numpy()
    close_h1 = df_h1['close'].to_numpy()
    
    # We use float64 for talib then cast back
    trend_m5 = compute_trend(close_m5.astype(np.float64))
    trend_m15 = compute_trend(close_m15.astype(np.float64))
    trend_m30 = compute_trend(close_m30.astype(np.float64))
    trend_h1 = compute_trend(close_h1.astype(np.float64))
    
    # Also calc RSI for contexts (Reversal detection)
    rsi_m15 = talib.RSI(close_m15.astype(np.float64), timeperiod=14)
    rsi_m30 = talib.RSI(close_m30.astype(np.float64), timeperiod=14)
    
    # Normalize Context RSIs
    rsi_m15_norm = (rsi_m15 - 50) / 25
    rsi_m30_norm = (rsi_m30 - 50) / 25
    
    # 3. Align Trends (Consensus)
    # Map Contexts back to M5 timestamps
    
    pdf_m5 = df_m5['date'].to_pandas()
    pdf_m15 = df_m15['date'].to_pandas()
    pdf_m30 = df_m30['date'].to_pandas()
    pdf_h1 = df_h1['date'].to_pandas()
    
    # Reindex Contexts to match M5 index (ffill)
    # We use reindex to propagate the last known macro state to the current micro state
    s_trend_m15 = pd.Series(trend_m15, index=pdf_m15).reindex(pdf_m5, method='ffill').fillna(0)
    s_trend_m30 = pd.Series(trend_m30, index=pdf_m30).reindex(pdf_m5, method='ffill').fillna(0)
    s_trend_h1 = pd.Series(trend_h1, index=pdf_h1).reindex(pdf_m5, method='ffill').fillna(0)
    
    s_rsi_m15 = pd.Series(rsi_m15_norm, index=pdf_m15).reindex(pdf_m5, method='ffill').fillna(0)
    s_rsi_m30 = pd.Series(rsi_m30_norm, index=pdf_m30).reindex(pdf_m5, method='ffill').fillna(0)
    
    aligned_trend_m5 = trend_m5 # M5 is base
    aligned_trend_m15 = s_trend_m15.values
    aligned_trend_m30 = s_trend_m30.values
    aligned_trend_h1 = s_trend_h1.values
    
    aligned_rsi_m15 = s_rsi_m15.values
    aligned_rsi_m30 = s_rsi_m30.values
    
    # Consensus Mask: H1 == M5 (Fractal Alignment)
    # Only keep samples where Medium and Long term agree
    consensus_mask = (aligned_trend_h1 == aligned_trend_m5) & (aligned_trend_h1 != 0)
    
    # Create Labels for M5 based on the consensus
    # IARA v9 Map: 0: LONG, 1: SHORT, 2: NEUTRAL (WAIT)
    # 🚀 Phase 11: Golden Dataset - Strict Thresholds
    threshold = MIN_RETURN_THRESHOLD.get(market_type, 0.001)
    
    labels = np.full(len(close_m5), 2.0, dtype=np.float32) # Default to 2.0 (Neutral)
    
    # Simple future return for the next period (approx 1 hour / 12 bars)
    # Note: feature_factory shifts targets later, here we just define the raw label
    # but we will use the threshold during the sequence loop for maximum refinement.
    
    labels[aligned_trend_h1 == 1] = 0.0  # UP -> LONG
    labels[aligned_trend_h1 == -1] = 1.0 # DOWN -> SHORT
    
    # 4. Generate M5 Features (using centralized indicators)
    c_m5 = close_m5.astype(np.float64)
    highs = df_m5['high'].to_numpy().astype(np.float64)
    lows = df_m5['low'].to_numpy().astype(np.float64)
    opens = df_m5['open'].to_numpy().astype(np.float64)
    
    # Time features from Polars 'date' column
    time_feats = compute_time_features(df_m5)
    
    # Order Flow / Macro Features (Phase 5)
    # Estimate Delta based on Close relative to High/Low
    range_total = df_m5['high'].to_numpy() - df_m5['low'].to_numpy()
    range_total = np.where(range_total == 0, 1e-8, range_total)
    buy_pressure = (df_m5['close'].to_numpy() - df_m5['low'].to_numpy()) / range_total
    sell_pressure = (df_m5['high'].to_numpy() - df_m5['close'].to_numpy()) / range_total
    
    volume_m5 = df_m5['volume'].to_numpy().astype(np.float64)
    delta = (buy_pressure - sell_pressure) * volume_m5
    
    # CVD (Cumulative Volume Delta) - Rolling sum over last 24 periods
    cvd_m5 = pd.Series(delta).rolling(window=24, min_periods=1).sum().to_numpy()
    
    # Open Interest Proxy (Using Volume Moving Average as a basic stand-in if OI not available)
    if 'open_interest' in df_m5.columns:
        oi_m5 = df_m5['open_interest'].to_numpy()
    else:
        oi_m5 = pd.Series(volume_m5).rolling(window=48, min_periods=1).mean().to_numpy()
        
    # Scale CVD and OI for NN consumption
    cvd_m5 = (cvd_m5 - np.mean(cvd_m5)) / (np.std(cvd_m5) + 1e-8)
    oi_m5 = (oi_m5 - np.mean(oi_m5)) / (np.std(oi_m5) + 1e-8)
    
    # 🌊 Phase 21: Fast Fourier Transform (FFT) Temporal Extraction
    def get_rolling_fft(series, window):
        shape = series.shape[:-1] + (series.shape[-1] - window + 1, window)
        strides = series.strides + (series.strides[-1],)
        rolled = np.lib.stride_tricks.as_strided(series, shape=shape, strides=strides)
        
        # Detrend to remove DC component dominance
        means = np.mean(rolled, axis=1, keepdims=True)
        detrended = rolled - means
        
        fft_vals = np.abs(np.fft.rfft(detrended, axis=1))
        # Max amplitude of frequencies (excluding DC)
        max_fft = np.max(fft_vals[:, 1:], axis=1) if fft_vals.shape[1] > 1 else np.zeros((fft_vals.shape[0],))
        
        padded = np.zeros(len(series))
        padded[window-1:] = max_fft
        return (padded - np.mean(padded)) / (np.std(padded) + 1e-8)

    fft_m5 = get_rolling_fft(c_m5, 24)

    # Compute canonical features (Core 14 + Mindset 6 + Time 4)
    features_24 = compute_features(
        close=c_m5,
        high=highs,
        low=lows,
        opens=opens,
        volume=volume_m5,
        timestamps=df_m5['date'].to_pandas() if 'date' in df_m5.columns else None,
        include_time=True,
    )
    
    # Append Macro/Order Flow features
    features = np.column_stack([
        features_24,                 # 24 Canonical features
        cvd_m5, oi_m5,               # +2 Volumes/Delta
        fft_m5,                      # +1 Temporal FFT
        opens, highs, lows, c_m5     # +4 OHLC
    ])                               # Total: 31 Features
    
    features = np.nan_to_num(features, nan=0.0).astype(np.float32)

    # Overflow guard on the model-fed indicator columns (everything except the
    # last 4 raw OHLC, which the backtester needs in price units). Normal
    # normalized values sit within ±250; this ±1000 clamp is invisible to them
    # but stops any future scale surprise (a new pair, an exotic asset) from
    # producing NaN loss the way raw-price MACD did.
    n_ohlc = 4
    features[:, :-n_ohlc] = np.clip(features[:, :-n_ohlc], -1000.0, 1000.0)

    # 5. Build Sequences (Predictive Mode - Lead Shifting)
    # Target is shifted lead_steps (12) into the future
    lead_steps = 12 # Predict 1 hour ahead
    X_list = []
    y_list = []
    
    for i in range(0, len(features) - seq_len - lead_steps, stride):
        target_idx = i + seq_len - 1
        future_idx = target_idx + lead_steps
            
        # 🚀 Phase 11: Golden Dataset - Maximum Refinement
        # 1. Strict Fractal Consensus (100% Agreement)
        is_consensus = consensus_mask[target_idx]
        
        if is_consensus:
             # 2. Return Threshold (Ensure move is large enough to be profit)
             price_now = c_m5[target_idx]
             price_future = c_m5[future_idx]
             ret = abs(price_future - price_now) / price_now
             
             if ret >= threshold:
                  X_list.append(features[i:i+seq_len])
                  if label_mode == "return":
                       # Tradeable target: sign of the realized future move.
                       y_list.append(0.0 if price_future > price_now else 1.0)
                  else:
                       y_list.append(labels[future_idx])
             else:
                  # If move is too small, label as Neutral (optional, or just skip)
                  # Skipping helps train only on "Golden" signals
                  continue
             
    if not X_list:
        logger.warning(f"  No consensus samples found for {file_path.name}")
        return None, None
        
    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)
    
    return X_arr, y_arr


def run_factory(
    seq_len: int = 128,
    stride: int = 2,
    val_split: float = 0.2,
    limit: int = 0,
    symbol: str = "EURUSD",
    market_type: str = "forex",
    reuse_mask_from: str = None,
    prune: bool = False,
    label_mode: str = "trend",
):
    """
    Main factory runner (Sharded Output).
    Isolated by SYMBOL to support Multi-Pair training.

    `reuse_mask_from`: path to a dataset cache dir (or feature_mask.npy)
    whose feature mask should be REUSED instead of recomputing ExtraTrees.
    Critical for reproducible out-of-sample tests — fresh data MUST use
    the exact feature selection (count + order) the model trained on.
    """
    logger.info("=" * 60)
    logger.info(f"  I.A.R.A. v6.11 - Hierarchical Feature Factory ({symbol} / {market_type})")
    logger.info("  Strategy: Fractal Consensus Filtering + Low RAM Mode")
    logger.info("=" * 60)

    processed_dir = DATA_PROCESSED_DIR
    cache_dir = CACHE_DIR / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Preload a feature mask to reuse (reproducible OOS). Supports either a
    # dir containing feature_mask.npy or a direct .npy path.
    preloaded_mask = None
    if reuse_mask_from:
        mp = Path(reuse_mask_from)
        if mp.is_dir():
            mp = mp / "feature_mask.npy"
        if mp.exists():
            preloaded_mask = np.load(mp)
            logger.info(f"♻️  Reusing feature mask from {mp} "
                        f"({int(preloaded_mask.sum())}/{len(preloaded_mask)} feats)")
        else:
            logger.warning(f"reuse_mask_from given but {mp} not found — will recompute")
    
    # Create subdirectories for shards
    train_shard_dir = cache_dir / "train_shards"
    val_shard_dir = cache_dir / "val_shards"
    train_shard_dir.mkdir(exist_ok=True)
    val_shard_dir.mkdir(exist_ok=True)
    
    # Cleanup old shards
    for f in train_shard_dir.glob("*.pt"): f.unlink()
    for f in val_shard_dir.glob("*.pt"): f.unlink()
    
    data_files = list(processed_dir.glob("*.csv")) + list(processed_dir.glob("*.parquet"))
    
    if not data_files:
        logger.error(f"No valid data files found in {processed_dir}")
        return

    logger.info(f"Found {len(data_files)} files for processing")
    
    total_samples = 0
    shard_count = 0
    
    import gc
    
    # Process file by file and save immediately
    samples_per_shard = 50000 
    global_feature_mask = None
    
    for f in data_files:
        # Match files by EXACT stem. Substring match was contaminating:
        # symbol "BTCUSD" also matched BTCUSDFRESH / BTCUSDFRESHM1 (the OOS
        # holdouts), folding future candles into the training set.
        if symbol and f.stem != symbol:
            continue
            
        if f.name.endswith(('.csv', '.parquet')):
            X, y = process_hierarchical(f, seq_len, stride, market_type=market_type, label_mode=label_mode)
            
            if X is not None and len(X) > 0:
                # ♻️ Reuse a preloaded mask (reproducible OOS) before any compute.
                if global_feature_mask is None and preloaded_mask is not None:
                    if len(preloaded_mask) == X.shape[-1]:
                        global_feature_mask = preloaded_mask
                        logger.info("♻️  Applied reused feature mask (skip ExtraTrees)")
                    else:
                        logger.warning(
                            f"reused mask len {len(preloaded_mask)} != feature count "
                            f"{X.shape[-1]} — recomputing (pipeline drift!)")
                # 🌲 Structural Feature Selection (ExtraTrees) — OPT-IN ONLY.
                # It selected a DIFFERENT subset/order per dataset (BTCUSD=9,
                # FRESH=16, FRESHM1=14), silently breaking the train↔OOS feature
                # contract. Default OFF → all 31 features kept in fixed order, so
                # every dataset is byte-compatible without a mask.
                if prune and global_feature_mask is None:
                    from sklearn.ensemble import ExtraTreesClassifier
                    from sklearn.feature_selection import SelectFromModel
                    
                    logger.info("🌲 Training ExtraTreesClassifier for Structural Feature Pruning...")
                    # Flatten sequences for global feature analysis
                    X_2d = X.reshape(-1, X.shape[-1])
                    # Duplicate labels for each timestep in the sequence to match X_2d
                    y_2d = np.repeat(y, X.shape[1])
                    
                    # Subsample for speed
                    if len(X_2d) > 100000:
                        idx = np.random.choice(len(X_2d), 100000, replace=False)
                        X_sub, y_sub = X_2d[idx], y_2d[idx]
                    else:
                        X_sub, y_sub = X_2d, y_2d
                        
                    # Target 0: LONG, 1: SHORT. Ignore neutral (2) for strict structural signal
                    valid_idx = y_sub != 2.0
                    if valid_idx.sum() > 1000: # Ensure we have enough signal samples
                        X_sub, y_sub = X_sub[valid_idx], y_sub[valid_idx]
                        
                    clf = ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    clf.fit(X_sub, y_sub)
                    
                    # Select features: threshold "0.75*mean" to aggressively prune noisy features
                    selector = SelectFromModel(clf, prefit=True, threshold="0.75*mean")
                    global_feature_mask = selector.get_support()

                    logger.info(f"🌲 Pruning complete! Kept {global_feature_mask.sum()} / {len(global_feature_mask)} high-signal features.")
                    # 💾 Persist the mask so retrains + OOS tests reuse the exact
                    # feature contract (count + order). Without this the model's
                    # input meaning silently drifts across datasets.
                    np.save(cache_dir / "feature_mask.npy", global_feature_mask)
                    logger.info(f"💾 Saved feature mask → {cache_dir / 'feature_mask.npy'}")
                
                # Apply structural mask to tensor: [batch, seq_len, features].
                # When pruning is off the mask is None → keep all features.
                if global_feature_mask is not None:
                    X = X[:, :, global_feature_mask]
                
                # 🚀 Phase 11: Shard Splitting logic
                # Instead of one shard per file, we split into chunks of samples_per_shard
                num_file_samples = len(X)
                
                for start_idx in range(0, num_file_samples, samples_per_shard):
                    end_idx = min(start_idx + samples_per_shard, num_file_samples)
                    X_chunk = X[start_idx:end_idx]
                    y_chunk = y[start_idx:end_idx]
                    
                    # Split each chunk into Train/Val
                    val_size = int(len(X_chunk) * val_split)
                    split_idx = len(X_chunk) - val_size
                    
                    X_train, y_train = X_chunk[:split_idx], y_chunk[:split_idx]
                    X_val, y_val = X_chunk[split_idx:], y_chunk[split_idx:]
                    
                    # Save Train Shard
                    if len(X_train) > 0:
                        train_shard_path = train_shard_dir / f"shard_{shard_count}.pt"
                        torch.save({
                            'X': torch.from_numpy(X_train), 
                            'y': torch.from_numpy(y_train)
                        }, train_shard_path)
                    
                    # Save Val Shard
                    if len(X_val) > 0:
                        val_shard_path = val_shard_dir / f"shard_{shard_count}.pt"
                        torch.save({
                            'X': torch.from_numpy(X_val), 
                            'y': torch.from_numpy(y_val)
                        }, val_shard_path)
                    
                    total_samples += len(X_chunk)
                    logger.info(f"  -> Saved Shard {shard_count}: {len(X_train)} train, {len(X_val)} val samples")
                    shard_count += 1
            
            # Explicit GC
            del X, y
            gc.collect()
    
    if total_samples == 0:
        logger.error("No valid sequences generated!")
        return
    
    logger.info(f"Total Processed Samples: {total_samples}")
    
    # Save meta params for inference (using standard paths)
    norm_path = cache_dir / "norm_params.pt"
    
    feature_names = [
        'rsi_norm', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
        'bb_width_norm', 'bb_pct_b_norm', 'atr_norm',
        'stoch_k_norm', 'stoch_d_norm',
        'velocity_norm', 'adx_norm', 'global_sentiment_norm',
        'dist_sma_200', 'dist_sma_20',
        'dist_res_50', 'dist_sup_50',
        'upper_wick_ratio', 'lower_wick_ratio',
        'dist_res_50_v2', 'dist_sup_50_v2',
        'upper_wick_ratio_v2', 'lower_wick_ratio_v2',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'cvd_m5', 'oi_m5', 'fft_m5',
        'open', 'high', 'low', 'close'
    ]
    
    # Filter the exact feature names that survived pruning
    if global_feature_mask is not None:
        if len(global_feature_mask) == len(feature_names):
            feature_names = [name for mask, name in zip(global_feature_mask, feature_names) if mask]
        else:
             # Fallback if there's a length mismatch
            feature_names = [f"feat_{i}" for i in range(global_feature_mask.sum())]
            
    meta = {
        'feature_cols': feature_names,
        'feature_mask': global_feature_mask.tolist() if global_feature_mask is not None else None,
        'seq_len': seq_len,
        'stride': stride,
        'strategy': 'v6.11_hierarchical_sharded',
        'symbol': symbol,
        'market_type': market_type,
    }
    torch.save(meta, norm_path)
    
    logger.info(f"Saved {shard_count} shards to {train_shard_dir}")
    
    # Return Directories instead of single files
    return train_shard_dir, val_shard_dir


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Hierarchical Factory")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples for Lab Mode (0=No limit)")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol to process (e.g. EURUSD, BTCUSD)")
    parser.add_argument("--market-type", type=str, default="forex", choices=["forex", "crypto"],
                        help="Market type (forex or crypto)")
    parser.add_argument("--reuse-mask-from", type=str, default=None,
                        help="Reuse feature mask from this dataset dir / .npy (reproducible OOS)")
    parser.add_argument("--prune", action="store_true", default=False,
                        help="Enable ExtraTrees feature pruning (default OFF — breaks train↔OOS contract)")
    parser.add_argument("--label-mode", type=str, default="trend", choices=["trend", "return"],
                        help="'return' = tradeable sign-of-future-return label; 'trend' = legacy EMA-slope (not tradeable)")
    args = parser.parse_args()

    run_factory(
        seq_len=args.seq_len,
        stride=args.stride,
        val_split=args.val_split,
        limit=args.limit,
        symbol=args.symbol,
        market_type=args.market_type,
        reuse_mask_from=args.reuse_mask_from,
        prune=args.prune,
        label_mode=args.label_mode,
    )


if __name__ == "__main__":
    main()
