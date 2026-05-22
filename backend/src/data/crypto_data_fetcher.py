#!/usr/bin/env python3
"""
I.A.R.A. - Crypto Historical Data Fetcher
==========================================
Downloads cryptocurrency OHLCV data from Alpaca Markets API
and saves it as Parquet files compatible with the feature_factory pipeline.

Usage:
    python src/data/crypto_data_fetcher.py --symbol BTC/USD --days 90
    python src/data/crypto_data_fetcher.py --symbol ETH/USD --days 30 --timeframe 5Min
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CryptoFetcher")

# Output directory (same as existing forex pipeline)
OUTPUT_DIR = Path("data/processed")


def get_timeframe(tf_str: str) -> TimeFrame:
    """Convert string timeframe to Alpaca TimeFrame object."""
    tf_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    return tf_map.get(tf_str, TimeFrame.Minute)


def normalize_symbol_for_filename(symbol: str) -> str:
    """Convert 'BTC/USD' -> 'BTCUSD' for filesystem compatibility."""
    return symbol.replace("/", "")


def fetch_crypto_bars(
    symbol: str,
    days: int = 90,
    timeframe_str: str = "1Min",
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Fetch historical crypto bars from Alpaca in chunks.
    
    Alpaca's crypto data API doesn't require authentication for market data,
    but we use credentials if available for higher rate limits.
    
    Args:
        symbol: Crypto pair (e.g., 'BTC/USD')
        days: Number of days of history to fetch
        timeframe_str: Candlestick timeframe ('1Min', '5Min', etc.)
        chunk_days: Days per API request chunk (to avoid hitting limits)
    
    Returns:
        pd.DataFrame with columns: date, open, high, low, close, volume
    """
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    # CryptoHistoricalDataClient works without auth for market data
    if api_key and secret_key:
        client = CryptoHistoricalDataClient(api_key, secret_key)
    else:
        client = CryptoHistoricalDataClient()
    
    tf = get_timeframe(timeframe_str)
    end = datetime.now()
    start = end - timedelta(days=days)
    
    logger.info(f"📊 Fetching {symbol} from {start.date()} to {end.date()} ({timeframe_str})")
    
    all_frames = []
    current_start = start
    chunk_num = 0
    
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days), end)
        chunk_num += 1
        
        logger.info(f"  Chunk {chunk_num}: {current_start.date()} → {current_end.date()}")
        
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=current_start,
                end=current_end,
                limit=10000,
            )
            bars = client.get_crypto_bars(request)
            df_chunk = bars.df
            
            if not df_chunk.empty:
                all_frames.append(df_chunk)
                logger.info(f"    → {len(df_chunk)} bars fetched")
            else:
                logger.warning(f"    → No data for this period")
                
        except Exception as e:
            logger.error(f"    → Error: {e}")
        
        current_start = current_end
    
    if not all_frames:
        logger.error("No data fetched!")
        return pd.DataFrame()
    
    # Combine all chunks
    df = pd.concat(all_frames)
    
    # Remove duplicates (overlapping chunks)
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by timestamp
    df = df.sort_index()
    
    # Flatten multi-index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()
    
    # Normalize column names for feature_factory compatibility
    df.columns = [c.lower() for c in df.columns]
    
    # Rename 'timestamp' to 'date' for feature_factory compatibility
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    
    # Drop symbol column if present (feature_factory doesn't expect it)
    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])
    
    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "volume"]
    # Handle trade_count, vwap etc. from Alpaca
    if "trade_count" in df.columns:
        df = df.drop(columns=["trade_count"], errors="ignore")
    if "vwap" in df.columns:
        df = df.drop(columns=["vwap"], errors="ignore")
    
    # Keep only the columns we need
    available = [c for c in required if c in df.columns]
    df = df[available]
    
    logger.info(f"✅ Total: {len(df)} bars ({df['date'].min()} → {df['date'].max()})")
    
    return df


def save_to_parquet(df: pd.DataFrame, symbol: str, output_dir: Path = OUTPUT_DIR):
    """Save DataFrame as Parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = normalize_symbol_for_filename(symbol)
    output_path = output_dir / f"{filename}.parquet"
    
    df.to_parquet(output_path, index=False)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"💾 Saved: {output_path} ({size_mb:.2f} MB, {len(df)} rows)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="I.A.R.A. Crypto Data Fetcher")
    parser.add_argument(
        "--symbol", type=str, default="BTC/USD",
        help="Crypto pair (e.g., BTC/USD, ETH/USD)"
    )
    parser.add_argument(
        "--days", type=int, default=90,
        help="Number of days of history to fetch"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1Min",
        choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
        help="Candlestick timeframe"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for Parquet files"
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  I.A.R.A. - Crypto Historical Data Fetcher")
    logger.info("=" * 60)
    
    df = fetch_crypto_bars(
        symbol=args.symbol,
        days=args.days,
        timeframe_str=args.timeframe,
    )
    
    if df.empty:
        logger.error("❌ No data fetched. Check your API credentials and symbol.")
        return
    
    save_to_parquet(df, args.symbol, Path(args.output_dir))
    
    logger.info("🎯 Done! Run feature_factory.py next to prepare training data.")


if __name__ == "__main__":
    main()
